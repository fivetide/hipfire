// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Private executable programs for sealed MoE calls.
//!
//! A [`SealedMoeCall`](super::sealed_moe::SealedMoeCall) is the public grammar
//! boundary, not an arithmetic implementation.  This module lowers each
//! validated call into a fixed-capacity sequence of typed stages and runs the
//! stages in order.  Stage descriptors borrow model-owned tensors and expert
//! metadata; the lowering itself never allocates.

use crate::context::DispatchCtx;
use crate::families::moe::{MoeParams, MoePrefillParams, MoePrefillResolution, MoeResolution};
use crate::pipeline::sealed_moe::{MoeProtocol, MoeRouterInput, SealedMoeCall};
use crate::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor};

const MAX_DECODE_STAGES: usize = 9;
const MAX_PREFILL_STAGES: usize = 8;

fn router_shared_fuse_allowed(
    skip_routing: bool,
    use_gpu_topk: bool,
    exact_wave64_router: bool,
    batch_size: usize,
    skip_shared: bool,
    smi: usize,
    shared_down_dtype: rdna_compute::DType,
    shared_down_has_awq: bool,
    env_enabled: bool,
) -> bool {
    !skip_routing
        && use_gpu_topk
        && exact_wave64_router
        && batch_size == 1
        && !skip_shared
        && smi == 512
        && shared_down_dtype == rdna_compute::DType::MQ4G256
        && !shared_down_has_awq
        && env_enabled
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeStage {
    InputBasis,
    GateSide,
    RouteGpu,
    RouteCpu,
    SharedDown,
    RoutedGateUp,
    RoutedActivation,
    RoutedDown,
    MutationFence,
    Combine,
    CpuExperts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefillStage {
    Scatter,
    InputBasis,
    GateUp,
    GateUpUnscatter,
    Activation,
    Down,
    MutationFence,
    Combine,
}

/// Borrowed decode program.  The array is deliberately fixed: stage count is
/// bounded by the grammar and no token-time plan vector can grow or allocate.
pub(super) struct DecodeProgram<'a> {
    params: &'a MoeParams<'a>,
    resolution: MoeResolution,
    stages: [DecodeStage; MAX_DECODE_STAGES],
    len: usize,
    target: &'a GpuTensor,
    activation_input: &'a GpuTensor,
    router_shared_fuse: bool,
    exact_wave64_router: bool,
    wave64_router: bool,
    down_last_combine: bool,
    ninepath_d3: bool,
    ninepath_d4: bool,
    ninepath_mq3l: bool,
    ninepath_mq4v2: bool,
    ninepath_mq6v2: bool,
}

impl<'a> DecodeProgram<'a> {
    fn push(&mut self, stage: DecodeStage) {
        debug_assert!(self.len < self.stages.len());
        self.stages[self.len] = stage;
        self.len += 1;
    }

    fn iter(&self) -> impl Iterator<Item = DecodeStage> + '_ {
        self.stages[..self.len].iter().copied()
    }

    fn lower(
        ctx: &DispatchCtx,
        gpu: &Gpu,
        call: &'a SealedMoeCall<'a>,
    ) -> Result<Self, DispatchError> {
        if call.protocol() != MoeProtocol::IndexedDecode {
            return Err(DispatchError::Hip(
                "sealed moe: decode program received a non-decode call".into(),
            ));
        }
        let params = call.decode_params().ok_or_else(|| {
            DispatchError::Hip("sealed moe: decode call has no decode operands".into())
        })?;
        super::check_moe_decode_batch_size(params.batch_size)?;

        // Keep resolution and every pre-launch refusal in this lowering pass.
        // No stage can run until all route and ownership choices are known.
        let resolution = MoeResolution::resolve_arch(&params.dtypes, params.k, ctx.arch.has_wmma());
        super::check_moe_decode_supported(
            resolution.use_gpu_topk,
            params.k,
            params.n_exp,
            !params.routed_experts.is_empty(),
        )?;
        if params.ep_mode == crate::families::moe::MoeEpMode::RootRoutedPartial {
            if !resolution.use_gpu_topk {
                return Err(DispatchError::Hip(
                    "root-routed EP decode requires the GPU top-K path; the CPU-top-K fallback is rejected"
                        .into(),
                ));
            }
            if params.defer_routed_combine {
                return Err(DispatchError::Hip(
                    "root-routed EP decode requires defer_routed_combine=false (weighted combine into the partial)"
                        .into(),
                ));
            }
            if params.routed_out.is_none() {
                return Err(DispatchError::Hip(
                    "root-routed EP decode requires routed_out=Some (the zeroed partial)".into(),
                ));
            }
        }

        let skip_routing = call.router_input() == MoeRouterInput::PrecomputedSoftmaxTopK;
        if !resolution.use_gpu_topk {
            // The host top-k route is intentionally a complete, explicit
            // program arm. Refuse all cases that cannot execute before the
            // gate-side stage has a chance to mutate a tensor.
            if skip_routing {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "cpu-topk-fallback-needs-gpu-route-for-precomputed-input",
                    arch: "",
                    quant: "",
                });
            }
            if params.routed_out.is_some() {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "ep-routed-out-unsupported-in-cpu-topk-fallback",
                    arch: "",
                    quant: "",
                });
            }
            if params.routed_experts.is_empty() {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "cpu-topk-fallback-needs-resident-experts",
                    arch: "",
                    quant: "",
                });
            }
            if gpu.graphs.replay.capturing.is_some() {
                return Err(DispatchError::UnsupportedVariant {
                    family: "moe",
                    variant: "cpu-topk-fallback-not-capture-safe(set HIPFIRE_GRAPH_MOE=0)",
                    arch: "",
                    quant: "",
                });
            }
        }
        if resolution.needs_x_rot_local && !params.x_rot_prerotated {
            // The sealer guarantees the Paro sidecar when this predicate is
            // true; keep the check explicit at the lowering boundary anyway.
            if resolution.routed_indexable_paro && params.routed_gate_up_paro.is_none() {
                return Err(DispatchError::Hip(
                    "sealed moe: Paro decode program has no gate-up sidecar".into(),
                ));
            }
        }

        let target = params.routed_out.unwrap_or(params.x_residual);
        let activation_input = if resolution.needs_x_rot_local {
            params.x_rot_local
        } else {
            params.x_norm
        };

        let gfx1100_router_mode = hipfire_config::developer_var("HIPFIRE_GFX1100_ROUTER_W64").ok();
        let exact_wave64_router = params.n_exp == 256
            && ((ctx.arch.is_gfx1100()
                && !matches!(gfx1100_router_mode.as_deref(), Some("0" | "approx")))
                || ctx.arch.is_gfx1151());
        let wave64_router = (ctx.arch.is_gfx1201()
            && hipfire_config::developer_var("HIPFIRE_GFX1201_ROUTER_W64").as_deref() != Ok("0"))
            || (ctx.arch.is_gfx1100()
                && params.n_exp == 256
                && gfx1100_router_mode.as_deref() == Some("approx"));
        static ROUTER_SHARED_FUSE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            hipfire_config::developer_var("HIPFIRE_MOE_ROUTER_SHARED_FUSE").as_deref() == Ok("1")
        });
        let router_shared_fuse = router_shared_fuse_allowed(
            skip_routing,
            resolution.use_gpu_topk,
            exact_wave64_router,
            params.batch_size,
            params.skip_shared,
            params.smi,
            params.shared_down_w.dtype,
            params.shared_down_w.awq_scale.is_some(),
            *ROUTER_SHARED_FUSE,
        );

        static DOWN_LAST_COMBINE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let down_last_combine = ctx.arch.is_gfx1100()
            && params.batch_size == 1
            && params.k == 8
            && params.expert_dtype_tags.is_none()
            && params.dtypes.routed_down == rdna_compute::DType::MQ4G256
            && *DOWN_LAST_COMBINE.get_or_init(|| {
                hipfire_config::developer_var("HIPFIRE_MOE_DOWN_LAST_COMBINE").as_deref() == Ok("1")
            });

        static MOE_NINEPATH: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            hipfire_config::developer_var("HIPFIRE_MOE_NINEPATH").unwrap_or_default()
        });
        let ninepath_mode = MOE_NINEPATH.as_str();
        let ninepath_shape_ok = params.k == 8
            && params.batch_size == 1
            && params.hidden <= 2048
            && params.mi == 512
            && params.expert_dtype_tags.is_none()
            && params.expert_down_awq_ptrs.is_none()
            && !params.defer_routed_combine;
        let ninepath_hfq4 = ninepath_shape_ok
            && params.dtypes.routed_gate_up == rdna_compute::DType::MQ4G256
            && params.dtypes.routed_down == rdna_compute::DType::MQ4G256;
        let ninepath_mq3l = ninepath_shape_ok
            && params.dtypes.routed_gate_up == rdna_compute::DType::MQ2G256Lloyd
            && params.dtypes.routed_down == rdna_compute::DType::MQ3G256Lloyd;
        let ninepath_mq4v2 = ninepath_shape_ok
            && params.dtypes.routed_gate_up == rdna_compute::DType::MQ4G256V2
            && params.dtypes.routed_down == rdna_compute::DType::MQ4G256V2;
        let ninepath_mq6v2 = ninepath_shape_ok
            && params.dtypes.routed_gate_up == rdna_compute::DType::MQ6G256V2
            && params.dtypes.routed_down == rdna_compute::DType::MQ6G256V2;
        let ninepath_eligible = ninepath_hfq4 || ninepath_mq3l || ninepath_mq4v2 || ninepath_mq6v2;
        let ninepath_d3 = ninepath_hfq4 && matches!(ninepath_mode, "1" | "d3" | "on");
        let ninepath_d4 = ninepath_eligible && !matches!(ninepath_mode, "0" | "off" | "d3");

        let mut program = Self {
            params,
            resolution,
            stages: [DecodeStage::InputBasis; MAX_DECODE_STAGES],
            len: 0,
            target,
            activation_input,
            router_shared_fuse,
            exact_wave64_router,
            wave64_router,
            down_last_combine,
            ninepath_d3,
            ninepath_d4,
            ninepath_mq3l,
            ninepath_mq4v2,
            ninepath_mq6v2,
        };

        if resolution.needs_x_rot_local {
            program.push(DecodeStage::InputBasis);
        }
        if !skip_routing {
            program.push(DecodeStage::GateSide);
            program.push(if resolution.use_gpu_topk {
                DecodeStage::RouteGpu
            } else {
                DecodeStage::RouteCpu
            });
        }
        if !params.skip_shared {
            program.push(DecodeStage::SharedDown);
        }
        if resolution.use_gpu_topk {
            program.push(DecodeStage::RoutedGateUp);
            program.push(DecodeStage::RoutedActivation);
            program.push(DecodeStage::RoutedDown);
            program.push(DecodeStage::MutationFence);
            let down_self_combines = program.down_last_combine
                || !crate::families::moe::moe_down_writes_expanded(
                    params.dtypes.routed_down,
                    params.expert_dtype_tags.is_some(),
                );
            if !program.ninepath_d4 && !down_self_combines && !params.defer_routed_combine {
                program.push(DecodeStage::Combine);
            }
        } else {
            // CPU route is intentionally split from route production and
            // shared-down work; this is not a call-through to a whole decoder.
            program.push(DecodeStage::CpuExperts);
        }

        Ok(program)
    }
}

pub(super) struct DecodeRuntime {
    pub(super) cpu_indices: Option<Vec<usize>>,
    pub(super) cpu_weights: Option<Vec<f32>>,
}
pub(super) fn execute_decode(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    call: &SealedMoeCall<'_>,
) -> Result<(), DispatchError> {
    // The call lifetime is strictly borrowed for this invocation; the program
    // owns no resource and cannot outlive the sealed call.
    let program = DecodeProgram::lower(ctx, gpu, call)?;
    let p = program.params;
    let shared_gate = super::slice_moe_f32_view(p.gate_buf, 0, p.smi);
    let shared_up = super::slice_moe_f32_view(p.up_buf, 0, p.smi);
    let mut runtime = DecodeRuntime {
        cpu_indices: None,
        cpu_weights: None,
    };

    for stage in program.iter() {
        match stage {
            DecodeStage::InputBasis => super::decode_input_basis_stage(gpu, p, program.resolution)?,
            DecodeStage::GateSide => super::decode_gate_side_stage(
                ctx,
                gpu,
                p,
                program.resolution,
                if program.resolution.needs_x_rot_local {
                    Some(program.activation_input)
                } else {
                    None
                },
                &shared_gate,
                &shared_up,
            )?,
            DecodeStage::RouteGpu => super::decode_route_gpu_stage(
                gpu,
                p,
                &shared_gate,
                &shared_up,
                program.router_shared_fuse,
                program.exact_wave64_router,
                program.wave64_router,
            )?,
            DecodeStage::RouteCpu => super::decode_route_cpu_stage(gpu, p, &mut runtime)?,
            DecodeStage::SharedDown => super::decode_shared_down_stage(
                ctx,
                gpu,
                p,
                &shared_gate,
                &shared_up,
                program.target,
                program.router_shared_fuse,
            )?,
            DecodeStage::RoutedGateUp => super::decode_gate_up_stage(
                gpu,
                p,
                program.resolution,
                program.activation_input,
                program.ninepath_d3,
            )?,
            DecodeStage::RoutedActivation => {
                super::decode_activation_stage(gpu, p, program.resolution)?
            }
            DecodeStage::RoutedDown => super::decode_down_stage(
                gpu,
                p,
                program.resolution,
                program.target,
                program.ninepath_d4,
                program.ninepath_mq3l,
                program.ninepath_mq4v2,
                program.ninepath_mq6v2,
                program.down_last_combine,
            )?,
            DecodeStage::MutationFence => super::decode_mutation_fence_stage()?,
            DecodeStage::Combine => super::decode_combine_stage(gpu, p, program.target)?,
            DecodeStage::CpuExperts => {
                let indices = runtime.cpu_indices.as_deref().ok_or_else(|| {
                    DispatchError::Hip("sealed moe: CPU route did not produce indices".into())
                })?;
                let weights = runtime.cpu_weights.as_deref().ok_or_else(|| {
                    DispatchError::Hip("sealed moe: CPU route did not produce weights".into())
                })?;
                super::decode_cpu_experts_stage(ctx, gpu, p, indices, weights)?;
            }
        }
    }
    Ok(())
}

pub(super) struct PrefillProgram<'a> {
    params: &'a MoePrefillParams<'a>,
    resolution: MoePrefillResolution,
    stages: [PrefillStage; MAX_PREFILL_STAGES],
    len: usize,
    target: &'a GpuTensor,
    path2_m_total: usize,
    force_mq4_grouped_fp16: bool,
}

impl<'a> PrefillProgram<'a> {
    fn push(&mut self, stage: PrefillStage) {
        debug_assert!(self.len < self.stages.len());
        self.stages[self.len] = stage;
        self.len += 1;
    }

    fn iter(&self) -> impl Iterator<Item = PrefillStage> + '_ {
        self.stages[..self.len].iter().copied()
    }

    fn lower(ctx: &DispatchCtx, call: &'a SealedMoeCall<'a>) -> Result<Self, DispatchError> {
        if call.protocol() != MoeProtocol::GroupedPrefill {
            return Err(DispatchError::Hip(
                "sealed moe: prefill program received a non-prefill call".into(),
            ));
        }
        let params = call.prefill_params().ok_or_else(|| {
            DispatchError::Hip("sealed moe: prefill call has no prefill operands".into())
        })?;
        let resolution = MoePrefillResolution::resolve(&params.dtypes, &ctx.arch, &ctx.flags);
        let use_path2 = resolution.use_path2;
        let paro_mode = resolution.paro_mode;
        let down_path0 = resolution.down_path0;
        let force_mq4_grouped_fp16 =
            resolution.force_mq4_grouped_fp16 || params.force_mq4_grouped_fp16;
        let total_slots = params
            .batch_size
            .checked_mul(params.k_top)
            .ok_or_else(|| DispatchError::Hip("sealed moe: prefill slot count overflows".into()))?;
        let path2_m_total = if use_path2 { params.m_total_max } else { 0 };
        let target = params.routed_out.unwrap_or(params.x_batch);
        let mut program = Self {
            params,
            resolution,
            stages: [PrefillStage::Scatter; MAX_PREFILL_STAGES],
            len: 0,
            target,
            path2_m_total,
            force_mq4_grouped_fp16,
        };
        if use_path2 {
            program.push(PrefillStage::Scatter);
        }
        if paro_mode {
            program.push(PrefillStage::InputBasis);
        }
        program.push(PrefillStage::GateUp);
        if use_path2 {
            program.push(PrefillStage::GateUpUnscatter);
        }
        program.push(PrefillStage::Activation);
        program.push(PrefillStage::Down);
        program.push(PrefillStage::MutationFence);
        if use_path2 || !down_path0 {
            program.push(PrefillStage::Combine);
        }
        let _ = total_slots;
        Ok(program)
    }
}

pub(super) fn execute_prefill(
    ctx: &DispatchCtx,
    gpu: &mut Gpu,
    call: &SealedMoeCall<'_>,
) -> Result<(), DispatchError> {
    let program = PrefillProgram::lower(ctx, call)?;
    let p = program.params;
    let total_slots = p
        .batch_size
        .checked_mul(p.k_top)
        .ok_or_else(|| DispatchError::Hip("sealed moe: prefill slot count overflows".into()))?;
    if hipfire_config::developer_var("HIPFIRE_MOE_PREFILL_TRACE")
        .ok()
        .as_deref()
        == Some("1")
    {
        eprintln!(
            "[moe-prefill] arch={} shared=({:?},{:?},{:?},{:?}) routed=({:?},{:?}) \
             path2={} force_mq4_fp16={} grouped_i8={:?}",
            ctx.arch.arch(),
            p.dtypes.shared_gate,
            p.dtypes.shared_expert_gate,
            p.dtypes.shared_expert_up,
            p.dtypes.shared_expert_down,
            p.dtypes.routed_gate_up,
            p.dtypes.routed_down,
            program.resolution.use_path2,
            program.force_mq4_grouped_fp16,
            ctx.flags.moe_grouped_i8,
        );
    }
    for stage in program.iter() {
        match stage {
            PrefillStage::Scatter => super::prefill_scatter_stage(gpu, p, program.path2_m_total)?,
            PrefillStage::InputBasis => super::prefill_input_basis_stage(gpu, p)?,
            PrefillStage::GateUp => super::prefill_gate_up_stage(
                gpu,
                p,
                &program.resolution,
                program.path2_m_total,
                program.force_mq4_grouped_fp16,
            )?,
            PrefillStage::GateUpUnscatter => {
                super::prefill_gate_up_unscatter_stage(gpu, p, program.path2_m_total)?
            }
            PrefillStage::Activation => {
                super::prefill_activation_stage(gpu, p, &program.resolution, total_slots)?
            }
            PrefillStage::Down => super::prefill_down_stage(
                gpu,
                p,
                &program.resolution,
                program.path2_m_total,
                total_slots,
                program.force_mq4_grouped_fp16,
            )?,
            PrefillStage::MutationFence => super::prefill_mutation_fence_stage()?,
            PrefillStage::Combine => {
                super::prefill_combine_stage(gpu, p, &program.resolution, program.target)?
            }
        }
    }
    Ok(())
}

// Stage lowering intentionally keeps the public sealed call as its only input.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_topk_disables_router_shared_fuse() {
        assert!(!router_shared_fuse_allowed(
            false,
            false,
            true,
            1,
            false,
            512,
            rdna_compute::DType::MQ4G256,
            false,
            true,
        ));
    }
}
