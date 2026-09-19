// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact Qwen4 top-10/QT44/QT53 grouped-prefill lowering.
//!
//! This module is deliberately narrow.  It is reached only after the sealed
//! MoE predicate has admitted the fixed Qwen4 geometry and replicated route.
//! Generic MoE prefill continues to own the k=8 path; no QT53 format is
//! admitted by this module through a representative-dtype fallback.

use crate::families::gemv::WeightRef;
use crate::families::moe::MoePrefillParams;
use crate::types::DispatchError;
use rdna_compute::qwen4::{qwen4_bf16_scaled_add_batched, Qwen4Bf16ScaledAddBatched};
use rdna_compute::{DType, Gpu, GpuTensor};

const QWEN4_TOP_K: usize = 10;
const QWEN4_EXPERTS: usize = 512;
const GROUPED_BLOCK_M: usize = 16;

#[inline]
fn hip<T>(result: Result<T, hip_bridge::HipError>) -> Result<T, DispatchError> {
    result.map_err(|error| DispatchError::Hip(error.to_string()))
}

#[inline]
fn f32_view(source: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    source.sub_offset(offset, len)
}

#[inline]
fn qwen4_geometry(p: &MoePrefillParams<'_>) -> bool {
    p.batch_size > 0
        && p.k_top == QWEN4_TOP_K
        && p.n_exp == QWEN4_EXPERTS
        && p.gate_up_k == 2560
        && p.down_m == 2560
        && p.down_k == 640
        && p.mi == 640
        && p.dtypes.routed_gate_up == DType::MQ4G256V2
        && p.dtypes.routed_down == DType::MQ4G128V2
}

fn require_geometry(p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    if qwen4_geometry(p) {
        Ok(())
    } else {
        Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "qwen4-prefill-shape",
            arch: "",
            quant: "QT44/QT53",
        })
    }
}

/// Prepare the routed expert activation basis.  Qwen4's router/shared
/// projections are BF16 and consume the natural input, while routed QT44
/// gate/up consumes the FWHT basis.
pub(crate) fn input_basis(gpu: &mut Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    require_geometry(p)?;
    hip(gpu.rotate_x_mq_batched(p.x_norm_batch, p.x_rot_batch, p.gate_up_k, p.batch_size))
}

fn batch_projection(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    x: &GpuTensor,
    y: &GpuTensor,
    batch_size: usize,
) -> Result<(), DispatchError> {
    match weight.dtype {
        DType::BF16 => super::qwen4_program::project_bf16_batch(gpu, weight, x, y, batch_size),
        DType::F32 => hip(gpu.gemm_f32_batched(weight.buf, x, y, weight.m, weight.k, batch_size)),
        DType::MQ4G256V2 => {
            hip(gpu.gemm_mq4g256v2(weight.buf, x, y, weight.m, weight.k, batch_size))
        }
        DType::MQ4G128V2 => {
            hip(gpu.gemm_mq4g128v2_batched(weight.buf, x, y, weight.m, weight.k, batch_size))
        }
        _ => Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "qwen4-shared-projection-dtype",
            arch: "",
            quant: "unsupported",
        }),
    }
}

/// Qwen4's router is a row-batched projection even though the generic MoE
/// table intentionally has no BF16 router entry on RDNA3.
pub(crate) fn router_projection(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    batch_projection(
        gpu,
        &p.prelude.router,
        p.x_norm_batch,
        p.prelude.router_logits,
        p.batch_size,
    )
}

/// Shared selector and gate/up are ordinary batched projections.  This is the
/// Qwen4-specific BF16 exception to the generic MoE prefill table: the common
/// family remains fail-closed for QT53 while this exact typed route uses the
/// native BF16 XF32 GEMM launcher.
pub(crate) fn shared_gate_up(gpu: &mut Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("Qwen4 prefill shared weights missing".into()))?;
    let x_selector = match shared.weights.selector.dtype {
        DType::BF16 | DType::F32 => p.x_norm_batch,
        DType::MQ4G256V2 => p.x_rot_batch,
        _ => {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "qwen4-shared-selector-dtype",
                arch: "",
                quant: "unsupported",
            })
        }
    };
    batch_projection(
        gpu,
        &shared.weights.selector,
        x_selector,
        shared.scalar,
        p.batch_size,
    )?;
    let x_gate = match shared.weights.gate.dtype {
        DType::BF16 | DType::F32 => p.x_norm_batch,
        DType::MQ4G256V2 => p.x_rot_batch,
        _ => {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "qwen4-shared-gate-dtype",
                arch: "",
                quant: "unsupported",
            })
        }
    };
    batch_projection(
        gpu,
        &shared.weights.gate,
        x_gate,
        shared.gate_out,
        p.batch_size,
    )?;
    batch_projection(gpu, &shared.weights.up, x_gate, shared.up_out, p.batch_size)?;
    // Qwen4's source arithmetic stores the gate-side BF16 values before the
    // nonlinear route.  Keep the explicit round trips even when the resident
    // projection happens to be F32 so the typed route remains source-bound.
    hip(gpu.bf16_round_trip_f32(p.prelude.router_logits))?;
    hip(gpu.bf16_round_trip_f32(shared.scalar))?;
    hip(gpu.bf16_round_trip_f32(shared.gate_out))?;
    hip(gpu.bf16_round_trip_f32(shared.up_out))
}

/// Apply the shared expert activation after the selector and gate/up projections.
pub(crate) fn shared_activation(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("Qwen4 prefill shared weights missing".into()))?;
    let scalar = f32_view(shared.scalar, 0, p.batch_size);
    #[cfg(feature = "deltanet")]
    {
        hip(gpu.sigmoid_f32(&scalar))?;
    }
    #[cfg(not(feature = "deltanet"))]
    {
        return Err(DispatchError::UnsupportedVariant {
            family: "moe",
            variant: "qwen4-shared-sigmoid-requires-deltanet",
            arch: "",
            quant: "",
        });
    }
    hip(gpu.bf16_round_trip_f32(&scalar))?;
    hip(gpu.silu_mul_f32(shared.gate_out, shared.up_out, shared.rotated))?;
    hip(gpu.bf16_round_trip_f32(shared.rotated))?;
    if shared.weights.down.dtype == DType::MQ4G128V2 {
        hip(gpu.rotate_x_mq_128_v2(
            shared.rotated,
            shared.rotated,
            shared.intermediate,
            p.batch_size,
        ))?;
    }
    Ok(())
}

/// Fold the shared expert after the routed combine.  The residual target is
/// already initialized and contains the routed result; this preserves the
/// source order and one-time sigmoid weighting.
pub(crate) fn shared_down(gpu: &mut Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let shared = p
        .prelude
        .shared
        .as_ref()
        .ok_or_else(|| DispatchError::Hip("Qwen4 prefill shared weights missing".into()))?;
    let down = &shared.weights.down;
    let target = p.routed_out.unwrap_or(p.x_batch);
    let out = f32_view(p.down_expanded, 0, p.batch_size * p.down_m);
    match down.dtype {
        DType::BF16 | DType::F32 | DType::MQ4G256V2 | DType::MQ4G128V2 => {
            // Shared activation already carries the source-required BF16
            // boundary and, for QT53, the G128 basis.  This launch therefore
            // performs the actual dense projection across all prompt rows
            // without reapplying the transform per row.
            batch_projection(gpu, down, shared.rotated, &out, p.batch_size)?;
        }
        _ => {
            return Err(DispatchError::UnsupportedVariant {
                family: "moe",
                variant: "qwen4-shared-down-dtype",
                arch: "",
                quant: "unsupported",
            })
        }
    }
    hip(gpu.bf16_round_trip_f32(&out))?;
    qwen4_bf16_scaled_add_batched(
        gpu,
        &Qwen4Bf16ScaledAddBatched {
            residual: target,
            value: &out,
            scalar: shared.scalar,
            rows: p.batch_size,
            elements: p.down_m,
        },
    )
    .map_err(|error| DispatchError::Hip(error.to_string()))?;
    hip(gpu.bf16_round_trip_f32(target))
}

pub(crate) fn scatter(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let total_slots = p.batch_size * QWEN4_TOP_K;
    hip(gpu.moe_scatter_fused_top10(
        p.topk_indices,
        p.expert_token_counts,
        p.expert_offsets,
        p.sorted_slot_index,
        p.expert_tile_ids,
        p.inverse_perm,
        total_slots,
        QWEN4_EXPERTS,
        grouped_rows,
        GROUPED_BLOCK_M,
    ))
}

pub(crate) fn gate_up(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    if use_path2 {
        hip(gpu.gemm_mq4g256v2_moe_grouped_top10(
            p.expert_gate_up_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.x_rot_batch,
            p.y_gate_up_grouped,
            2 * p.mi,
            p.gate_up_k,
            QWEN4_TOP_K,
            grouped_rows,
            p.batch_size,
        ))
    } else {
        hip(gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
            p.expert_gate_up_ptrs,
            p.topk_indices,
            p.x_rot_batch,
            p.gate_batch,
            p.up_batch,
            2 * p.mi,
            p.gate_up_k,
            p.batch_size,
        ))?;
        let active = p
            .batch_size
            .checked_mul(QWEN4_TOP_K)
            .and_then(|slots| slots.checked_mul(p.mi))
            .ok_or_else(|| DispatchError::Hip("Qwen4 gate/up extent overflows".into()))?;
        hip(gpu.bf16_round_trip_f32(&f32_view(p.gate_batch, 0, active)))?;
        hip(gpu.bf16_round_trip_f32(&f32_view(p.up_batch, 0, active)))?;
        Ok(())
    }
}

pub(crate) fn unscatter(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    hip(gpu.moe_gate_up_unscatter_top10(
        p.y_gate_up_grouped,
        p.sorted_slot_index,
        p.gate_batch,
        p.up_batch,
        p.mi,
        grouped_rows,
        p.batch_size,
    ))?;
    let active = p
        .batch_size
        .checked_mul(QWEN4_TOP_K)
        .and_then(|slots| slots.checked_mul(p.mi))
        .ok_or_else(|| DispatchError::Hip("Qwen4 gate/up extent overflows".into()))?;
    hip(gpu.bf16_round_trip_f32(&f32_view(p.gate_batch, 0, active)))?;
    hip(gpu.bf16_round_trip_f32(&f32_view(p.up_batch, 0, active)))
}

pub(crate) fn activation(gpu: &mut Gpu, p: &MoePrefillParams<'_>) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let total_slots = p.batch_size * QWEN4_TOP_K;
    hip(gpu.silu_mul_f32(p.gate_batch, p.up_batch, p.rot_batch))?;
    hip(gpu.bf16_round_trip_f32(p.rot_batch))?;
    hip(gpu.rotate_x_mq_128_v2(p.rot_batch, p.rot_batch, p.mi, total_slots))
}

pub(crate) fn down(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let total_slots = p.batch_size * QWEN4_TOP_K;
    if use_path2 {
        hip(gpu.gemm_mq4g128v2_moe_grouped_top10(
            p.expert_down_ptrs,
            p.expert_tile_ids,
            p.sorted_slot_index,
            p.rot_batch,
            p.y_down_grouped,
            p.down_m,
            p.down_k,
            1,
            grouped_rows,
            total_slots,
            QWEN4_EXPERTS,
        ))?;
        // The scalar QT53 route rounds each expert output to BF16 before
        // weighted combination.  The grouped kernel only decodes/accumulates
        // its own route row, so perform that same boundary on every active
        // grouped row before the combine stage.
        let grouped = f32_view(p.y_down_grouped, 0, grouped_rows * p.down_m);
        hip(gpu.bf16_round_trip_f32(&grouped))?;
    } else {
        hip(gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
            p.expert_down_ptrs,
            p.topk_indices,
            p.rot_batch,
            p.down_expanded,
            p.down_m,
            p.down_k,
            p.batch_size,
            QWEN4_EXPERTS,
        ))?;
        let expanded = f32_view(p.down_expanded, 0, total_slots * p.down_m);
        hip(gpu.bf16_round_trip_f32(&expanded))?;
    }
    Ok(())
}

pub(crate) fn combine(
    gpu: &mut Gpu,
    p: &MoePrefillParams<'_>,
    use_path2: bool,
    grouped_rows: usize,
) -> Result<(), DispatchError> {
    require_geometry(p)?;
    let target = p.routed_out.unwrap_or(p.x_batch);
    if use_path2 {
        hip(gpu.moe_down_combine_grouped_top10(
            p.y_down_grouped,
            p.inverse_perm,
            p.topk_indices,
            p.topk_weights,
            target,
            p.down_m,
            grouped_rows,
            p.batch_size,
        ))?;
    } else {
        let expanded = f32_view(p.down_expanded, 0, p.batch_size * QWEN4_TOP_K * p.down_m);
        hip(gpu.moe_down_combine_top10_batched(
            &expanded,
            p.topk_indices,
            p.topk_weights,
            target,
            p.down_m,
            p.batch_size,
        ))?;
    }
    hip(gpu.bf16_round_trip_f32(target))
}
