// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — typed Qwen4 sealed MoE execution.

//! Typed, fail-closed MoE programs for Qwen4.
//!
//! This module is deliberately separate from [`super::moe`].  The incumbent
//! family is a compatibility path for older k=8 models; this family describes
//! the Qwen4 grammar and never widens the old route by accepting a different
//! `k` at an existing call site.
//!
//! A program is sealed before it receives any GPU work.  The seal covers the
//! model geometry, projection wire formats, route/permutation provenance,
//! placement/collective schedule, execution epoch, and all scratch capacities.
//! Execution validates every typed operand before the first launch.

use rdna_compute::{DType, Gpu, GpuTensor};

use crate::types::DispatchError;

pub const QWEN4_ARCH_ID: u16 = 16;
pub const QWEN4_N_EXPERTS: usize = 512;
pub const QWEN4_TOP_K: usize = 10;
pub const QWEN4_HIDDEN: usize = 2560;
pub const QWEN4_INTERMEDIATE: usize = 640;
pub const QWEN4_BLOCK_M: usize = 16;

/// Which fixed execution grammar a sealed program uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoePhase {
    IndexedDecode,
    GroupedPrefill,
}

/// Storage/layout contract for a projection.  The layout is part of the seal;
/// equal shapes with a different wire format are not interchangeable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeLayout {
    /// Device pointer table indexing one expert matrix per route slot.
    IndexedExpertRows,
    /// Device pointer table plus scatter-generated expert tiles.
    GroupedExpertTiles,
    /// Token-major dense rows (`[tokens, width]`).
    TokenMajor,
    /// Route-slot-major rows (`[tokens * top_k, width]`).
    SlotMajor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionSpec {
    pub dtype: DType,
    pub layout: MoeLayout,
    /// Logical output rows in one expert matrix.
    pub rows: usize,
    /// Logical input columns in one expert matrix.
    pub cols: usize,
}

impl ProjectionSpec {
    pub const fn qwen4_gate_up(layout: MoeLayout) -> Self {
        Self {
            dtype: DType::MQ4G256V2,
            layout,
            rows: QWEN4_INTERMEDIATE * 2,
            cols: QWEN4_HIDDEN,
        }
    }

    pub const fn qwen4_down(layout: MoeLayout) -> Self {
        Self {
            // Q8F16/qt3 is carried by the existing Q8_0 wire tag.  The
            // activation side remains F32; this is not a Qwen35 MQ3 route.
            dtype: DType::Q8_0,
            layout,
            rows: QWEN4_HIDDEN,
            cols: QWEN4_INTERMEDIATE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouterPolicy {
    pub logits_dtype: DType,
    pub normalize_topk_prob: bool,
}

impl Default for RouterPolicy {
    fn default() -> Self {
        Self {
            logits_dtype: DType::F32,
            normalize_topk_prob: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoePlacement {
    /// One GPU.  This schedule must not carry a synthetic collective.
    Single,
    /// A validated mesh plan.  Physical execution is still gated by the
    /// caller; this value records the plan identity instead of re-deriving it
    /// from raw topology flags.
    Mesh { plan_id: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollectiveSchedule {
    None,
    Ordered { schedule_id: u64, operations: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RouteIdentity(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PermutationIdentity(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionEpoch(pub u64);

/// Scratch/capacity limits are sealed alongside logical dimensions.  A launch
/// may consume less than a capacity, but it may never launch a grid whose
/// output exceeds one of these bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeCapacities {
    pub max_tokens: usize,
    pub route_slots: usize,
    pub expanded_rows: usize,
    pub grouped_rows: usize,
    pub block_m: usize,
}

impl MoeCapacities {
    pub fn for_tokens(tokens: usize, grouped: bool) -> Result<Self, DispatchError> {
        let slots = checked_mul(tokens, QWEN4_TOP_K, "token count × top-k")?;
        let grouped_rows = if grouped {
            let live_experts = slots.min(QWEN4_N_EXPERTS);
            let padding = checked_mul(live_experts, QWEN4_BLOCK_M - 1, "grouped expert padding")?;
            checked_round_up(
                slots
                    .checked_add(padding)
                    .ok_or_else(|| invalid("capacity-overflow", "grouped rows"))?,
                QWEN4_BLOCK_M,
                "grouped rows",
            )?
        } else {
            0
        };
        Ok(Self {
            max_tokens: tokens,
            route_slots: slots,
            expanded_rows: slots,
            grouped_rows,
            block_m: QWEN4_BLOCK_M,
        })
    }
}

/// Complete immutable signature for one Qwen4 MoE program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeSignature {
    pub phase: MoePhase,
    pub n_experts: usize,
    pub top_k: usize,
    pub tokens: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub gate_up: ProjectionSpec,
    pub down: ProjectionSpec,
    pub router: RouterPolicy,
    pub placement: MoePlacement,
    pub collectives: CollectiveSchedule,
    pub route: RouteIdentity,
    pub permutation: PermutationIdentity,
    pub epoch: ExecutionEpoch,
    pub capacities: MoeCapacities,
}

impl MoeSignature {
    /// Canonical Qwen4 indexed/decode signature.  Callers may build a struct
    /// directly when the loader has a different bounded scratch budget, but
    /// [`MoeProgram::seal_indexed`] still checks every field.
    pub fn qwen4_indexed(
        tokens: usize,
        route: RouteIdentity,
        permutation: PermutationIdentity,
        epoch: ExecutionEpoch,
        placement: MoePlacement,
        collectives: CollectiveSchedule,
    ) -> Result<Self, DispatchError> {
        Ok(Self {
            phase: MoePhase::IndexedDecode,
            n_experts: QWEN4_N_EXPERTS,
            top_k: QWEN4_TOP_K,
            tokens,
            hidden: QWEN4_HIDDEN,
            intermediate: QWEN4_INTERMEDIATE,
            gate_up: ProjectionSpec::qwen4_gate_up(MoeLayout::IndexedExpertRows),
            down: ProjectionSpec::qwen4_down(MoeLayout::IndexedExpertRows),
            router: RouterPolicy::default(),
            placement,
            collectives,
            route,
            permutation,
            epoch,
            capacities: MoeCapacities::for_tokens(tokens, false)?,
        })
    }

    pub fn qwen4_grouped(
        tokens: usize,
        route: RouteIdentity,
        permutation: PermutationIdentity,
        epoch: ExecutionEpoch,
        placement: MoePlacement,
        collectives: CollectiveSchedule,
    ) -> Result<Self, DispatchError> {
        Ok(Self {
            phase: MoePhase::GroupedPrefill,
            n_experts: QWEN4_N_EXPERTS,
            top_k: QWEN4_TOP_K,
            tokens,
            hidden: QWEN4_HIDDEN,
            intermediate: QWEN4_INTERMEDIATE,
            gate_up: ProjectionSpec::qwen4_gate_up(MoeLayout::GroupedExpertTiles),
            down: ProjectionSpec::qwen4_down(MoeLayout::GroupedExpertTiles),
            router: RouterPolicy::default(),
            placement,
            collectives,
            route,
            permutation,
            epoch,
            capacities: MoeCapacities::for_tokens(tokens, true)?,
        })
    }
}

/// Fixed operation grammar.  Shared expert work is explicit and appears once
/// after the routed combine; it is not hidden in an atomic down path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeOp {
    Route,
    Scatter,
    GateUp,
    Unscatter,
    Activation,
    ExpandedDown,
    Combine,
    SharedExpertGate,
    SharedExpertGateUp,
    SharedExpertDown,
    SharedExpertAdd,
}

const INDEXED_OPS: &[MoeOp] = &[
    MoeOp::Route,
    MoeOp::GateUp,
    MoeOp::Activation,
    MoeOp::ExpandedDown,
    MoeOp::Combine,
    MoeOp::SharedExpertGate,
    MoeOp::SharedExpertGateUp,
    MoeOp::SharedExpertDown,
    MoeOp::SharedExpertAdd,
];

const GROUPED_OPS: &[MoeOp] = &[
    MoeOp::Route,
    MoeOp::Scatter,
    MoeOp::GateUp,
    MoeOp::Unscatter,
    MoeOp::Activation,
    MoeOp::ExpandedDown,
    MoeOp::Combine,
    MoeOp::SharedExpertGate,
    MoeOp::SharedExpertGateUp,
    MoeOp::SharedExpertDown,
    MoeOp::SharedExpertAdd,
];

/// Sealed execution object.  There is no constructor from arbitrary
/// instructions; only the two fixed grammars can produce this type.
#[derive(Clone, Copy, Debug)]
pub struct SealedMoeProgram {
    signature: MoeSignature,
    operations: &'static [MoeOp],
}

impl SealedMoeProgram {
    pub fn signature(&self) -> &MoeSignature {
        &self.signature
    }

    pub fn phase(&self) -> MoePhase {
        self.signature.phase
    }

    pub fn operations(&self) -> &'static [MoeOp] {
        self.operations
    }
}

pub struct MoeProgram;

impl MoeProgram {
    pub fn seal_indexed(signature: MoeSignature) -> Result<SealedMoeProgram, DispatchError> {
        validate_signature(&signature, MoePhase::IndexedDecode)?;
        Ok(SealedMoeProgram {
            signature,
            operations: INDEXED_OPS,
        })
    }

    pub fn seal_grouped(signature: MoeSignature) -> Result<SealedMoeProgram, DispatchError> {
        validate_signature(&signature, MoePhase::GroupedPrefill)?;
        Ok(SealedMoeProgram {
            signature,
            operations: GROUPED_OPS,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorRole {
    RouterLogits,
    TopKIndices,
    TopKWeights,
    GateUpExpertPointers,
    DownExpertPointers,
    GateInput,
    Gate,
    Up,
    Activation,
    ExpandedDown,
    Residual,
    SharedGateWeight,
    SharedGateProjWeight,
    SharedUpProjWeight,
    SharedDownProjWeight,
    SharedGateLogits,
    SharedGateProj,
    SharedUpProj,
    SharedActivation,
    SharedOutput,
    ExpertTokenCounts,
    ExpertOffsets,
    SortedSlotIndex,
    ExpertTileIds,
    InversePermutation,
    GroupedGateUp,
    GroupedDown,
}

/// The shared Qwen4 expert is part of the sealed program, not a caller-owned
/// precomputed contribution.  All four weights are native BF16 so the
/// projection kernels widen the exact checkpoint values to an FP32
/// accumulator.  Scratch outputs remain F32 and are consumed in order.
pub struct SharedExpertOperands<'a> {
    pub gate_weight: MoeTensorHandle<'a>,
    pub gate_proj_weight: MoeTensorHandle<'a>,
    pub up_proj_weight: MoeTensorHandle<'a>,
    pub down_proj_weight: MoeTensorHandle<'a>,
    pub gate_logits: MoeTensorHandle<'a>,
    pub gate_proj: MoeTensorHandle<'a>,
    pub up_proj: MoeTensorHandle<'a>,
    pub activation: MoeTensorHandle<'a>,
    pub output: MoeTensorHandle<'a>,
}

/// A handle carries the semantic role assigned at construction; the executor
/// never receives a tensor name or consults `WeightStore`.
#[derive(Clone, Copy)]
pub struct MoeTensorHandle<'a> {
    role: TensorRole,
    tensor: &'a GpuTensor,
}

impl<'a> MoeTensorHandle<'a> {
    pub fn new(tensor: &'a GpuTensor, role: TensorRole) -> Self {
        Self { role, tensor }
    }

    pub fn role(self) -> TensorRole {
        self.role
    }

    pub fn tensor(self) -> &'a GpuTensor {
        self.tensor
    }
}
impl<'a> SharedExpertOperands<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gate_weight: &'a GpuTensor,
        gate_proj_weight: &'a GpuTensor,
        up_proj_weight: &'a GpuTensor,
        down_proj_weight: &'a GpuTensor,
        gate_logits: &'a GpuTensor,
        gate_proj: &'a GpuTensor,
        up_proj: &'a GpuTensor,
        activation: &'a GpuTensor,
        output: &'a GpuTensor,
    ) -> Self {
        Self {
            gate_weight: MoeTensorHandle::new(gate_weight, TensorRole::SharedGateWeight),
            gate_proj_weight: MoeTensorHandle::new(
                gate_proj_weight,
                TensorRole::SharedGateProjWeight,
            ),
            up_proj_weight: MoeTensorHandle::new(up_proj_weight, TensorRole::SharedUpProjWeight),
            down_proj_weight: MoeTensorHandle::new(
                down_proj_weight,
                TensorRole::SharedDownProjWeight,
            ),
            gate_logits: MoeTensorHandle::new(gate_logits, TensorRole::SharedGateLogits),
            gate_proj: MoeTensorHandle::new(gate_proj, TensorRole::SharedGateProj),
            up_proj: MoeTensorHandle::new(up_proj, TensorRole::SharedUpProj),
            activation: MoeTensorHandle::new(activation, TensorRole::SharedActivation),
            output: MoeTensorHandle::new(output, TensorRole::SharedOutput),
        }
    }
}

pub struct IndexedDecodeOperands<'a> {
    pub route_logits: MoeTensorHandle<'a>,
    pub topk_indices: MoeTensorHandle<'a>,
    pub topk_weights: MoeTensorHandle<'a>,
    pub gate_up_expert_ptrs: MoeTensorHandle<'a>,
    pub down_expert_ptrs: MoeTensorHandle<'a>,
    pub gate_input: MoeTensorHandle<'a>,
    pub gate: MoeTensorHandle<'a>,
    pub up: MoeTensorHandle<'a>,
    pub activation: MoeTensorHandle<'a>,
    pub expanded_down: MoeTensorHandle<'a>,
    pub residual: MoeTensorHandle<'a>,
    pub shared_expert: SharedExpertOperands<'a>,
}

impl<'a> IndexedDecodeOperands<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        route_logits: &'a GpuTensor,
        topk_indices: &'a GpuTensor,
        topk_weights: &'a GpuTensor,
        gate_up_expert_ptrs: &'a GpuTensor,
        down_expert_ptrs: &'a GpuTensor,
        gate_input: &'a GpuTensor,
        gate: &'a GpuTensor,
        up: &'a GpuTensor,
        activation: &'a GpuTensor,
        expanded_down: &'a GpuTensor,
        residual: &'a GpuTensor,
        shared_expert: SharedExpertOperands<'a>,
    ) -> Self {
        Self {
            route_logits: MoeTensorHandle::new(route_logits, TensorRole::RouterLogits),
            topk_indices: MoeTensorHandle::new(topk_indices, TensorRole::TopKIndices),
            topk_weights: MoeTensorHandle::new(topk_weights, TensorRole::TopKWeights),
            gate_up_expert_ptrs: MoeTensorHandle::new(
                gate_up_expert_ptrs,
                TensorRole::GateUpExpertPointers,
            ),
            down_expert_ptrs: MoeTensorHandle::new(
                down_expert_ptrs,
                TensorRole::DownExpertPointers,
            ),
            gate_input: MoeTensorHandle::new(gate_input, TensorRole::GateInput),
            gate: MoeTensorHandle::new(gate, TensorRole::Gate),
            up: MoeTensorHandle::new(up, TensorRole::Up),
            activation: MoeTensorHandle::new(activation, TensorRole::Activation),
            expanded_down: MoeTensorHandle::new(expanded_down, TensorRole::ExpandedDown),
            residual: MoeTensorHandle::new(residual, TensorRole::Residual),
            shared_expert,
        }
    }
}

pub struct GroupedPrefillOperands<'a> {
    pub route_logits: MoeTensorHandle<'a>,
    pub topk_indices: MoeTensorHandle<'a>,
    pub topk_weights: MoeTensorHandle<'a>,
    pub gate_up_expert_ptrs: MoeTensorHandle<'a>,
    pub down_expert_ptrs: MoeTensorHandle<'a>,
    pub gate_input: MoeTensorHandle<'a>,
    pub expert_token_counts: MoeTensorHandle<'a>,
    pub expert_offsets: MoeTensorHandle<'a>,
    pub sorted_slot_index: MoeTensorHandle<'a>,
    pub expert_tile_ids: MoeTensorHandle<'a>,
    pub inverse_permutation: MoeTensorHandle<'a>,
    pub grouped_gate_up: MoeTensorHandle<'a>,
    pub gate: MoeTensorHandle<'a>,
    pub up: MoeTensorHandle<'a>,
    pub activation: MoeTensorHandle<'a>,
    pub grouped_down: MoeTensorHandle<'a>,
    pub residual: MoeTensorHandle<'a>,
    pub shared_expert: SharedExpertOperands<'a>,
}

impl<'a> GroupedPrefillOperands<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        route_logits: &'a GpuTensor,
        topk_indices: &'a GpuTensor,
        topk_weights: &'a GpuTensor,
        gate_up_expert_ptrs: &'a GpuTensor,
        down_expert_ptrs: &'a GpuTensor,
        gate_input: &'a GpuTensor,
        expert_token_counts: &'a GpuTensor,
        expert_offsets: &'a GpuTensor,
        sorted_slot_index: &'a GpuTensor,
        expert_tile_ids: &'a GpuTensor,
        inverse_permutation: &'a GpuTensor,
        grouped_gate_up: &'a GpuTensor,
        gate: &'a GpuTensor,
        up: &'a GpuTensor,
        activation: &'a GpuTensor,
        grouped_down: &'a GpuTensor,
        residual: &'a GpuTensor,
        shared_expert: SharedExpertOperands<'a>,
    ) -> Self {
        Self {
            route_logits: MoeTensorHandle::new(route_logits, TensorRole::RouterLogits),
            topk_indices: MoeTensorHandle::new(topk_indices, TensorRole::TopKIndices),
            topk_weights: MoeTensorHandle::new(topk_weights, TensorRole::TopKWeights),
            gate_up_expert_ptrs: MoeTensorHandle::new(
                gate_up_expert_ptrs,
                TensorRole::GateUpExpertPointers,
            ),
            down_expert_ptrs: MoeTensorHandle::new(
                down_expert_ptrs,
                TensorRole::DownExpertPointers,
            ),
            gate_input: MoeTensorHandle::new(gate_input, TensorRole::GateInput),
            expert_token_counts: MoeTensorHandle::new(
                expert_token_counts,
                TensorRole::ExpertTokenCounts,
            ),
            expert_offsets: MoeTensorHandle::new(expert_offsets, TensorRole::ExpertOffsets),
            sorted_slot_index: MoeTensorHandle::new(sorted_slot_index, TensorRole::SortedSlotIndex),
            expert_tile_ids: MoeTensorHandle::new(expert_tile_ids, TensorRole::ExpertTileIds),
            inverse_permutation: MoeTensorHandle::new(
                inverse_permutation,
                TensorRole::InversePermutation,
            ),
            grouped_gate_up: MoeTensorHandle::new(grouped_gate_up, TensorRole::GroupedGateUp),
            gate: MoeTensorHandle::new(gate, TensorRole::Gate),
            up: MoeTensorHandle::new(up, TensorRole::Up),
            activation: MoeTensorHandle::new(activation, TensorRole::Activation),
            grouped_down: MoeTensorHandle::new(grouped_down, TensorRole::GroupedDown),
            residual: MoeTensorHandle::new(residual, TensorRole::Residual),
            shared_expert,
        }
    }
}

pub enum MoeOperands<'a> {
    Indexed(IndexedDecodeOperands<'a>),
    Grouped(GroupedPrefillOperands<'a>),
}

pub struct MoeExecutor;

impl MoeExecutor {
    pub fn execute(
        gpu: &mut Gpu,
        program: &SealedMoeProgram,
        operands: &MoeOperands<'_>,
    ) -> Result<(), DispatchError> {
        // All shape, dtype, role, capacity, placement, and collective checks
        // happen before the first kernel call.  Do not move these below a
        // launch: a failed seal must be side-effect free.
        validate_signature(&program.signature, program.signature.phase)?;
        match (program.phase(), operands) {
            (MoePhase::IndexedDecode, MoeOperands::Indexed(values)) => {
                validate_indexed_operands(&program.signature, values)?;
                Self::execute_indexed(gpu, &program.signature, values)
            }
            (MoePhase::GroupedPrefill, MoeOperands::Grouped(values)) => {
                validate_grouped_operands(&program.signature, values)?;
                Self::execute_grouped(gpu, &program.signature, values)
            }
            (MoePhase::IndexedDecode, MoeOperands::Grouped(_))
            | (MoePhase::GroupedPrefill, MoeOperands::Indexed(_)) => Err(invalid(
                "operand-phase",
                "operand grammar does not match sealed phase",
            )),
        }
    }

    /// Execute the native BF16 shared expert exactly once after routed
    /// combine.  The scalar gate projection, gate/up projections, SiLU
    /// activation, down projection, sigmoid, and residual add are all owned
    /// by the sealed program; callers cannot smuggle in a precomputed shared
    /// contribution.
    fn execute_shared(
        gpu: &mut Gpu,
        signature: &MoeSignature,
        gate_input: MoeTensorHandle<'_>,
        shared: &SharedExpertOperands<'_>,
        residual: MoeTensorHandle<'_>,
    ) -> Result<(), DispatchError> {
        let t = signature.tokens;
        let h = signature.hidden;
        let mi = signature.intermediate;
        hip(gpu.gemm_bf16_xf32_batched(
            shared.gate_weight.tensor(),
            gate_input.tensor(),
            shared.gate_logits.tensor(),
            1,
            h,
            t,
        ))?;
        hip(gpu.gemm_bf16_xf32_batched(
            shared.gate_proj_weight.tensor(),
            gate_input.tensor(),
            shared.gate_proj.tensor(),
            mi,
            h,
            t,
        ))?;
        hip(gpu.gemm_bf16_xf32_batched(
            shared.up_proj_weight.tensor(),
            gate_input.tensor(),
            shared.up_proj.tensor(),
            mi,
            h,
            t,
        ))?;
        hip(gpu.silu_mul_f32(
            shared.gate_proj.tensor(),
            shared.up_proj.tensor(),
            shared.activation.tensor(),
        ))?;
        hip(gpu.gemm_bf16_xf32_batched(
            shared.down_proj_weight.tensor(),
            shared.activation.tensor(),
            shared.output.tensor(),
            h,
            mi,
            t,
        ))?;
        // The add kernel applies sigmoid(gate_logits) exactly once.
        hip(gpu.sigmoid_scaled_residual_add_batched_f32(
            residual.tensor(),
            shared.output.tensor(),
            shared.gate_logits.tensor(),
            t,
            h,
        ))
    }

    fn execute_indexed(
        gpu: &mut Gpu,
        signature: &MoeSignature,
        values: &IndexedDecodeOperands<'_>,
    ) -> Result<(), DispatchError> {
        let t = signature.tokens;
        let h = signature.hidden;
        let mi = signature.intermediate;
        hip(gpu.moe_router_softmax_top10_f32(
            values.route_logits.tensor(),
            values.topk_indices.tensor(),
            values.topk_weights.tensor(),
            t,
            signature.router.normalize_topk_prob,
        ))?;
        hip(gpu.gemv_mq4g256v2_moe_gate_up_top10_indexed_batched(
            values.gate_up_expert_ptrs.tensor(),
            values.topk_indices.tensor(),
            values.gate_input.tensor(),
            values.gate.tensor(),
            values.up.tensor(),
            mi * 2,
            h,
            t,
        ))?;
        hip(gpu.silu_mul_f32(
            values.gate.tensor(),
            values.up.tensor(),
            values.activation.tensor(),
        ))?;
        hip(gpu.gemv_q8_0_moe_down_top10_indexed_batched_expanded(
            values.down_expert_ptrs.tensor(),
            values.topk_indices.tensor(),
            values.activation.tensor(),
            values.expanded_down.tensor(),
            h,
            mi,
            t,
            signature.n_experts,
        ))?;
        hip(gpu.moe_down_combine_top10_batched(
            values.expanded_down.tensor(),
            values.topk_weights.tensor(),
            values.residual.tensor(),
            h,
            t,
        ))?;
        Self::execute_shared(
            gpu,
            signature,
            values.gate_input,
            &values.shared_expert,
            values.residual,
        )
    }

    fn execute_grouped(
        gpu: &mut Gpu,
        signature: &MoeSignature,
        values: &GroupedPrefillOperands<'_>,
    ) -> Result<(), DispatchError> {
        let t = signature.tokens;
        let h = signature.hidden;
        let mi = signature.intermediate;
        let grouped_rows = signature.capacities.grouped_rows;
        hip(gpu.moe_router_softmax_top10_f32(
            values.route_logits.tensor(),
            values.topk_indices.tensor(),
            values.topk_weights.tensor(),
            t,
            signature.router.normalize_topk_prob,
        ))?;
        hip(gpu.moe_scatter_fused_top10(
            values.topk_indices.tensor(),
            values.expert_token_counts.tensor(),
            values.expert_offsets.tensor(),
            values.sorted_slot_index.tensor(),
            values.expert_tile_ids.tensor(),
            values.inverse_permutation.tensor(),
            t * QWEN4_TOP_K,
            signature.n_experts,
            grouped_rows,
            QWEN4_BLOCK_M,
        ))?;
        hip(gpu.gemm_mq4g256v2_moe_grouped_top10(
            values.gate_up_expert_ptrs.tensor(),
            values.expert_tile_ids.tensor(),
            values.sorted_slot_index.tensor(),
            values.gate_input.tensor(),
            values.grouped_gate_up.tensor(),
            mi * 2,
            h,
            QWEN4_TOP_K,
            grouped_rows,
            t,
        ))?;
        hip(gpu.moe_gate_up_unscatter_top10(
            values.grouped_gate_up.tensor(),
            values.sorted_slot_index.tensor(),
            values.gate.tensor(),
            values.up.tensor(),
            mi,
            grouped_rows,
            t,
        ))?;
        hip(gpu.silu_mul_f32(
            values.gate.tensor(),
            values.up.tensor(),
            values.activation.tensor(),
        ))?;
        hip(gpu.gemm_q8_0_moe_grouped_top10(
            values.down_expert_ptrs.tensor(),
            values.expert_tile_ids.tensor(),
            values.sorted_slot_index.tensor(),
            values.activation.tensor(),
            values.grouped_down.tensor(),
            h,
            mi,
            1,
            grouped_rows,
            t * QWEN4_TOP_K,
            signature.n_experts,
        ))?;
        hip(gpu.moe_down_combine_grouped_top10(
            values.grouped_down.tensor(),
            values.inverse_permutation.tensor(),
            values.topk_weights.tensor(),
            values.residual.tensor(),
            h,
            grouped_rows,
            t,
        ))?;
        Self::execute_shared(
            gpu,
            signature,
            values.gate_input,
            &values.shared_expert,
            values.residual,
        )
    }
}

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize, DispatchError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("capacity-overflow", format!("{what}: {a} × {b}")))
}

fn checked_round_up(value: usize, multiple: usize, what: &str) -> Result<usize, DispatchError> {
    if multiple == 0 {
        return Err(invalid("capacity", format!("{what}: zero multiple")));
    }
    let rem = value % multiple;
    if rem == 0 {
        Ok(value)
    } else {
        value
            .checked_add(multiple - rem)
            .ok_or_else(|| invalid("capacity-overflow", format!("{what}: {value} + {multiple}")))
    }
}

fn invalid(variant: &'static str, detail: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(format!("qwen4 sealed MoE {variant}: {detail}"))
}

fn validate_signature(
    signature: &MoeSignature,
    expected_phase: MoePhase,
) -> Result<(), DispatchError> {
    if signature.phase != expected_phase {
        return Err(invalid(
            "phase",
            "signature phase does not match seal entry",
        ));
    }
    if signature.n_experts != QWEN4_N_EXPERTS {
        return Err(invalid(
            "n-experts",
            format!("expected {QWEN4_N_EXPERTS}, got {}", signature.n_experts),
        ));
    }
    if signature.top_k != QWEN4_TOP_K {
        return Err(invalid(
            "top-k",
            format!("expected {QWEN4_TOP_K}, got {}", signature.top_k),
        ));
    }
    if signature.tokens == 0 {
        return Err(invalid("tokens", "zero-token programs are not launchable"));
    }
    if signature.hidden != QWEN4_HIDDEN || signature.intermediate != QWEN4_INTERMEDIATE {
        return Err(invalid(
            "geometry",
            format!(
                "expected H={} MI={}, got H={} MI={}",
                QWEN4_HIDDEN, QWEN4_INTERMEDIATE, signature.hidden, signature.intermediate
            ),
        ));
    }
    if signature.router.logits_dtype != DType::F32 {
        return Err(invalid("router-dtype", "Qwen4 router logits must be F32"));
    }
    let expected_layout = match expected_phase {
        MoePhase::IndexedDecode => MoeLayout::IndexedExpertRows,
        MoePhase::GroupedPrefill => MoeLayout::GroupedExpertTiles,
    };
    if signature.gate_up != ProjectionSpec::qwen4_gate_up(expected_layout)
        || signature.down != ProjectionSpec::qwen4_down(expected_layout)
    {
        return Err(invalid(
            "projection",
            "expected MQ4G256V2 gate/up [2*640,2560] plus Q8_0 down [2560,640]",
        ));
    }
    let slots = checked_mul(signature.tokens, QWEN4_TOP_K, "token count × top-k")?;
    let capacities = signature.capacities;
    if capacities.max_tokens < signature.tokens {
        return Err(invalid("capacity", "max_tokens is smaller than tokens"));
    }
    if capacities.route_slots < slots || capacities.expanded_rows < slots {
        return Err(invalid(
            "capacity",
            format!("route/expanded capacity must cover {slots} slots"),
        ));
    }
    if capacities.block_m != QWEN4_BLOCK_M {
        return Err(invalid("capacity", "Qwen4 grouped block_m must be 16"));
    }
    match expected_phase {
        MoePhase::IndexedDecode => {
            if capacities.grouped_rows != 0 {
                return Err(invalid(
                    "capacity",
                    "indexed programs must not carry grouped rows",
                ));
            }
            if !matches!(signature.placement, MoePlacement::Single)
                && matches!(signature.collectives, CollectiveSchedule::None)
            {
                return Err(invalid(
                    "collectives",
                    "mesh placement requires its validated ordered collective schedule",
                ));
            }
        }
        MoePhase::GroupedPrefill => {
            if capacities.grouped_rows < slots
                || !capacities.grouped_rows.is_multiple_of(QWEN4_BLOCK_M)
            {
                return Err(invalid(
                    "capacity",
                    "grouped_rows must cover slots and be block_m aligned",
                ));
            }
        }
    }
    if matches!(signature.placement, MoePlacement::Single)
        && !matches!(signature.collectives, CollectiveSchedule::None)
    {
        return Err(invalid(
            "collectives",
            "Single placement cannot emit a synthetic collective",
        ));
    }
    Ok(())
}

fn require(
    handle: MoeTensorHandle<'_>,
    role: TensorRole,
    dtype: Option<DType>,
    elements: usize,
) -> Result<(), DispatchError> {
    if handle.role() != role {
        return Err(invalid(
            "operand-role",
            format!("expected {role:?}, got {:?}", handle.role()),
        ));
    }
    if handle.tensor().numel() < elements {
        return Err(invalid(
            "operand-capacity",
            format!(
                "{role:?}: need {elements} elements, got {}",
                handle.tensor().numel()
            ),
        ));
    }
    if let Some(expected) = dtype {
        if handle.tensor().dtype != expected {
            return Err(invalid(
                "operand-dtype",
                format!(
                    "{role:?}: expected {expected:?}, got {:?}",
                    handle.tensor().dtype
                ),
            ));
        }
    }
    Ok(())
}

fn validate_common_operands(
    signature: &MoeSignature,
    route_logits: MoeTensorHandle<'_>,
    topk_indices: MoeTensorHandle<'_>,
    topk_weights: MoeTensorHandle<'_>,
    gate_up_ptrs: MoeTensorHandle<'_>,
    down_ptrs: MoeTensorHandle<'_>,
    gate_input: MoeTensorHandle<'_>,
    gate: MoeTensorHandle<'_>,
    up: MoeTensorHandle<'_>,
    activation: MoeTensorHandle<'_>,
    residual: MoeTensorHandle<'_>,
    shared: &SharedExpertOperands<'_>,
) -> Result<(), DispatchError> {
    let slots = checked_mul(signature.tokens, QWEN4_TOP_K, "token count × top-k")?;
    let hidden_rows = checked_mul(signature.tokens, signature.hidden, "token count × hidden")?;
    let route_logits_len = checked_mul(signature.tokens, signature.n_experts, "router logits")?;
    let expert_mi = checked_mul(slots, signature.intermediate, "route slots × intermediate")?;
    require(
        route_logits,
        TensorRole::RouterLogits,
        Some(DType::F32),
        route_logits_len,
    )?;
    // Indices are i32 bit patterns in F32-typed GPU storage, matching the
    // existing RDNA compute convention.
    require(
        topk_indices,
        TensorRole::TopKIndices,
        Some(DType::F32),
        slots,
    )?;
    require(
        topk_weights,
        TensorRole::TopKWeights,
        Some(DType::F32),
        slots,
    )?;
    require(
        gate_up_ptrs,
        TensorRole::GateUpExpertPointers,
        Some(DType::F32),
        signature.n_experts,
    )?;
    require(
        down_ptrs,
        TensorRole::DownExpertPointers,
        Some(DType::F32),
        signature.n_experts,
    )?;
    require(
        gate_input,
        TensorRole::GateInput,
        Some(DType::F32),
        hidden_rows,
    )?;
    require(gate, TensorRole::Gate, Some(DType::F32), expert_mi)?;
    require(up, TensorRole::Up, Some(DType::F32), expert_mi)?;
    require(
        activation,
        TensorRole::Activation,
        Some(DType::F32),
        expert_mi,
    )?;
    require(
        residual,
        TensorRole::Residual,
        Some(DType::F32),
        hidden_rows,
    )?;
    validate_shared_operands(signature, shared)
}

fn validate_shared_operands(
    signature: &MoeSignature,
    shared: &SharedExpertOperands<'_>,
) -> Result<(), DispatchError> {
    let hidden_rows = checked_mul(signature.tokens, signature.hidden, "shared output")?;
    let shared_mi = checked_mul(
        signature.tokens,
        signature.intermediate,
        "shared token count × intermediate",
    )?;
    require(
        shared.gate_weight,
        TensorRole::SharedGateWeight,
        Some(DType::BF16),
        signature.hidden,
    )?;
    require(
        shared.gate_proj_weight,
        TensorRole::SharedGateProjWeight,
        Some(DType::BF16),
        checked_mul(
            signature.intermediate,
            signature.hidden,
            "shared gate projection",
        )?,
    )?;
    require(
        shared.up_proj_weight,
        TensorRole::SharedUpProjWeight,
        Some(DType::BF16),
        checked_mul(
            signature.intermediate,
            signature.hidden,
            "shared up projection",
        )?,
    )?;
    require(
        shared.down_proj_weight,
        TensorRole::SharedDownProjWeight,
        Some(DType::BF16),
        checked_mul(
            signature.hidden,
            signature.intermediate,
            "shared down projection",
        )?,
    )?;
    require(
        shared.gate_logits,
        TensorRole::SharedGateLogits,
        Some(DType::F32),
        signature.tokens,
    )?;
    require(
        shared.gate_proj,
        TensorRole::SharedGateProj,
        Some(DType::F32),
        shared_mi,
    )?;
    require(
        shared.up_proj,
        TensorRole::SharedUpProj,
        Some(DType::F32),
        shared_mi,
    )?;
    require(
        shared.activation,
        TensorRole::SharedActivation,
        Some(DType::F32),
        shared_mi,
    )?;
    require(
        shared.output,
        TensorRole::SharedOutput,
        Some(DType::F32),
        hidden_rows,
    )
}

fn validate_indexed_operands(
    signature: &MoeSignature,
    values: &IndexedDecodeOperands<'_>,
) -> Result<(), DispatchError> {
    validate_common_operands(
        signature,
        values.route_logits,
        values.topk_indices,
        values.topk_weights,
        values.gate_up_expert_ptrs,
        values.down_expert_ptrs,
        values.gate_input,
        values.gate,
        values.up,
        values.activation,
        values.residual,
        &values.shared_expert,
    )?;
    let expanded = checked_mul(
        signature.tokens,
        checked_mul(signature.top_k, signature.hidden, "top-k × hidden")?,
        "expanded down",
    )?;
    require(
        values.expanded_down,
        TensorRole::ExpandedDown,
        Some(DType::F32),
        expanded,
    )
}

fn validate_grouped_operands(
    signature: &MoeSignature,
    values: &GroupedPrefillOperands<'_>,
) -> Result<(), DispatchError> {
    validate_common_operands(
        signature,
        values.route_logits,
        values.topk_indices,
        values.topk_weights,
        values.gate_up_expert_ptrs,
        values.down_expert_ptrs,
        values.gate_input,
        values.gate,
        values.up,
        values.activation,
        values.residual,
        &values.shared_expert,
    )?;
    let grouped = signature.capacities.grouped_rows;
    let grouped_gate_up = checked_mul(
        grouped,
        checked_mul(2, signature.intermediate, "grouped gate/up width")?,
        "grouped gate/up",
    )?;
    let grouped_down = checked_mul(grouped, signature.hidden, "grouped down")?;
    let tiles = grouped / QWEN4_BLOCK_M;
    require(
        values.expert_token_counts,
        TensorRole::ExpertTokenCounts,
        Some(DType::F32),
        signature.n_experts,
    )?;
    require(
        values.expert_offsets,
        TensorRole::ExpertOffsets,
        Some(DType::F32),
        signature.n_experts + 1,
    )?;
    require(
        values.sorted_slot_index,
        TensorRole::SortedSlotIndex,
        Some(DType::F32),
        grouped,
    )?;
    require(
        values.expert_tile_ids,
        TensorRole::ExpertTileIds,
        Some(DType::F32),
        tiles,
    )?;
    require(
        values.inverse_permutation,
        TensorRole::InversePermutation,
        Some(DType::F32),
        signature.tokens * QWEN4_TOP_K,
    )?;
    require(
        values.grouped_gate_up,
        TensorRole::GroupedGateUp,
        Some(DType::F32),
        grouped_gate_up,
    )?;
    require(
        values.grouped_down,
        TensorRole::GroupedDown,
        Some(DType::F32),
        grouped_down,
    )
}

fn hip(result: hip_bridge::HipResult<()>) -> Result<(), DispatchError> {
    result.map_err(|error| DispatchError::Hip(format!("{error:?}")))
}

/// CPU oracle used by deterministic tests and admission tooling.  Production
/// execution uses [`MoeExecutor`] and the GPU router; this helper is not a
/// host top-k fallback.
#[derive(Clone, Debug, PartialEq)]
pub struct RouteSelection {
    pub indices: [usize; QWEN4_TOP_K],
    pub weights: [f32; QWEN4_TOP_K],
}

pub fn select_top10_reference(
    logits: &[f32],
    normalize_topk_prob: bool,
) -> Result<RouteSelection, DispatchError> {
    if logits.len() != QWEN4_N_EXPERTS {
        return Err(invalid(
            "reference-router",
            format!("expected {} logits, got {}", QWEN4_N_EXPERTS, logits.len()),
        ));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(invalid("reference-router", "router logits must be finite"));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs = [0.0f32; QWEN4_N_EXPERTS];
    let mut sum = 0.0f32;
    for (prob, &logit) in probs.iter_mut().zip(logits) {
        *prob = (logit - max).exp();
        sum += *prob;
    }
    if !sum.is_finite() || sum <= 0.0 {
        return Err(invalid(
            "reference-router",
            "softmax denominator is not positive",
        ));
    }
    let inv_sum = 1.0 / sum;
    for prob in &mut probs {
        *prob *= inv_sum;
    }

    let mut selected = [false; QWEN4_N_EXPERTS];
    let mut result = RouteSelection {
        indices: [0; QWEN4_TOP_K],
        weights: [0.0; QWEN4_TOP_K],
    };
    for rank in 0..QWEN4_TOP_K {
        let mut best_index = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        let mut found = false;
        for (index, &value) in probs.iter().enumerate() {
            if selected[index] {
                continue;
            }
            if !found || value > best_value || (value == best_value && index < best_index) {
                best_index = index;
                best_value = value;
                found = true;
            }
        }
        selected[best_index] = true;
        result.indices[rank] = best_index;
        result.weights[rank] = best_value;
    }
    if normalize_topk_prob {
        let selected_sum: f32 = result.weights.iter().sum();
        if !selected_sum.is_finite() || selected_sum <= 0.0 {
            return Err(invalid(
                "reference-router",
                "selected denominator is not positive",
            ));
        }
        for weight in &mut result.weights {
            *weight /= selected_sum;
        }
    }
    Ok(result)
}

/// CPU numeric oracle for the unweighted expanded-down contract.  Each route
/// output is multiplied exactly once by its normalized route weight.
pub fn combine_top10_reference(
    expanded_down: &[f32],
    topk_weights: &[f32],
    tokens: usize,
    hidden: usize,
) -> Result<Vec<f32>, DispatchError> {
    if hidden == 0 {
        return Err(invalid("reference-combine", "hidden width is zero"));
    }
    let slots = checked_mul(tokens, QWEN4_TOP_K, "token count × top-k")?;
    let expanded_len = checked_mul(slots, hidden, "expanded down")?;
    if expanded_down.len() != expanded_len {
        return Err(invalid(
            "reference-combine",
            format!(
                "expected {expanded_len} expanded values, got {}",
                expanded_down.len()
            ),
        ));
    }
    if topk_weights.len() != slots {
        return Err(invalid(
            "reference-combine",
            format!("expected {slots} route weights, got {}", topk_weights.len()),
        ));
    }
    let mut output = vec![0.0f32; checked_mul(tokens, hidden, "combined output")?];
    for token in 0..tokens {
        for rank in 0..QWEN4_TOP_K {
            let weight = topk_weights[token * QWEN4_TOP_K + rank];
            let source = (token * QWEN4_TOP_K + rank) * hidden;
            let destination = token * hidden;
            for column in 0..hidden {
                output[destination + column] += weight * expanded_down[source + column];
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen4_indexed_seal_has_fixed_grammar_and_no_collective() {
        let signature = MoeSignature::qwen4_indexed(
            3,
            RouteIdentity(7),
            PermutationIdentity(11),
            ExecutionEpoch(13),
            MoePlacement::Single,
            CollectiveSchedule::None,
        )
        .expect("canonical signature");
        let program = MoeProgram::seal_indexed(signature).expect("indexed seal");
        assert_eq!(program.operations(), INDEXED_OPS);
        assert!(!program
            .operations()
            .iter()
            .any(|operation| matches!(operation, MoeOp::Scatter)));
    }

    #[test]
    fn qwen4_seal_rejects_k8_without_touching_a_gpu() {
        let mut signature = MoeSignature::qwen4_indexed(
            1,
            RouteIdentity(1),
            PermutationIdentity(1),
            ExecutionEpoch(1),
            MoePlacement::Single,
            CollectiveSchedule::None,
        )
        .expect("canonical signature");
        signature.top_k = 8;
        assert!(MoeProgram::seal_indexed(signature).is_err());
    }

    #[test]
    fn reference_router_uses_lower_index_for_ties_and_renormalizes() {
        let logits = vec![0.0f32; QWEN4_N_EXPERTS];
        let result = select_top10_reference(&logits, true).expect("reference route");
        assert_eq!(result.indices, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let sum: f32 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn reference_combine_weights_each_expanded_row_once() {
        let tokens = 1;
        let hidden = 2;
        let mut expanded = vec![0.0f32; QWEN4_TOP_K * hidden];
        for rank in 0..QWEN4_TOP_K {
            expanded[rank * hidden] = rank as f32 + 1.0;
            expanded[rank * hidden + 1] = 2.0;
        }
        let mut weights = [0.0f32; QWEN4_TOP_K];
        weights[0] = 0.25;
        weights[1] = 0.75;
        let output = combine_top10_reference(&expanded, &weights, tokens, hidden)
            .expect("reference combine");
        assert_eq!(output, vec![1.75, 2.0]);
    }
}
