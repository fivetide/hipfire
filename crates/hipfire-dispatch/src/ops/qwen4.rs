// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026
// hipfire — Qwen4-specific typed operation adapters.

//! Adapters from the Qwen4 carrier's layer operands to the sealed MoE grammar.
//! The adapter never resolves names, consults `WeightStore`, or invents a
//! fallback: route identity, permutation identity, epoch, placement, and
//! collectives are explicit arguments to the seal.

use crate::families::sealed_moe::{
    CollectiveSchedule, ExecutionEpoch, GroupedPrefillOperands, IndexedDecodeOperands, MoeExecutor,
    MoeOperands, MoePlacement, MoeProgram, MoeSignature, PermutationIdentity, RouteIdentity,
    SealedMoeProgram,
};
use crate::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor};

/// Qwen4 indexed/decode operand bundle.  The route and all scratch buffers are
/// model-owned `GpuTensor`s; semantic roles are assigned by the sealed
/// constructor, not by this adapter.
pub struct Qwen4IndexedOperands<'a> {
    pub route_logits: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub gate_up_ptrs: &'a GpuTensor,
    pub down_ptrs: &'a GpuTensor,
    pub gate_input: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub activation: &'a GpuTensor,
    pub expanded_down: &'a GpuTensor,
    pub residual: &'a GpuTensor,
    pub shared_output: &'a GpuTensor,
    pub shared_gate: &'a GpuTensor,
}

/// Qwen4 grouped/prefill operand bundle.
pub struct Qwen4GroupedOperands<'a> {
    pub route_logits: &'a GpuTensor,
    pub topk_indices: &'a GpuTensor,
    pub topk_weights: &'a GpuTensor,
    pub gate_up_ptrs: &'a GpuTensor,
    pub down_ptrs: &'a GpuTensor,
    pub gate_input: &'a GpuTensor,
    pub expert_token_counts: &'a GpuTensor,
    pub expert_offsets: &'a GpuTensor,
    pub sorted_slot_index: &'a GpuTensor,
    pub expert_tile_ids: &'a GpuTensor,
    pub inverse_permutation: &'a GpuTensor,
    pub grouped_gate_up: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub activation: &'a GpuTensor,
    pub grouped_down: &'a GpuTensor,
    pub residual: &'a GpuTensor,
    pub shared_output: &'a GpuTensor,
    pub shared_gate: &'a GpuTensor,
}

/// Seal and execute one Qwen4 decode MoE block.  All signature validation runs
/// inside `MoeExecutor` before its first GPU launch.
pub fn execute_indexed(
    gpu: &mut Gpu,
    tokens: usize,
    route_id: RouteIdentity,
    permutation_id: PermutationIdentity,
    epoch: ExecutionEpoch,
    operands: &Qwen4IndexedOperands<'_>,
) -> Result<SealedMoeProgram, DispatchError> {
    let signature = MoeSignature::qwen4_indexed(
        tokens,
        route_id,
        permutation_id,
        epoch,
        MoePlacement::Single,
        CollectiveSchedule::None,
    )?;
    let program = MoeProgram::seal_indexed(signature)?;
    let typed = IndexedDecodeOperands::new(
        operands.route_logits,
        operands.topk_indices,
        operands.topk_weights,
        operands.gate_up_ptrs,
        operands.down_ptrs,
        operands.gate_input,
        operands.gate,
        operands.up,
        operands.activation,
        operands.expanded_down,
        operands.residual,
        operands.shared_output,
        operands.shared_gate,
    );
    MoeExecutor::execute(gpu, &program, &MoeOperands::Indexed(typed))?;
    Ok(program)
}

/// Seal and execute one bounded Qwen4 grouped prefill block.
pub fn execute_grouped(
    gpu: &mut Gpu,
    tokens: usize,
    route_id: RouteIdentity,
    permutation_id: PermutationIdentity,
    epoch: ExecutionEpoch,
    operands: &Qwen4GroupedOperands<'_>,
) -> Result<SealedMoeProgram, DispatchError> {
    let signature = MoeSignature::qwen4_grouped(
        tokens,
        route_id,
        permutation_id,
        epoch,
        MoePlacement::Single,
        CollectiveSchedule::None,
    )?;
    let program = MoeProgram::seal_grouped(signature)?;
    let typed = GroupedPrefillOperands::new(
        operands.route_logits,
        operands.topk_indices,
        operands.topk_weights,
        operands.gate_up_ptrs,
        operands.down_ptrs,
        operands.gate_input,
        operands.expert_token_counts,
        operands.expert_offsets,
        operands.sorted_slot_index,
        operands.expert_tile_ids,
        operands.inverse_permutation,
        operands.grouped_gate_up,
        operands.gate,
        operands.up,
        operands.activation,
        operands.grouped_down,
        operands.residual,
        operands.shared_output,
        operands.shared_gate,
    );
    MoeExecutor::execute(gpu, &program, &MoeOperands::Grouped(typed))?;
    Ok(program)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexed_adapter_uses_single_without_collectives() {
        let signature = MoeSignature::qwen4_indexed(
            1,
            RouteIdentity(1),
            PermutationIdentity(2),
            ExecutionEpoch(3),
            MoePlacement::Single,
            CollectiveSchedule::None,
        )
        .unwrap();
        let program = MoeProgram::seal_indexed(signature).unwrap();
        assert_eq!(program.signature().top_k, 10);
        assert_eq!(program.signature().collectives, CollectiveSchedule::None);
    }
}
