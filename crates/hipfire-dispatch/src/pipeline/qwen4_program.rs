// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Shared Qwen4 layer-major execution.
//!
//! The architecture crate binds resident weights, state tensors, and scratch
//! views into these descriptors.  This module owns the layer sequence and the
//! batch-shaped projection launches; family code does not interpret GDN/QSA
//! kernels or choose GEMV versus GEMM.

use crate::families::gemv::WeightRef;
use crate::families::moe::{
    MoeDtypes, MoeEpMode, MoeNormalization, MoeParams, MoePrefillParams, MoePrefillPrelude,
    MoeQ8RouterPolicy, MoeRecipe, MoeSharedDecode, MoeSharedDtypes, MoeSharedPrefill,
    MoeSharedWeights, RoutedExpertWeights,
};
use crate::pipeline::sealed_moe::PrefillRouteMode;
use crate::pipeline::{
    execute_steps, seal_decode, seal_prefill, BoundMoeExperts, ExpertBindingCache, ExpertTable,
    Step,
};
use crate::types::DispatchError;
use rdna_compute::qwen4::{
    qwen4_gdn_bf16_roundtrip, qwen4_gdn_conv, qwen4_gdn_gate, qwen4_gdn_params, qwen4_gdn_step,
    qwen4_hc_norm, qwen4_hc_read_projected, qwen4_hc_write, qwen4_qsa_attention,
    qwen4_qsa_cache_append, qwen4_qsa_norm_rope, qwen4_qsa_pool_rope, qwen4_qsa_select,
    qwen4_scale, Qwen4GdnBf16Roundtrip, Qwen4GdnConv, Qwen4GdnGate, Qwen4GdnParams, Qwen4GdnStep,
    Qwen4HcNorm, Qwen4HcReadProjected, Qwen4HcWrite, Qwen4QsaAttention, Qwen4QsaCacheAppend,
    Qwen4QsaNormRope, Qwen4QsaPoolRope, Qwen4QsaSelect, Qwen4Scale,
};
use rdna_compute::{DType, Gpu, GpuTensor};

const GROUPED_BLOCK_M: usize = 16;
const QWEN4_EXPERTS: usize = 512;
const QWEN4_TOP_K: usize = 10;

#[inline]
fn hip<T>(result: Result<T, hip_bridge::HipError>) -> Result<T, DispatchError> {
    result.map_err(|error| DispatchError::Hip(error.to_string()))
}

#[inline]
fn view(source: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    source.sub_offset(offset, len)
}

fn shaped_prefix(
    source: &GpuTensor,
    rows: usize,
    cols: usize,
    label: &'static str,
) -> Result<GpuTensor, DispatchError> {
    let elements = checked_mul(rows, cols, label)?;
    if source.numel() < elements {
        return Err(DispatchError::Hip(format!(
            "Qwen4 {label} scratch capacity {} is below {elements}",
            source.numel()
        )));
    }
    // SAFETY: this is a non-owning metadata view of the family-owned scratch
    // tensor.  The alias is never passed to free_tensor; the owner outlives
    // the sealed call that borrows this view.
    Ok(GpuTensor {
        buf: unsafe { source.buf.alias() },
        shape: vec![rows, cols],
        dtype: source.dtype,
    })
}

#[inline]
fn checked_mul(a: usize, b: usize, label: &'static str) -> Result<usize, DispatchError> {
    a.checked_mul(b)
        .ok_or_else(|| DispatchError::Hip(format!("Qwen4 {label} scratch overflows")))
}

/// Dimensions used by all Qwen4 layer-major operations.  The architecture
/// binds these from its validated config; no family-specific config type
/// crosses the dispatch boundary.
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ProgramDims {
    pub hidden: usize,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub indexer_n_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub moe_intermediate: usize,
    pub shared_intermediate: usize,
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub ple_conv_kernel_dim: usize,
    pub norm_eps: f32,
}

impl Qwen4ProgramDims {
    #[inline]
    pub fn wide(self) -> usize {
        self.hc_count * self.hidden
    }
    #[inline]
    pub fn q_width(self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    #[inline]
    pub fn kv_width(self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    #[inline]
    pub fn gdn_qk(self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }
    #[inline]
    pub fn gdn_value(self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
    #[inline]
    pub fn gdn_qkv(self) -> usize {
        2 * self.gdn_qk() + self.gdn_value()
    }
    #[inline]
    pub fn index_width(self) -> usize {
        (self.indexer_n_heads + self.indexer_kv_heads) * self.indexer_head_dim
    }
}
/// Central scratch geometry plan. Allocation remains family-owned, but every
/// extent used by the family binder comes from this checked plan.
#[derive(Clone, Copy, Debug)]
pub struct Qwen4ScratchLayout {
    pub rows: usize,
    pub hidden: usize,
    pub wide: usize,
    pub hc_low: usize,
    pub q_width: usize,
    pub kv_width: usize,
    pub index_width: usize,
    pub gdn_qkv: usize,
    pub gdn_value: usize,
    pub slots: usize,
    pub grouped_rows: usize,
    pub shared_rows: usize,
    pub routed_rows: usize,
    pub expanded_rows: usize,
}

impl Qwen4ScratchLayout {
    pub fn for_rows(dims: Qwen4ProgramDims, rows: usize) -> Result<Self, DispatchError> {
        validate_dims(dims)?;
        if rows == 0 {
            return Err(DispatchError::Hip("Qwen4 scratch rows are zero".into()));
        }
        let wide = checked_mul(dims.hc_count, dims.hidden, "wide")?;
        let slots = checked_mul(rows, dims.experts_per_token, "MoE slots")?;
        Ok(Self {
            rows,
            hidden: dims.hidden,
            wide,
            hc_low: checked_mul(rows, dims.hc_lowrank, "HC low")?,
            q_width: checked_mul(dims.num_attention_heads, dims.head_dim, "Q width")?,
            kv_width: checked_mul(dims.num_key_value_heads, dims.head_dim, "KV width")?,
            index_width: dims.index_width(),
            gdn_qkv: dims.gdn_qkv(),
            gdn_value: dims.gdn_value(),
            slots,
            grouped_rows: grouped_m_total_bound(slots, dims.num_experts)?,
            shared_rows: checked_mul(rows, dims.shared_intermediate, "shared rows")?,
            routed_rows: checked_mul(slots, dims.moe_intermediate, "routed rows")?,
            expanded_rows: checked_mul(slots, dims.hidden, "expanded route")?,
        })
    }
}

/// Read-only hyper-connection operands used by HC read/final collapse.
///
/// The final trunk mixer has exactly this shape; it intentionally has no
/// block-inject operand because final HC is a read, not a write.
pub struct Qwen4HyperReadWeights<'a> {
    pub norm: &'a GpuTensor,
    pub input_mix_down: WeightRef<'a>,
    pub input_mix_up: WeightRef<'a>,
}

/// Write-side hyper-connection operands used by layer HC writes.
pub struct Qwen4HyperWriteWeights<'a> {
    pub norm: &'a GpuTensor,
    pub block_inject: WeightRef<'a>,
}

/// Complete per-layer hyper-connection descriptor.
pub struct Qwen4HyperWeights<'a> {
    pub read: Qwen4HyperReadWeights<'a>,
    pub write: Qwen4HyperWriteWeights<'a>,
}

pub struct Qwen4GdnWeights<'a> {
    pub qkv: WeightRef<'a>,
    pub conv: &'a GpuTensor,
    pub in_proj_a: WeightRef<'a>,
    pub in_proj_b: WeightRef<'a>,
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    pub z: WeightRef<'a>,
    pub norm: &'a GpuTensor,
    pub output: WeightRef<'a>,
}

pub struct Qwen4QsaWeights<'a> {
    pub indexer_qk: WeightRef<'a>,
    pub indexer_q_norm: &'a GpuTensor,
    pub indexer_k_norm: &'a GpuTensor,
    pub q: WeightRef<'a>,
    pub k: WeightRef<'a>,
    pub v: WeightRef<'a>,
    pub q_norm: &'a GpuTensor,
    pub k_norm: &'a GpuTensor,
    pub output: WeightRef<'a>,
}

pub enum Qwen4AttentionWeights<'a> {
    Linear(Qwen4GdnWeights<'a>),
    Full(Qwen4QsaWeights<'a>),
}

pub struct Qwen4LayerDescription<'a> {
    pub attn_hyper: Qwen4HyperWeights<'a>,
    pub mlp_hyper: Qwen4HyperWeights<'a>,
    pub attention: Qwen4AttentionWeights<'a>,
    pub moe: Qwen4MoeBinding<'a>,
}

/// lifecycle control; convolution position is derived from the request
/// position supplied to the layer operation.
pub struct Qwen4GdnState<'a> {
    pub recurrent: &'a GpuTensor,
    pub conv: &'a GpuTensor,
}

/// A mutable view of one QSA cache and its causal metadata.
///
/// The descriptor owns scalar metadata for the duration of one shared Step
/// list.  The family commits these values back to its request state after the
/// list succeeds; device cache tensors remain borrowed views.
pub struct Qwen4QsaState<'a> {
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub raw_index_keys: &'a GpuTensor,
    pub pooled_keys: &'a GpuTensor,
    pub selected_indices: &'a GpuTensor,
    pub full_capacity: usize,
    pub raw_capacity: usize,
    pub pooled_capacity: usize,
    pub selected_capacity: usize,
    pub position_capacity: usize,
    pub full_len: usize,
    pub raw_len: usize,
    pub pooled_len: usize,
    pub selected_len: usize,
    pub position: usize,
}

/// MoE resources are source-bound once by the family and consumed by the
/// shared prefill sealer/lowering. The dispatch layer owns all route policy.
pub struct Qwen4MoeBinding<'a> {
    pub table: &'a ExpertTable,
    pub cache: &'a ExpertBindingCache,
    pub routed_experts: &'a dyn RoutedExpertWeights,
    pub router: WeightRef<'a>,
    pub shared: MoeSharedWeights<'a>,
    pub intermediate: usize,
    pub experts_all_gate_up_mq4: bool,
    pub expert_gate_up_ptrs: &'a GpuTensor,
    pub expert_down_ptrs: &'a GpuTensor,
    pub layer_idx: u16,
    pub norm_topk_prob: bool,
}

/// Family-owned reusable views into the central program's scratch layout.
/// Every field is a borrowed subview; this struct performs no allocation.
pub struct Qwen4LayerScratch<'a> {
    pub streams: &'a GpuTensor,
    pub hc_normalized: &'a GpuTensor,
    pub hc_low: &'a GpuTensor,
    pub hc_up: &'a GpuTensor,
    pub hc_mixed: &'a GpuTensor,
    pub hc_gates: &'a GpuTensor,
    pub projection: &'a GpuTensor,
    pub projection2: &'a GpuTensor,
    pub gdn_a: &'a GpuTensor,
    pub gdn_b: &'a GpuTensor,
    pub gdn_gate: &'a GpuTensor,
    pub gdn_beta: &'a GpuTensor,
    pub gdn_recurrent_output: &'a GpuTensor,
    pub gdn_bf16: &'a GpuTensor,
    pub gdn_z: &'a GpuTensor,
    pub gdn_output: &'a GpuTensor,
    pub qsa_index: &'a GpuTensor,
    pub qsa_qgate: &'a GpuTensor,
    pub qsa_k: &'a GpuTensor,
    pub qsa_v: &'a GpuTensor,
    pub qsa_output: &'a GpuTensor,
    pub attention_output: &'a GpuTensor,
    pub moe_output: &'a GpuTensor,
    pub moe_shared_output: &'a GpuTensor,
    pub moe_router_logits: &'a GpuTensor,
    pub moe_x_rot: &'a GpuTensor,
    pub moe_gate_up: &'a GpuTensor,
    pub moe_scalar: &'a GpuTensor,
    pub moe_gate: &'a GpuTensor,
    pub moe_up: &'a GpuTensor,
    pub moe_hidden: &'a GpuTensor,
    pub moe_gate_batch: &'a GpuTensor,
    pub moe_up_batch: &'a GpuTensor,
    pub moe_rot_batch: &'a GpuTensor,
    pub moe_topk_indices: &'a GpuTensor,
    pub moe_topk_weights: &'a GpuTensor,
    pub moe_down_expanded: &'a GpuTensor,
    pub moe_expert_token_counts: &'a GpuTensor,
    pub moe_expert_offsets: &'a GpuTensor,
    pub moe_sorted_slot_index: &'a GpuTensor,
    pub moe_expert_tile_ids: &'a GpuTensor,
    pub moe_inverse_perm: &'a GpuTensor,
    pub moe_y_gate_up_grouped: &'a GpuTensor,
    pub moe_y_down_grouped: &'a GpuTensor,
}
pub struct Qwen4LayerOp<'a> {
    pub dims: Qwen4ProgramDims,
    pub layer: Qwen4LayerDescription<'a>,
    pub state_gdn: Option<Qwen4GdnState<'a>>,
    pub state_qsa: Option<Qwen4QsaState<'a>>,
    pub scratch: &'a Qwen4LayerScratch<'a>,
    pub rows: usize,
    pub start_position: usize,
}

pub struct Qwen4PleWeights<'a> {
    pub key: WeightRef<'a>,
    pub value: WeightRef<'a>,
    pub norm_key: &'a GpuTensor,
    pub norm_query: &'a GpuTensor,
    pub norm_conv: &'a GpuTensor,
    pub conv: &'a GpuTensor,
}

pub struct Qwen4PleOp<'a> {
    pub dims: Qwen4ProgramDims,
    pub weights: Qwen4PleWeights<'a>,
    pub state: &'a GpuTensor,
    pub streams: &'a GpuTensor,
    pub ple_rows: &'a GpuTensor,
    pub query: &'a GpuTensor,
    pub key: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub gated: &'a GpuTensor,
    pub normed: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub rows: usize,
}

pub fn validate_ple(op: &Qwen4PleOp<'_>) -> Result<(), DispatchError> {
    validate_dims(op.dims)?;
    if op.rows == 0 {
        return Err(DispatchError::Hip("Qwen4 PLE rows are zero".into()));
    }
    let channels = checked_mul(op.dims.hc_count, op.dims.hidden, "PLE channels")?;
    let row_hidden = checked_mul(op.rows, op.dims.hidden, "PLE value rows")?;
    let row_channels = checked_mul(op.rows, channels, "PLE channel rows")?;
    let history_rows = checked_mul(
        op.dims.ple_conv_kernel_dim.saturating_sub(1),
        3,
        "PLE history rows",
    )?;
    let history_elements = checked_mul(history_rows, channels, "PLE state")?;
    for (tensor, elements, name, dtype) in [
        (op.streams, row_channels, "PLE streams", DType::F32),
        (op.ple_rows, row_hidden, "PLE rows", DType::F32),
        (op.query, row_channels, "PLE query", DType::F32),
        (op.key, row_channels, "PLE key", DType::F32),
        (op.value, row_hidden, "PLE value", DType::F32),
        (op.gated, row_channels, "PLE gated", DType::F32),
        (op.normed, row_channels, "PLE normed", DType::F32),
        (op.output, row_channels, "PLE output", DType::F32),
        (op.state, history_elements, "PLE state", DType::F32),
    ] {
        require_tensor(tensor, elements, dtype, name)?;
    }
    Ok(())
}

pub fn execute_ple(gpu: &mut Gpu, op: &Qwen4PleOp<'_>) -> Result<(), DispatchError> {
    validate_ple(op)?;
    let channels = checked_mul(op.dims.hc_count, op.dims.hidden, "PLE channels")?;
    let stream_batch = op.streams.sub_offset(0, op.rows * channels);
    let query_batch = op.query.sub_offset(0, op.rows * channels);
    let key_batch = op.key.sub_offset(0, op.rows * channels);
    let value_batch = op.value.sub_offset(0, op.rows * op.dims.hidden);
    let gated_batch = op.gated.sub_offset(0, op.rows * channels);
    let normed_batch = op.normed.sub_offset(0, op.rows * channels);
    let output_batch = op.output.sub_offset(0, op.rows * channels);
    project_bf16_batch(gpu, &op.weights.key, op.ple_rows, &key_batch, op.rows)?;
    project_bf16_batch(gpu, &op.weights.value, op.ple_rows, &value_batch, op.rows)?;
    hip(gpu.copy_d2d(&stream_batch, &query_batch, stream_batch.byte_size()))?;
    hip(gpu.qwen4_ple_gate_bf16(
        &key_batch,
        &query_batch,
        &value_batch,
        op.weights.norm_key,
        op.weights.norm_query,
        &gated_batch,
        op.rows,
        op.dims.hc_count,
        op.dims.hidden,
        op.dims.norm_eps,
    ))?;
    hip(gpu.qwen4_ple_norm_bf16(
        &gated_batch,
        op.weights.norm_conv,
        &normed_batch,
        op.rows,
        op.dims.hc_count,
        op.dims.hidden,
        op.dims.norm_eps,
    ))?;
    hip(gpu.qwen4_ple_depthwise_conv_silu_add_bf16(
        &gated_batch,
        &normed_batch,
        op.weights.conv,
        op.state,
        &output_batch,
        op.rows,
        channels,
        op.dims.ple_conv_kernel_dim,
        3,
    ))?;
    hip(gpu.add_f32(&stream_batch, &output_batch, &stream_batch))
}

/// Return the tile-aligned grouped scratch bound for a bounded chunk.
pub fn grouped_m_total_bound(total_slots: usize, n_exp: usize) -> Result<usize, DispatchError> {
    let live = total_slots.min(n_exp);
    let padded = total_slots
        .checked_add(checked_mul(live, GROUPED_BLOCK_M - 1, "grouped MoE")?)
        .ok_or_else(|| DispatchError::Hip("Qwen4 grouped MoE scratch overflows".into()))?;
    Ok(padded.div_ceil(GROUPED_BLOCK_M) * GROUPED_BLOCK_M)
}

pub(crate) fn project_bf16_batch(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    input: &GpuTensor,
    output: &GpuTensor,
    rows: usize,
) -> Result<(), DispatchError> {
    if weight.dtype != DType::BF16 {
        return Err(DispatchError::UnsupportedVariant {
            family: "qwen4-program",
            variant: "non-bf16-stateful-projection",
            arch: "",
            quant: "non-BF16",
        });
    }
    let result = if rows > 1 {
        gpu.gemm_bf16_xf32_multirow(weight.buf, input, output, weight.m, weight.k, rows)
    } else {
        gpu.gemv_bf16_xf32(weight.buf, input, output, weight.m, weight.k)
    };
    hip(result)
}

pub fn execute_final_hyper(
    gpu: &mut Gpu,
    dims: Qwen4ProgramDims,
    weights: &Qwen4HyperReadWeights<'_>,
    streams: &GpuTensor,
    scratch: &Qwen4LayerScratch<'_>,
    rows: usize,
) -> Result<(), DispatchError> {
    validate_final_hyper(dims, weights, streams, scratch, rows)?;
    hc_read_batch(gpu, dims, weights, streams, scratch, rows)
}

/// Validate all final HC operands without launching a kernel.
///
/// Forward callers run this before constructing or executing mutable layer
/// steps so a missing final resource cannot surface after state mutation.
pub fn validate_final_hyper(
    dims: Qwen4ProgramDims,
    weights: &Qwen4HyperReadWeights<'_>,
    streams: &GpuTensor,
    scratch: &Qwen4LayerScratch<'_>,
    rows: usize,
) -> Result<(), DispatchError> {
    validate_dims(dims)?;
    if rows == 0 {
        return Err(DispatchError::Hip("Qwen4 final hyper rows are zero".into()));
    }
    let wide = dims.wide();
    let low = dims.hc_lowrank;
    let wide_rows = checked_mul(rows, wide, "final hyper input")?;
    let hidden = checked_mul(rows, dims.hidden, "final hyper output")?;
    require_tensor(weights.norm, wide, DType::BF16, "final HC norm")?;
    require_dense_weight(
        &weights.input_mix_down,
        low,
        wide,
        DType::BF16,
        "final HC input mix down",
    )?;
    require_dense_weight(
        &weights.input_mix_up,
        wide,
        low,
        DType::BF16,
        "final HC input mix up",
    )?;
    require_tensor(streams, wide_rows, DType::F32, "final hyper streams")?;
    require_tensor(scratch.hc_mixed, hidden, DType::F32, "final hyper mixed")
}

/// Validate the LM-head weight, hidden stream, and requested output before
/// any stateful layer program is launched.
pub fn validate_lm_head(
    weight: &WeightRef<'_>,
    hidden_batch: &GpuTensor,
    logits: &GpuTensor,
    rows: usize,
    requested_rows: usize,
) -> Result<(), DispatchError> {
    if rows == 0 || requested_rows == 0 || requested_rows > rows {
        return Err(DispatchError::Hip(
            "Qwen4 LM-head row request is invalid".into(),
        ));
    }
    if weight.m == 0 || weight.k == 0 {
        return Err(DispatchError::Hip("Qwen4 LM-head geometry is empty".into()));
    }
    if !matches!(weight.dtype, DType::BF16 | DType::F32) {
        return Err(DispatchError::UnsupportedVariant {
            family: "qwen4-program",
            variant: "lm-head",
            arch: "",
            quant: "unsupported",
        });
    }
    require_dense_weight(weight, weight.m, weight.k, weight.dtype, "LM-head weight")?;
    let hidden_elements = checked_mul(rows, weight.k, "LM-head hidden")?;
    require_tensor(hidden_batch, hidden_elements, DType::F32, "LM-head hidden")?;
    let output_elements = checked_mul(requested_rows, weight.m, "LM-head output")?;
    require_tensor(logits, output_elements, DType::F32, "LM-head output")?;
    Ok(())
}

/// Compute language logits for either every requested row or only the final
/// row of an already-batched hidden stream.  The ordinary public chunk path
/// passes `requested_rows == rows`, preserving its all-row capture contract;
/// a consumer that only needs the final AR row can pass `1` without a second
/// forward implementation.
pub fn execute_lm_head(
    gpu: &mut Gpu,
    weight: &WeightRef<'_>,
    hidden_batch: &GpuTensor,
    logits: &GpuTensor,
    rows: usize,
    requested_rows: usize,
) -> Result<(), DispatchError> {
    validate_lm_head(weight, hidden_batch, logits, rows, requested_rows)?;
    if requested_rows == rows {
        match weight.dtype {
            DType::BF16 => project_bf16_batch(gpu, weight, hidden_batch, logits, rows),
            DType::F32 => hip(gpu.gemm_f32_batched(
                weight.buf,
                hidden_batch,
                logits,
                weight.m,
                weight.k,
                rows,
            )),
            _dtype => Err(DispatchError::UnsupportedVariant {
                family: "qwen4-program",
                variant: "lm-head",
                arch: "",
                quant: "unsupported",
            }),
        }
    } else if requested_rows == 1 {
        let input = view(
            hidden_batch,
            checked_mul(rows - 1, weight.k, "final LM-head row")?,
            weight.k,
        );
        match weight.dtype {
            DType::BF16 => hip(gpu.gemv_bf16_xf32(weight.buf, &input, logits, weight.m, weight.k)),
            DType::F32 => {
                hip(gpu.gemm_f32_batched(weight.buf, &input, logits, weight.m, weight.k, 1))
            }
            _dtype => Err(DispatchError::UnsupportedVariant {
                family: "qwen4-program",
                variant: "lm-head",
                arch: "",
                quant: "unsupported",
            }),
        }
    } else {
        Err(DispatchError::Hip(
            "Qwen4 LM-head supports all rows or final row only".into(),
        ))
    }
}

fn hc_read_batch(
    gpu: &mut Gpu,
    dims: Qwen4ProgramDims,
    weights: &Qwen4HyperReadWeights<'_>,
    input: &GpuTensor,
    scratch: &Qwen4LayerScratch<'_>,
    rows: usize,
) -> Result<(), DispatchError> {
    let wide = dims.wide();
    let low = dims.hc_lowrank;
    let input_batch = input.sub_offset(0, rows * wide);
    let normalized = scratch.hc_normalized.sub_offset(0, rows * wide);
    let low_batch = scratch.hc_low.sub_offset(0, rows * low);
    let up_batch = scratch.hc_up.sub_offset(0, rows * wide);
    hip(qwen4_hc_norm(
        gpu,
        &Qwen4HcNorm {
            input: &input_batch,
            norm_weight: weights.norm,
            normalized: &normalized,
            branches: dims.hc_count,
            hidden: dims.hidden,
        },
    ))?;
    project_bf16_batch(gpu, &weights.input_mix_down, &normalized, &low_batch, rows)?;
    for row in 0..rows {
        let low_row = view(&low_batch, row * low, low);
        hip(qwen4_gdn_bf16_roundtrip(
            gpu,
            &Qwen4GdnBf16Roundtrip {
                input: &low_row,
                scratch: scratch.gdn_bf16,
                output: &low_row,
                elements: low,
            },
        ))?;
    }
    hip(qwen4_scale(
        gpu,
        &Qwen4Scale {
            values: &low_batch,
            scale: 1.0 / dims.hc_count as f32,
        },
    ))?;
    for row in 0..rows {
        let low_row = view(&low_batch, row * low, low);
        hip(qwen4_gdn_bf16_roundtrip(
            gpu,
            &Qwen4GdnBf16Roundtrip {
                input: &low_row,
                scratch: scratch.gdn_bf16,
                output: &low_row,
                elements: low,
            },
        ))?;
    }
    hip(gpu.silu_f32(&low_batch, &low_batch))?;
    for row in 0..rows {
        let low_row = view(&low_batch, row * low, low);
        hip(qwen4_gdn_bf16_roundtrip(
            gpu,
            &Qwen4GdnBf16Roundtrip {
                input: &low_row,
                scratch: scratch.gdn_bf16,
                output: &low_row,
                elements: low,
            },
        ))?;
    }
    project_bf16_batch(gpu, &weights.input_mix_up, &low_batch, &up_batch, rows)?;
    let mixed = scratch.hc_mixed.sub_offset(0, rows * dims.hidden);
    hip(qwen4_hc_read_projected(
        gpu,
        &Qwen4HcReadProjected {
            input: &input_batch,
            norm_weight: weights.norm,
            up: &up_batch,
            normalized: &normalized,
            mixed: &mixed,
            branches: dims.hc_count,
            hidden: dims.hidden,
        },
    ))?;
    Ok(())
}

fn hc_write_batch(
    gpu: &mut Gpu,
    dims: Qwen4ProgramDims,
    weights: &Qwen4HyperWriteWeights<'_>,
    streams: &GpuTensor,
    mixed: &GpuTensor,
    scratch: &Qwen4LayerScratch<'_>,
    rows: usize,
) -> Result<(), DispatchError> {
    let wide = dims.wide();
    let streams_batch = streams.sub_offset(0, rows * wide);
    let mixed_batch = mixed.sub_offset(0, rows * dims.hidden);
    let normalized = scratch.hc_normalized.sub_offset(0, rows * wide);
    let gates = scratch.hc_gates.sub_offset(0, rows * dims.hc_count);
    hip(qwen4_hc_norm(
        gpu,
        &Qwen4HcNorm {
            input: &streams_batch,
            norm_weight: weights.norm,
            normalized: &normalized,
            branches: dims.hc_count,
            hidden: dims.hidden,
        },
    ))?;
    project_bf16_batch(gpu, &weights.block_inject, &normalized, &gates, rows)?;
    hip(qwen4_hc_write(
        gpu,
        &Qwen4HcWrite {
            input: &streams_batch,
            normalized: &normalized,
            mixed: &mixed_batch,
            gates: &gates,
            output: &streams_batch,
            branches: dims.hc_count,
            hidden: dims.hidden,
        },
    ))?;
    Ok(())
}

fn gdn_batch(
    gpu: &mut Gpu,
    dims: Qwen4ProgramDims,
    weights: &Qwen4GdnWeights<'_>,
    state: &mut Qwen4GdnState<'_>,
    input: &GpuTensor,
    output: &GpuTensor,
    scratch: &Qwen4LayerScratch<'_>,
    rows: usize,
    start_position: usize,
) -> Result<(), DispatchError> {
    let qkv = dims.gdn_qkv();
    let value = dims.gdn_value();
    let qk = dims.gdn_qk();
    let projection = scratch.projection.sub_offset(0, rows * qkv);
    let projection2 = scratch.projection2.sub_offset(0, rows * qkv);
    project_bf16_batch(gpu, &weights.qkv, input, &projection, rows)?;
    let a_batch = scratch
        .gdn_a
        .sub_offset(0, rows * dims.linear_num_value_heads);
    let b_batch = scratch
        .gdn_b
        .sub_offset(0, rows * dims.linear_num_value_heads);
    let z_batch = scratch.gdn_z.sub_offset(0, rows * value);
    project_bf16_batch(gpu, &weights.in_proj_a, input, &a_batch, rows)?;
    project_bf16_batch(gpu, &weights.in_proj_b, input, &b_batch, rows)?;
    project_bf16_batch(gpu, &weights.z, input, &z_batch, rows)?;
    let history_rows = dims.linear_conv_kernel_dim.saturating_sub(1);
    for row in 0..rows {
        let position = start_position.saturating_add(row);
        let cursor = if history_rows == 0 {
            0
        } else {
            position % history_rows
        };
        let projection_row = view(&projection, row * qkv, qkv);
        let projection2_row = view(&projection2, row * qkv, qkv);
        hip(qwen4_gdn_conv(
            gpu,
            &Qwen4GdnConv {
                input: &projection_row,
                kernel: weights.conv,
                history: state.conv,
                output: &projection2_row,
                next_history: state.conv,
                channels: qkv,
                history_rows,
                kernel_size: dims.linear_conv_kernel_kernel_dim(),
                cursor,
            },
        ))?;
        let a_row = view(
            &a_batch,
            row * dims.linear_num_value_heads,
            dims.linear_num_value_heads,
        );
        let b_row = view(
            &b_batch,
            row * dims.linear_num_value_heads,
            dims.linear_num_value_heads,
        );
        let gate_row = view(
            scratch.gdn_gate,
            row * dims.linear_num_value_heads,
            dims.linear_num_value_heads,
        );
        let beta_row = view(
            scratch.gdn_beta,
            row * dims.linear_num_value_heads,
            dims.linear_num_value_heads,
        );
        hip(qwen4_gdn_params(
            gpu,
            &Qwen4GdnParams {
                a: &a_row,
                b: &b_row,
                a_log: weights.a_log,
                dt_bias: weights.dt_bias,
                gate: &gate_row,
                beta: &beta_row,
            },
            dims.linear_num_value_heads,
        ))?;
        let q = view(&projection2_row, 0, qk);
        let k = view(&projection2_row, qk, qk);
        let v = view(&projection2_row, 2 * qk, value);
        let recurrent_row = state.recurrent;
        let recurrent_output = view(scratch.gdn_recurrent_output, row * value, value);
        hip(qwen4_gdn_step(
            gpu,
            &Qwen4GdnStep {
                q: &q,
                k: &k,
                v: &v,
                gate: &gate_row,
                beta: &beta_row,
                state: recurrent_row,
                output: &recurrent_output,
                key_heads: dims.linear_num_key_heads,
                value_heads: dims.linear_num_value_heads,
                key_dim: dims.linear_key_head_dim,
                value_dim: dims.linear_value_head_dim,
            },
        ))?;
        let recurrent_bf16 = view(scratch.gdn_bf16, 0, value);
        hip(qwen4_gdn_bf16_roundtrip(
            gpu,
            &Qwen4GdnBf16Roundtrip {
                input: &recurrent_output,
                scratch: &recurrent_bf16,
                output: &recurrent_output,
                elements: value,
            },
        ))?;
        let z_row = view(&z_batch, row * value, value);
        let gdn_output = view(scratch.gdn_output, row * value, value);
        hip(qwen4_gdn_gate(
            gpu,
            &Qwen4GdnGate {
                recurrent_output: &recurrent_output,
                z: &z_row,
                norm: weights.norm,
                output: &gdn_output,
                value_heads: dims.linear_num_value_heads,
                value_dim: dims.linear_value_head_dim,
            },
        ))?;
    }
    let output_batch = output.sub_offset(0, rows * dims.hidden);
    project_bf16_batch(
        gpu,
        &weights.output,
        &scratch.gdn_output.sub_offset(0, rows * value),
        &output_batch,
        rows,
    )?;
    for row in 0..rows {
        let output_row = view(&output_batch, row * dims.hidden, dims.hidden);
        let bf16_row = view(scratch.gdn_bf16, 0, dims.hidden);
        hip(qwen4_gdn_bf16_roundtrip(
            gpu,
            &Qwen4GdnBf16Roundtrip {
                input: &output_row,
                scratch: &bf16_row,
                output: &output_row,
                elements: dims.hidden,
            },
        ))?;
    }
    Ok(())
}

impl Qwen4ProgramDims {
    #[inline]
    fn linear_conv_kernel_kernel_dim(self) -> usize {
        self.linear_conv_kernel_dim
    }
}

fn qsa_batch(
    gpu: &mut Gpu,
    dims: Qwen4ProgramDims,
    weights: &Qwen4QsaWeights<'_>,
    state: &mut Qwen4QsaState<'_>,
    input: &GpuTensor,
    scratch: &Qwen4LayerScratch<'_>,
    rows: usize,
) -> Result<(), DispatchError> {
    let index_dim = dims.indexer_head_dim;
    let index_q_width = dims.indexer_n_heads * index_dim;
    let index_width = dims.index_width();
    let q_width = dims.q_width();
    let kv_width = dims.kv_width();
    let index_batch = scratch.qsa_index.sub_offset(0, rows * index_width);
    let qgate_batch = scratch.qsa_qgate.sub_offset(0, rows * 2 * q_width);
    let k_batch = scratch.qsa_k.sub_offset(0, rows * kv_width);
    let v_batch = scratch.qsa_v.sub_offset(0, rows * kv_width);
    project_bf16_batch(gpu, &weights.indexer_qk, input, &index_batch, rows)?;
    project_bf16_batch(gpu, &weights.q, input, &qgate_batch, rows)?;
    project_bf16_batch(gpu, &weights.k, input, &k_batch, rows)?;
    project_bf16_batch(gpu, &weights.v, input, &v_batch, rows)?;
    let attention_batch = scratch.attention_output.sub_offset(0, rows * dims.hidden);
    let initial_position = state.position;

    let qsa_output_batch = scratch.qsa_output.sub_offset(0, rows * q_width);
    for row in 0..rows {
        let position = initial_position
            .checked_add(row)
            .ok_or_else(|| DispatchError::Hip("QSA row position overflows".into()))?;
        let index_row = view(&index_batch, row * index_width, index_width);
        let index_q = view(&index_row, 0, index_q_width);
        let index_k = view(&index_row, index_q_width, dims.indexer_kv_heads * index_dim);
        hip(qwen4_qsa_norm_rope(
            gpu,
            &Qwen4QsaNormRope {
                values: &index_q,
                norm: weights.indexer_q_norm,
                heads: dims.indexer_n_heads,
                head_dim: index_dim,
                head_stride: index_dim,
                position: position,
                rotary_dim: index_dim.min(64),
            },
        ))?;
        hip(gpu.bf16_round_trip_f32(&index_k))?;
        hip(gpu.memcpy_dtod_at_auto(
            &state.raw_index_keys.buf,
            position * dims.indexer_kv_heads * index_dim * 4,
            &index_k.buf,
            0,
            dims.indexer_kv_heads * index_dim * 4,
        ))?;
        let qgate_row = view(&qgate_batch, row * 2 * q_width, 2 * q_width);
        let k_row = view(&k_batch, row * kv_width, kv_width);
        let v_row = view(&v_batch, row * kv_width, kv_width);
        hip(qwen4_qsa_norm_rope(
            gpu,
            &Qwen4QsaNormRope {
                values: &qgate_row,
                norm: weights.q_norm,
                heads: dims.num_attention_heads,
                head_dim: dims.head_dim,
                head_stride: 2 * dims.head_dim,
                position: position,
                rotary_dim: dims.head_dim.min(64),
            },
        ))?;
        hip(qwen4_qsa_norm_rope(
            gpu,
            &Qwen4QsaNormRope {
                values: &k_row,
                norm: weights.k_norm,
                heads: dims.num_key_value_heads,
                head_dim: dims.head_dim,
                head_stride: dims.head_dim,
                position: position,
                rotary_dim: dims.head_dim.min(64),
            },
        ))?;
        hip(qwen4_qsa_cache_append(
            gpu,
            &Qwen4QsaCacheAppend {
                key: &k_row,
                value: &v_row,
                full_keys: state.full_keys,
                full_values: state.full_values,
                position: position,
                kv_width,
            },
        ))?;
        let visible = position
            .checked_add(1)
            .ok_or_else(|| DispatchError::Hip("QSA position overflows".into()))?;
        let complete = visible / dims.indexer_compress_ratio;
        if complete > 0 {
            hip(qwen4_qsa_pool_rope(
                gpu,
                &Qwen4QsaPoolRope {
                    raw_keys: state.raw_index_keys,
                    pooled: state.pooled_keys,
                    norm: Some(weights.indexer_k_norm),
                    block_count: complete,
                    compress: dims.indexer_compress_ratio,
                    index_dim: dims.indexer_kv_heads * index_dim,
                },
            ))?;
        }
        let budget_blocks = dims.indexer_budget / dims.indexer_compress_ratio;
        hip(qwen4_qsa_select(
            gpu,
            &Qwen4QsaSelect {
                query: &index_q,
                pooled: state.pooled_keys,
                selected: state.selected_indices,
                block_count: complete,
                index_heads: dims.indexer_n_heads,
                index_dim,
                budget_blocks,
                compress: dims.indexer_compress_ratio,
                visible,
                capacity: state.selected_capacity,
            },
        ))?;
        let selected = (budget_blocks.min(complete) * dims.indexer_compress_ratio + visible
            - complete * dims.indexer_compress_ratio)
            .min(state.selected_capacity);
        let qsa_row = view(&qsa_output_batch, row * q_width, q_width);
        hip(qwen4_qsa_attention(
            gpu,
            &Qwen4QsaAttention {
                q_with_gate: &qgate_row,
                full_keys: state.full_keys,
                full_values: state.full_values,
                selected: state.selected_indices,
                output: &qsa_row,
                n_heads: dims.num_attention_heads,

                n_kv_heads: dims.num_key_value_heads,
                head_dim: dims.head_dim,
                selected_len: selected,
                full_capacity: state.full_capacity,
            },
        ))?;
        state.full_len = visible;
        state.raw_len = visible;
        state.pooled_len = complete;
        state.selected_len = selected;
        state.position = visible;
    }
    state.position = initial_position
        .checked_add(rows)
        .ok_or_else(|| DispatchError::Hip("QSA batch position overflows".into()))?;
    project_bf16_batch(
        gpu,
        &weights.output,
        &qsa_output_batch,
        &attention_batch,
        rows,
    )
}
fn validate_dims(dims: Qwen4ProgramDims) -> Result<(), DispatchError> {
    let products = [
        (dims.hidden, dims.hc_count, "wide"),
        (dims.indexer_n_heads, dims.indexer_head_dim, "indexer Q"),
        (dims.indexer_kv_heads, dims.indexer_head_dim, "indexer K"),
        (dims.num_attention_heads, dims.head_dim, "Q width"),
        (dims.num_key_value_heads, dims.head_dim, "KV width"),
        (
            dims.linear_num_key_heads,
            dims.linear_key_head_dim,
            "GDN QK",
        ),
        (
            dims.linear_num_value_heads,
            dims.linear_value_head_dim,
            "GDN V",
        ),
    ];
    for (a, b, label) in products {
        checked_mul(a, b, label)?;
    }
    checked_mul(2, dims.gdn_qk(), "GDN QKV")?
        .checked_add(dims.gdn_value())
        .ok_or_else(|| DispatchError::Hip("Qwen4 GDN QKV overflows".into()))?;
    if dims.hc_count == 0
        || dims.hc_lowrank == 0
        || dims.indexer_compress_ratio == 0
        || dims.indexer_budget == 0
        || dims.moe_intermediate == 0
        || dims.shared_intermediate == 0
        || dims.ple_conv_kernel_dim == 0
        || dims.num_experts != QWEN4_EXPERTS
        || dims.experts_per_token != QWEN4_TOP_K
        || dims.norm_eps <= 0.0
    {
        return Err(DispatchError::Hip(
            "Qwen4 layer descriptor has invalid geometry".into(),
        ));
    }
    if dims.experts_per_token > dims.num_experts {
        return Err(DispatchError::Hip(
            "Qwen4 top-k exceeds expert capacity".into(),
        ));
    }
    Ok(())
}

fn require_tensor(
    tensor: &GpuTensor,
    elements: usize,
    dtype: DType,
    name: &'static str,
) -> Result<(), DispatchError> {
    if tensor.dtype != dtype {
        return Err(DispatchError::Hip(format!(
            "Qwen4 {name} dtype mismatch: expected {dtype:?}, got {:?}",
            tensor.dtype
        )));
    }
    let bytes = elements
        .checked_mul(dtype.size())
        .ok_or_else(|| DispatchError::Hip(format!("Qwen4 {name} size overflows")))?;
    if tensor.numel() < elements || tensor.buf.size() < bytes {
        return Err(DispatchError::Hip(format!(
            "Qwen4 {name} capacity too small: need {elements} elements, have {}",
            tensor.numel()
        )));
    }
    Ok(())
}

fn require_raw_bytes(
    tensor: &GpuTensor,
    bytes: usize,
    name: &'static str,
) -> Result<(), DispatchError> {
    if tensor.dtype != DType::Raw {
        return Err(DispatchError::Hip(format!(
            "Qwen4 {name} dtype mismatch: expected Raw, got {:?}",
            tensor.dtype
        )));
    }
    if tensor.numel() < bytes || tensor.buf.size() < bytes {
        return Err(DispatchError::Hip(format!(
            "Qwen4 {name} capacity too small: need {bytes} bytes, have {}",
            tensor.buf.size()
        )));
    }
    Ok(())
}

fn require_weight(
    weight: &WeightRef<'_>,
    m: usize,
    k: usize,
    name: &'static str,
) -> Result<(), DispatchError> {
    if weight.m != m || weight.k != k {
        return Err(DispatchError::Hip(format!(
            "Qwen4 {name} geometry mismatch: expected {m}x{k}, got {}x{}",
            weight.m, weight.k
        )));
    }
    Ok(())
}

fn require_dense_weight(
    weight: &WeightRef<'_>,
    m: usize,
    k: usize,
    dtype: DType,
    name: &'static str,
) -> Result<(), DispatchError> {
    require_weight(weight, m, k, name)?;
    if weight.dtype != dtype {
        return Err(DispatchError::Hip(format!(
            "Qwen4 {name} dtype mismatch: expected {dtype:?}, got {:?}",
            weight.dtype
        )));
    }
    let bytes = m
        .checked_mul(k)
        .and_then(|elements| elements.checked_mul(dtype.size()))
        .ok_or_else(|| DispatchError::Hip(format!("Qwen4 {name} size overflows")))?;
    if weight.buf.buf.size() < bytes {
        return Err(DispatchError::Hip(format!(
            "Qwen4 {name} capacity too small: need {bytes} bytes, have {}",
            weight.buf.buf.size()
        )));
    }
    Ok(())
}

fn overlap(a: &GpuTensor, b: &GpuTensor) -> bool {
    let ap = a.buf.as_ptr() as usize;
    let bp = b.buf.as_ptr() as usize;
    if ap == 0 || bp == 0 {
        return false;
    }
    let ae = ap.saturating_add(a.byte_size());
    let be = bp.saturating_add(b.byte_size());
    ap < be && bp < ae
}

fn require_disjoint(a: &GpuTensor, b: &GpuTensor, name: &'static str) -> Result<(), DispatchError> {
    if overlap(a, b) {
        return Err(DispatchError::Hip(format!(
            "Qwen4 scratch alias is not permitted: {name}"
        )));
    }
    Ok(())
}

/// Fail-closed metadata and resource preflight for a complete layer.
///
/// This function performs no HIP work.  It is called by the public Step
/// interpreter before the first operation, and again by `execute_layer` so a
/// directly-called descriptor has the same safety boundary.
pub fn validate_layer(op: &Qwen4LayerOp<'_>) -> Result<(), DispatchError> {
    let dims = op.dims;
    validate_dims(dims)?;
    if op.rows == 0 {
        return Err(DispatchError::Hip("Qwen4 layer rows are zero".into()));
    }
    let wide = checked_mul(dims.hc_count, dims.hidden, "wide")?;
    let slots = checked_mul(op.rows, dims.experts_per_token, "MoE slots")?;
    let grouped = if op.rows > 1 {
        grouped_m_total_bound(slots, dims.num_experts)?
    } else {
        0
    };
    let row_hidden = checked_mul(op.rows, dims.hidden, "row hidden")?;
    let row_wide = checked_mul(op.rows, wide, "row wide")?;
    let shared_rows = checked_mul(op.rows, dims.shared_intermediate, "shared rows")?;
    let routed_rows = checked_mul(slots, dims.moe_intermediate, "routed rows")?;
    let route_expanded = checked_mul(slots, dims.hidden, "expanded route")?;

    require_tensor(op.scratch.streams, row_wide, DType::F32, "streams")?;
    for (tensor, elements, name) in [
        (op.scratch.hc_normalized, row_wide, "HC normalized"),
        (op.scratch.hc_up, row_wide, "HC up"),
        (op.scratch.hc_mixed, row_hidden, "HC mixed"),
        (op.scratch.hc_gates, op.rows * dims.hc_count, "HC gates"),
        (
            op.scratch.projection,
            op.rows * dims.gdn_qkv(),
            "projection",
        ),
        (
            op.scratch.projection2,
            op.rows * dims.gdn_qkv(),
            "projection2",
        ),
        (
            op.scratch.gdn_a,
            op.rows * dims.linear_num_value_heads,
            "GDN A",
        ),
        (
            op.scratch.gdn_b,
            op.rows * dims.linear_num_value_heads,
            "GDN B",
        ),
        (
            op.scratch.gdn_gate,
            op.rows * dims.linear_num_value_heads,
            "GDN gate",
        ),
        (
            op.scratch.gdn_beta,
            op.rows * dims.linear_num_value_heads,
            "GDN beta",
        ),
        (
            op.scratch.gdn_recurrent_output,
            op.rows * dims.gdn_value(),
            "GDN recurrent",
        ),
        (op.scratch.gdn_z, op.rows * dims.gdn_value(), "GDN Z"),
        (
            op.scratch.gdn_output,
            op.rows * dims.gdn_value(),
            "GDN output",
        ),
        (
            op.scratch.qsa_index,
            op.rows * dims.index_width(),
            "QSA index",
        ),
        (
            op.scratch.qsa_qgate,
            op.rows * 2 * dims.q_width(),
            "QSA Q/G",
        ),
        (op.scratch.qsa_k, op.rows * dims.kv_width(), "QSA K"),
        (op.scratch.qsa_v, op.rows * dims.kv_width(), "QSA V"),
        (
            op.scratch.qsa_output,
            op.rows * dims.q_width(),
            "QSA output",
        ),
        (op.scratch.attention_output, row_hidden, "attention output"),
        (op.scratch.moe_output, row_hidden, "MoE output"),
        (
            op.scratch.moe_shared_output,
            row_hidden,
            "MoE shared projection",
        ),
        (
            op.scratch.moe_router_logits,
            op.rows * dims.num_experts,
            "router logits",
        ),
        (op.scratch.moe_x_rot, row_hidden, "MoE rotated input"),
        (op.scratch.moe_scalar, shared_rows, "shared selector"),
        (op.scratch.moe_gate, shared_rows, "shared gate"),
        (op.scratch.moe_up, shared_rows, "shared up"),
        (op.scratch.moe_hidden, shared_rows, "shared hidden"),
        (op.scratch.moe_gate_batch, routed_rows, "routed gate"),
        (op.scratch.moe_up_batch, routed_rows, "routed up"),
        (op.scratch.moe_rot_batch, routed_rows, "routed activation"),
        (op.scratch.moe_down_expanded, route_expanded, "routed down"),
    ] {
        require_tensor(tensor, elements, DType::F32, name)?;
    }
    if op.rows == 1 {
        require_tensor(
            op.scratch.moe_gate_up,
            checked_mul(2, dims.moe_intermediate, "decode gate/up")?,
            DType::F32,
            "decode gate/up scratch",
        )?;
    }
    require_tensor(op.scratch.gdn_bf16, dims.hidden, DType::BF16, "GDN BF16")?;
    require_tensor(
        op.scratch.hc_low,
        checked_mul(op.rows, dims.hc_lowrank, "HC low")?,
        DType::F32,
        "HC low",
    )?;
    require_tensor(
        op.scratch.projection2,
        checked_mul(op.rows, dims.gdn_qkv(), "projection2")?,
        DType::F32,
        "projection2",
    )?;
    require_raw_bytes(
        op.scratch.moe_topk_indices,
        checked_mul(slots, std::mem::size_of::<u32>(), "top-k index")?,
        "top-k indices",
    )?;
    require_tensor(
        op.scratch.moe_topk_weights,
        slots,
        DType::F32,
        "top-k weights",
    )?;
    if op.rows > 1 {
        require_raw_bytes(
            op.scratch.moe_expert_token_counts,
            checked_mul(
                dims.num_experts,
                std::mem::size_of::<u32>(),
                "expert counts",
            )?,
            "expert counts",
        )?;
        require_raw_bytes(
            op.scratch.moe_expert_offsets,
            checked_mul(
                dims.num_experts
                    .checked_add(1)
                    .ok_or_else(|| DispatchError::Hip("expert offsets overflows".into()))?,
                std::mem::size_of::<u32>(),
                "expert offsets",
            )?,
            "expert offsets",
        )?;
        require_raw_bytes(
            op.scratch.moe_sorted_slot_index,
            checked_mul(grouped, std::mem::size_of::<u32>(), "sorted slots")?,
            "sorted slots",
        )?;
        require_raw_bytes(
            op.scratch.moe_expert_tile_ids,
            checked_mul(
                grouped / GROUPED_BLOCK_M,
                std::mem::size_of::<u32>(),
                "expert tiles",
            )?,
            "expert tiles",
        )?;
        require_raw_bytes(
            op.scratch.moe_inverse_perm,
            checked_mul(slots, std::mem::size_of::<u32>(), "inverse permutation")?,
            "inverse permutation",
        )?;
        require_tensor(
            op.scratch.moe_y_gate_up_grouped,
            grouped * (2 * dims.moe_intermediate),
            DType::F32,
            "grouped gate/up",
        )?;
        require_tensor(
            op.scratch.moe_y_down_grouped,
            grouped * dims.hidden,
            DType::F32,
            "grouped down",
        )?;
    }

    // Streams are intentionally updated in place by the HC write.  All other
    // layer outputs must remain distinct from the residual stream and from the
    // final MoE result.
    require_disjoint(
        op.scratch.streams,
        op.scratch.moe_output,
        "streams/moe output",
    )?;
    require_disjoint(
        op.scratch.hc_mixed,
        op.scratch.moe_output,
        "HC mixed/MoE output",
    )?;
    require_disjoint(
        op.scratch.attention_output,
        op.scratch.moe_output,
        "attention/MoE output",
    )?;
    require_disjoint(
        op.scratch.moe_shared_output,
        op.scratch.moe_output,
        "MoE shared projection/residual",
    )?;

    let (linear, full) = match (
        &op.layer.attention,
        op.state_gdn.as_ref(),
        op.state_qsa.as_ref(),
    ) {
        (Qwen4AttentionWeights::Linear(weights), Some(state), None) => {
            require_tensor(
                state.recurrent,
                dims.gdn_value() * dims.linear_key_head_dim,
                DType::F32,
                "GDN recurrent state",
            )?;
            require_tensor(
                state.conv,
                dims.gdn_qkv() * dims.linear_conv_kernel_dim.saturating_sub(1),
                DType::F32,
                "GDN convolution state",
            )?;
            require_weight(&weights.qkv, dims.gdn_qkv(), dims.hidden, "GDN QKV")?;
            require_weight(
                &weights.in_proj_a,
                dims.linear_num_value_heads,
                dims.hidden,
                "GDN A projection",
            )?;
            require_weight(
                &weights.in_proj_b,
                dims.linear_num_value_heads,
                dims.hidden,
                "GDN B projection",
            )?;
            require_weight(
                &weights.z,
                dims.gdn_value(),
                dims.hidden,
                "GDN Z projection",
            )?;
            require_weight(&weights.output, dims.hidden, dims.gdn_value(), "GDN output")?;
            (true, false)
        }
        (Qwen4AttentionWeights::Full(weights), None, Some(state)) => {
            if state.position != state.full_len
                || state.position != state.raw_len
                || state.position > state.position_capacity
                || op.start_position != state.position
                || op.rows > state.position_capacity.saturating_sub(state.position)
            {
                return Err(DispatchError::Hip(
                    "Qwen4 QSA state position/capacity mismatch".into(),
                ));
            }
            if state.full_len > state.full_capacity
                || state.raw_len > state.raw_capacity
                || state.pooled_len > state.pooled_capacity
                || state.selected_len > state.selected_capacity
            {
                return Err(DispatchError::Hip(
                    "Qwen4 QSA state length exceeds capacity".into(),
                ));
            }
            require_tensor(
                state.full_keys,
                state.full_capacity * dims.kv_width(),
                DType::F32,
                "QSA full keys",
            )?;
            require_tensor(
                state.full_values,
                state.full_capacity * dims.kv_width(),
                DType::F32,
                "QSA full values",
            )?;
            require_tensor(
                state.raw_index_keys,
                state.raw_capacity * dims.indexer_kv_heads * dims.indexer_head_dim,
                DType::F32,
                "QSA raw keys",
            )?;
            require_tensor(
                state.pooled_keys,
                state.pooled_capacity * dims.indexer_kv_heads * dims.indexer_head_dim,
                DType::F32,
                "QSA pooled keys",
            )?;
            require_tensor(
                state.selected_indices,
                state.selected_capacity,
                DType::Raw,
                "QSA selected indices",
            )?;
            require_weight(
                &weights.indexer_qk,
                dims.index_width(),
                dims.hidden,
                "QSA index projection",
            )?;
            require_weight(
                &weights.q,
                2 * dims.q_width(),
                dims.hidden,
                "QSA Q/G projection",
            )?;
            require_weight(&weights.k, dims.kv_width(), dims.hidden, "QSA K projection")?;
            require_weight(&weights.v, dims.kv_width(), dims.hidden, "QSA V projection")?;
            (false, true)
        }
        _ => {
            return Err(DispatchError::Hip(
                "Qwen4 layer state/weight kind mismatch".into(),
            ))
        }
    };
    let _ = (linear, full);
    if op.layer.moe.intermediate != dims.moe_intermediate
        || op.layer.moe.table.n_experts() != dims.num_experts
    {
        return Err(DispatchError::Hip(
            "Qwen4 MoE binding geometry mismatch".into(),
        ));
    }
    Ok(())
}

/// Complete launch-free Qwen4 layer preflight.  The structural checks above
/// cover family-owned scratch/state; this boundary additionally binds the
/// live expert cache and runs the shared sealer validation plus kernel
/// selection, so every later layer is rejected before an earlier layer can
/// issue an effect.
pub fn validate_layer_for_gpu(gpu: &Gpu, op: &Qwen4LayerOp<'_>) -> Result<(), DispatchError> {
    validate_layer(op)?;
    let dims = op.dims;
    let rows = op.rows;
    let wide = checked_mul(dims.hc_count, dims.hidden, "wide")?;
    let hidden_elements = checked_mul(rows, dims.hidden, "layer hidden")?;
    let _streams = op
        .scratch
        .streams
        .sub_offset(0, checked_mul(rows, wide, "layer streams")?);
    let moe_input = op.scratch.hc_mixed.sub_offset(0, hidden_elements);
    let bound = BoundMoeExperts::from_cache(op.layer.moe.table, op.layer.moe.cache)
        .map_err(|error| DispatchError::Hip(format!("bound Qwen4 experts: {error:?}")))?;
    let ctx = crate::context::DispatchCtx::new(gpu);
    if rows == 1 {
        let params = build_moe_decode(dims, &op.layer.moe, &moe_input, op.scratch)?;
        crate::pipeline::sealed_moe::preflight_decode(bound, &ctx, params).map_err(|error| {
            DispatchError::Hip(format!("preflight Qwen4 decode MoE: {error:?}"))
        })?;
    } else {
        let router_matrix = shaped_prefix(
            op.scratch.moe_router_logits,
            rows,
            dims.num_experts,
            "MoE router matrix",
        )?;
        let params = build_moe_prefill(dims, &op.layer.moe, op.scratch, &router_matrix, rows)?;
        crate::pipeline::sealed_moe::preflight_prefill(bound, &ctx, params).map_err(|error| {
            DispatchError::Hip(format!("preflight Qwen4 prefill MoE: {error:?}"))
        })?;
    }
    Ok(())
}

/// Execute one layer over a bounded row chunk.  Matrix projections are always
/// launched with the row count; GDN recurrence and QSA cache/selection remain
/// ordered inside this central operation.
/// Execute one complete layer through the shared program boundary.
///
/// Validation and MoE sealing both happen before the first GPU launch.  The
/// resulting sealed call is retained across the stateful attention section and
/// consumed by the normal Step interpreter after the MLP input is ready.
pub fn execute_layer(gpu: &mut Gpu, op: &mut Qwen4LayerOp<'_>) -> Result<(), DispatchError> {
    validate_layer_for_gpu(gpu, op)?;
    let dims = op.dims;
    let layer = &op.layer;
    let scratch = op.scratch;
    let rows = op.rows;
    let start_position = op.start_position;
    let wide = checked_mul(dims.hc_count, dims.hidden, "wide")?;
    let stream_elements = checked_mul(rows, wide, "layer streams")?;
    let hidden_elements = checked_mul(rows, dims.hidden, "layer hidden")?;
    let streams = scratch.streams.sub_offset(0, stream_elements);
    let moe_input = scratch.hc_mixed.sub_offset(0, hidden_elements);
    let bound = BoundMoeExperts::from_cache(layer.moe.table, layer.moe.cache)
        .map_err(|error| DispatchError::Hip(format!("bound Qwen4 experts: {error:?}")))?;
    let ctx = crate::context::DispatchCtx::new(gpu);
    let router_matrix = if rows > 1 {
        Some(shaped_prefix(
            scratch.moe_router_logits,
            rows,
            dims.num_experts,
            "MoE router matrix",
        )?)
    } else {
        None
    };
    let sealed = if rows == 1 {
        // AR/decode keeps the exact indexed route and GEMV scratch contract.
        // Prefill's grouped route is only lowered when a real row tile exists.
        let params = build_moe_decode(dims, &layer.moe, &moe_input, scratch)?;
        seal_decode(bound, &ctx, params)
            .map_err(|error| DispatchError::Hip(format!("seal Qwen4 decode MoE: {error:?}")))?
    } else {
        let router_matrix = router_matrix
            .as_ref()
            .ok_or_else(|| DispatchError::Hip("Qwen4 prefill router matrix missing".into()))?;
        let params = build_moe_prefill(dims, &layer.moe, scratch, router_matrix, rows)?;
        seal_prefill(bound, &ctx, params)
            .map_err(|error| DispatchError::Hip(format!("seal Qwen4 prefill MoE: {error:?}")))?
    };

    hc_read_batch(gpu, dims, &layer.attn_hyper.read, &streams, scratch, rows)?;
    let attn_input = &moe_input;
    let attention_output = match (
        &layer.attention,
        op.state_gdn.as_mut(),
        op.state_qsa.as_mut(),
    ) {
        (Qwen4AttentionWeights::Linear(weights), Some(state), None) => {
            let out = scratch.attention_output.sub_offset(0, hidden_elements);
            gdn_batch(
                gpu,
                dims,
                weights,
                state,
                &attn_input,
                &out,
                scratch,
                rows,
                start_position,
            )?;
            out
        }
        (Qwen4AttentionWeights::Full(weights), None, Some(state)) => {
            qsa_batch(gpu, dims, weights, state, &attn_input, scratch, rows)?;
            scratch.attention_output.sub_offset(0, hidden_elements)
        }
        _ => {
            return Err(DispatchError::Hip(
                "Qwen4 layer state/weight kind mismatch".into(),
            ))
        }
    };
    hc_write_batch(
        gpu,
        dims,
        &layer.attn_hyper.write,
        &streams,
        &attention_output,
        scratch,
        rows,
    )?;
    hc_read_batch(gpu, dims, &layer.mlp_hyper.read, &streams, scratch, rows)?;
    hip(gpu.hip.memset(
        &scratch.moe_output.buf,
        0,
        checked_mul(
            hidden_elements,
            std::mem::size_of::<f32>(),
            "MoE output bytes",
        )?,
    ))?;
    execute_steps(gpu, &ctx, &mut [Step::Moe(sealed)])
        .map_err(|error| DispatchError::Hip(format!("execute Qwen4 MoE: {error:?}")))?;
    hc_write_batch(
        gpu,
        dims,
        &layer.mlp_hyper.write,
        &streams,
        scratch.moe_output,
        scratch,
        rows,
    )?;
    Ok(())
}

/// Build the indexed decode descriptor for a single-token layer invocation.
///
/// Rows-one execution deliberately keeps the old decode contract: one
/// normalized activation, GEMV-sized scratch, and `seal_decode`'s exact
/// indexed route.  Grouped prefill descriptors are built only for a real
/// multi-row tile.
fn build_moe_decode<'a>(
    dims: Qwen4ProgramDims,
    moe: &'a Qwen4MoeBinding<'a>,
    input: &'a GpuTensor,
    scratch: &'a Qwen4LayerScratch<'a>,
) -> Result<MoeParams<'a>, DispatchError> {
    let (first_gate_up, first_down) = moe
        .routed_experts
        .get(0)
        .ok_or_else(|| DispatchError::Hip("Qwen4 routed expert table is empty".into()))?;
    let experts_all_gate_up_mq4 = moe.experts_all_gate_up_mq4;
    let dtypes = MoeDtypes {
        router: moe.router.dtype,
        shared: Some(MoeSharedDtypes {
            selector: moe.shared.selector.dtype,
            gate: moe.shared.gate.dtype,
            up: moe.shared.up.dtype,
            down: moe.shared.down.dtype,
        }),
        experts_all_gate_up_mq4,
        routed_gate_up: first_gate_up.dtype,
        routed_down: first_down.dtype,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
    };
    Ok(MoeParams {
        dtypes,
        recipe: MoeRecipe::SoftmaxGatedShared,
        normalization: MoeNormalization::Provided,
        batch_size: 1,
        hidden: dims.hidden,
        mi: dims.moe_intermediate,
        k: dims.experts_per_token,
        n_exp: dims.num_experts,
        norm_topk_prob: moe.norm_topk_prob,
        x_rot_prerotated: false,
        defer_routed_combine: false,
        ep_mode: MoeEpMode::None,
        layer_idx: moe.layer_idx,
        x_norm: input,
        x_residual: scratch.moe_output,
        routed_out: None,
        skip_shared: false,
        router: moe.router,
        shared: Some(MoeSharedDecode {
            weights: MoeSharedWeights {
                selector: moe.shared.selector,
                gate: moe.shared.gate,
                up: moe.shared.up,
                down: moe.shared.down,
            },
            intermediate: dims.shared_intermediate,
            scalar: scratch.moe_scalar,
            gate_out: scratch.moe_gate,
            up_out: scratch.moe_up,
        }),
        expert_gate_up_ptrs: moe.expert_gate_up_ptrs,
        expert_down_ptrs: moe.expert_down_ptrs,
        expert_down_awq_ptrs: None,
        expert_dtype_tags: None,
        routed_gate_up_k: dims.hidden,
        routed_down_m: dims.hidden,
        routed_down_k: dims.moe_intermediate,
        routed_experts: moe.routed_experts,
        routed_gate_up_paro: None,
        routed_down_paro: None,
        router_logits: scratch.moe_router_logits,
        x_rot_local: scratch.moe_x_rot,
        gate_up_buf: scratch.moe_gate_up,
        ffn_hidden: scratch.moe_hidden,
        ffn_out: scratch.moe_shared_output,
        gate_batch: scratch.moe_gate_batch,
        up_batch: scratch.moe_up_batch,
        rot_batch: scratch.moe_rot_batch,
        topk_indices: scratch.moe_topk_indices,
        topk_weights: scratch.moe_topk_weights,
        down_expanded: scratch.moe_down_expanded,
    })
}

fn build_moe_prefill<'a>(
    dims: Qwen4ProgramDims,
    moe: &'a Qwen4MoeBinding<'a>,
    scratch: &'a Qwen4LayerScratch<'a>,
    router_matrix: &'a GpuTensor,
    rows: usize,
) -> Result<MoePrefillParams<'a>, DispatchError> {
    let slots = checked_mul(rows, dims.experts_per_token, "MoE slots")?;
    let m_total_max = grouped_m_total_bound(slots, dims.num_experts)?;
    let shared = MoeSharedPrefill {
        weights: MoeSharedWeights {
            selector: moe.shared.selector,
            gate: moe.shared.gate,
            up: moe.shared.up,
            down: moe.shared.down,
        },
        intermediate: moe.intermediate,
        scalar: scratch.moe_scalar,
        gate_out: scratch.moe_gate,
        up_out: scratch.moe_up,
        rotated: scratch.moe_hidden,
    };
    let dtypes = MoeDtypes {
        router: moe.router.dtype,
        shared: Some(MoeSharedDtypes {
            selector: moe.shared.selector.dtype,
            gate: moe.shared.gate.dtype,
            up: moe.shared.up.dtype,
            down: moe.shared.down.dtype,
        }),
        experts_all_gate_up_mq4: true,
        routed_gate_up: DType::MQ4G256V2,
        routed_down: DType::MQ4G128V2,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
    };
    let prelude = MoePrefillPrelude {
        normalization: MoeNormalization::Provided,
        router: moe.router,
        router_logits: router_matrix,
        router_scores: router_matrix,
        norm_topk_prob: moe.norm_topk_prob,
        route: PrefillRouteMode::Replicated,
        shared: Some(shared),
        q8_router_policy: MoeQ8RouterPolicy::DispatcherEntry,
    };
    Ok(MoePrefillParams {
        dtypes,
        recipe: MoeRecipe::SoftmaxGatedShared,
        prelude,
        batch_size: rows,
        mi: dims.moe_intermediate,
        down_m: dims.hidden,
        down_k: dims.moe_intermediate,
        gate_up_k: dims.hidden,
        k_top: dims.experts_per_token,
        n_exp: dims.num_experts,
        m_total_max,
        force_mq4_grouped_fp16: false,
        topk_indices: scratch.moe_topk_indices,
        topk_weights: scratch.moe_topk_weights,
        x_batch: scratch.moe_output,
        x_norm_batch: scratch.hc_mixed,
        x_rot_batch: scratch.moe_x_rot,
        expert_gate_up_ptrs: moe.expert_gate_up_ptrs,
        expert_down_ptrs: moe.expert_down_ptrs,
        routed_experts: moe.routed_experts,
        expert_down_awq_ptrs: None,
        expert_dtype_tags: None,
        gate_batch: scratch.moe_gate_batch,
        up_batch: scratch.moe_up_batch,
        rot_batch: scratch.moe_rot_batch,
        down_expanded: scratch.moe_down_expanded,
        expert_token_counts: scratch.moe_expert_token_counts,
        expert_offsets: scratch.moe_expert_offsets,
        sorted_slot_index: scratch.moe_sorted_slot_index,
        expert_tile_ids: scratch.moe_expert_tile_ids,
        inverse_perm: scratch.moe_inverse_perm,
        y_gate_up_grouped: scratch.moe_y_gate_up_grouped,
        y_down_grouped: scratch.moe_y_down_grouped,
        paro_gate_up: None,
        paro_down: None,
        down_awq_scale: None,
        routed_out: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_scratch(bytes: usize) -> GpuTensor {
        GpuTensor {
            buf: unsafe {
                hip_bridge::DeviceBuffer::from_raw(
                    std::ptr::dangling_mut::<std::ffi::c_void>(),
                    bytes,
                )
            },
            shape: vec![bytes],
            dtype: DType::Raw,
        }
    }

    #[test]
    fn raw_scratch_validator_rejects_short_byte_extent() {
        let short = raw_scratch(std::mem::size_of::<u32>() - 1);
        assert!(require_raw_bytes(&short, std::mem::size_of::<u32>(), "raw scratch").is_err());

        let exact = raw_scratch(std::mem::size_of::<u32>());
        assert!(require_raw_bytes(&exact, std::mem::size_of::<u32>(), "raw scratch").is_ok());
    }

    #[test]
    fn layer_preflight_rejects_shared_projection_alias() {
        let residual = raw_scratch(std::mem::size_of::<f32>());
        let shared_projection = raw_scratch(std::mem::size_of::<f32>());
        assert!(require_disjoint(
            &shared_projection,
            &residual,
            "MoE shared projection/residual"
        )
        .is_err());
    }
}
