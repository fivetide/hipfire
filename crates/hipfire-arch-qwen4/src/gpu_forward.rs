// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Ordinary-HIP Qwen4 forward execution.
//!
//! This module is the production device path.  The reference equations in
//! `forward.rs` are intentionally separate and are never called here.  All
//! learned projections consume the resident tensor supplied by the assembled
//! bundle; MQv2 projections are rotated and dispatched through their native
//! qt44/qt53 kernels, while recurrent/cache state remains resident on the GPU.

use crate::bundle::Qwen4Bundle;
use crate::config::{LayerType, Qwen4Config};
use crate::ple_rows::{
    PlePrefetch, PleRowLease, PleRows, PleRowsError, PLE_ROWS_PER_TOKEN, PLE_ROW_BYTES,
};
use crate::projection::{dispatch_gemv, row_stride, ProjectionView};
use crate::state::{GdnGpuState, QsaGpuState};
use crate::weights::{
    HyperConnectionWeights, MoeWeights, Qwen4LayerWeights, Qwen4Weights, TensorRef, WeightError,
};
use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::families::moe::{MoeDtypes, MoeParams, RoutedExpertWeights};
use hipfire_dispatch::pipeline::{
    execute_steps, seal_decode, BoundMoeExperts, ExpertBindingCache, ExpertMetadata,
    ExpertResource, ExpertResources, ExpertTable, Step,
};
use hipfire_dispatch::types::dtype_rotation_plan;
use hipfire_runtime::weight_manifest::ExpertSourceLayout;
use rdna_compute::qwen4::{
    qwen4_argmax, qwen4_gdn_bf16_roundtrip, qwen4_gdn_conv, qwen4_gdn_gate, qwen4_gdn_params,
    qwen4_gdn_step, qwen4_hc_norm, qwen4_hc_read, qwen4_hc_write, qwen4_qsa_attention,
    qwen4_qsa_cache_append, qwen4_qsa_norm_rope, qwen4_qsa_pool_rope, qwen4_qsa_select,
    qwen4_scale, Qwen4Argmax, Qwen4GdnBf16Roundtrip, Qwen4GdnConv, Qwen4GdnGate, Qwen4GdnParams,
    Qwen4GdnStep, Qwen4HcNorm, Qwen4HcRead, Qwen4HcWrite, Qwen4QsaAttention, Qwen4QsaCacheAppend,
    Qwen4QsaNormRope, Qwen4QsaPoolRope, Qwen4QsaSelect, Qwen4Scale,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::fmt;
use std::time::Duration;

const EPSILON: f32 = 1.0e-6;
const PLE_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);

/// Errors returned by the native ordinary-HIP path.
#[derive(Debug)]
pub enum Qwen4GpuForwardError {
    Hip(hip_bridge::HipError),
    Weights(WeightError),
    Invalid(String),
    Dispatch(String),
    Ple(String),
}

impl fmt::Display for Qwen4GpuForwardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip(error) => write!(f, "Qwen4 GPU forward HIP error: {error}"),
            Self::Weights(error) => write!(f, "Qwen4 GPU forward weights error: {error}"),
            Self::Invalid(error) => write!(f, "Qwen4 GPU forward invalid input: {error}"),
            Self::Dispatch(error) => write!(f, "Qwen4 GPU forward dispatch error: {error}"),
            Self::Ple(error) => write!(f, "Qwen4 GPU forward PLE error: {error}"),
        }
    }
}

impl std::error::Error for Qwen4GpuForwardError {}

impl From<hip_bridge::HipError> for Qwen4GpuForwardError {
    fn from(error: hip_bridge::HipError) -> Self {
        Self::Hip(error)
    }
}

impl From<WeightError> for Qwen4GpuForwardError {
    fn from(error: WeightError) -> Self {
        Self::Weights(error)
    }
}

fn invalid(message: impl Into<String>) -> Qwen4GpuForwardError {
    Qwen4GpuForwardError::Invalid(message.into())
}

fn checked_bytes(rows: usize, stride: usize) -> Result<usize, Qwen4GpuForwardError> {
    crate::projection::checked_bytes(rows, stride)
        .ok_or_else(|| invalid("packed row byte size overflow"))
}

fn f32_view(tensor: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    debug_assert_eq!(tensor.dtype, DType::F32);
    tensor.sub_offset(offset, len)
}

fn view(tensor: &GpuTensor, offset: usize, len: usize) -> GpuTensor {
    tensor.sub_offset(offset, len)
}

fn weight_dims(reference: &TensorRef) -> Result<(usize, usize), Qwen4GpuForwardError> {
    if reference.shape.len() != 2 {
        return Err(invalid(format!(
            "projection {} has logical shape {:?}, expected rank two",
            reference.name, reference.shape
        )));
    }
    Ok((reference.shape[0], reference.shape[1]))
}

pub(crate) struct Qwen4ExpertView {
    gate_up: ProjectionView,
    down: ProjectionView,
}

struct ExpertViewSet<'a> {
    experts: &'a [Qwen4ExpertView],
}

impl RoutedExpertWeights for ExpertViewSet<'_> {
    fn len(&self) -> usize {
        self.experts.len()
    }

    fn get(&self, expert_idx: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
        self.experts
            .get(expert_idx)
            .map(|expert| (expert.gate_up.dispatch_ref(), expert.down.dispatch_ref()))
    }
}

/// Architecture-owned sealed-MoE resources.  Pointer tables and expert views
/// are built once with the assembled resident weights and reused for every
/// token; no per-token resource or metadata allocation occurs.
pub(crate) struct Qwen4MoeLayerRuntime {
    table: ExpertTable,
    cache: ExpertBindingCache,
    router: ProjectionView,
    shared_scalar: ProjectionView,
    shared_gate: ProjectionView,
    shared_up: ProjectionView,
    shared_down: ProjectionView,
    experts: Vec<Qwen4ExpertView>,
    expert_gate_up_ptrs: GpuTensor,
    expert_down_ptrs: GpuTensor,
}

impl RoutedExpertWeights for Qwen4MoeLayerRuntime {
    fn len(&self) -> usize {
        self.experts.len()
    }

    fn get(&self, expert_idx: usize) -> Option<(WeightRef<'_>, WeightRef<'_>)> {
        self.experts
            .get(expert_idx)
            .map(|expert| (expert.gate_up.dispatch_ref(), expert.down.dispatch_ref()))
    }
}

impl Qwen4MoeLayerRuntime {
    fn new(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        layer: &Qwen4LayerWeights,
        config: &Qwen4Config,
    ) -> Result<Self, Qwen4GpuForwardError> {
        Self::from_moe(gpu, weights, &layer.moe, layer.layer, config)
    }

    pub(crate) fn from_moe(
        gpu: &mut Gpu,
        weights: &Qwen4Weights,
        moe: &MoeWeights,
        layer_label: usize,
        config: &Qwen4Config,
    ) -> Result<Self, Qwen4GpuForwardError> {
        let router_source = weights.resident(&moe.gate)?;
        let scalar_source = weights.resident(&moe.shared_gate_scalar)?;
        let shared_gate_source = weights.resident(&moe.shared_gate)?;
        let shared_up_source = weights.resident(&moe.shared_up)?;
        let shared_down_source = weights.resident(&moe.shared_down)?;
        let router =
            ProjectionView::from_source(router_source, config.num_experts, config.hidden_size);
        let shared_scalar = ProjectionView::from_source(scalar_source, 1, config.hidden_size);
        let shared_gate = ProjectionView::from_source(
            shared_gate_source,
            config.shared_expert_intermediate_size,
            config.hidden_size,
        );
        let shared_up = ProjectionView::from_source(
            shared_up_source,
            config.shared_expert_intermediate_size,
            config.hidden_size,
        );
        let shared_down = ProjectionView::from_source(
            shared_down_source,
            config.hidden_size,
            config.shared_expert_intermediate_size,
        );

        let gate_up_source = weights.resident(&moe.experts_gate_up)?;
        let down_source = weights.resident(&moe.experts_down)?;
        let gate_up_dtype = gate_up_source.dtype;
        let down_dtype = down_source.dtype;
        let gate_up_rows = 2 * config.moe_intermediate_size;
        let gate_up_row_stride = row_stride(gate_up_dtype, config.hidden_size);
        let down_row_stride = row_stride(down_dtype, config.moe_intermediate_size);
        let gate_up_expert_bytes = checked_bytes(gate_up_rows, gate_up_row_stride)?;
        let down_expert_bytes = checked_bytes(config.hidden_size, down_row_stride)?;
        let expected_gate_up_bytes = checked_bytes(config.num_experts, gate_up_expert_bytes)?;
        let expected_down_bytes = checked_bytes(config.num_experts, down_expert_bytes)?;
        if gate_up_source.buf.size() < expected_gate_up_bytes {
            return Err(invalid(format!(
                "layer {} gate/up packed bytes {} < expected {}",
                layer_label,
                gate_up_source.buf.size(),
                expected_gate_up_bytes
            )));
        }
        if down_source.buf.size() < expected_down_bytes {
            return Err(invalid(format!(
                "layer {} down packed bytes {} < expected {}",
                layer_label,
                down_source.buf.size(),
                expected_down_bytes
            )));
        }

        let (gate_up_source_name, down_source_name) = match &moe.expert_source_layout {
            ExpertSourceLayout::PackedFused { gate_up, down, .. } => {
                (gate_up.as_str(), down.as_str())
            }
            layout => {
                return Err(invalid(format!(
                    "layer {} uses unsupported expert source layout {layout:?}",
                    layer_label
                )));
            }
        };
        let mut experts = Vec::with_capacity(config.num_experts);
        let mut metadata = Vec::with_capacity(config.num_experts);
        for expert in 0..config.num_experts {
            let gate_up_offset = expert * gate_up_expert_bytes;
            let down_offset = expert * down_expert_bytes;
            let gate_up = ProjectionView::from_packed(
                gate_up_source,
                gate_up_offset,
                gate_up_expert_bytes,
                gate_up_dtype,
                gate_up_rows,
                config.hidden_size,
            );
            let down = ProjectionView::from_packed(
                down_source,
                down_offset,
                down_expert_bytes,
                down_dtype,
                config.hidden_size,
                config.moe_intermediate_size,
            );
            experts.push(Qwen4ExpertView { gate_up, down });

            let gate_up_resource = ExpertResource::new(
                gate_up_source_name,
                "qwen4-resident-mqv2-v1",
                vec![gate_up_rows, config.hidden_size],
                gate_up_dtype,
                gate_up_expert_bytes,
                gate_up_row_stride,
                16,
                dtype_rotation_plan(gate_up_dtype),
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert resource: {error:?}"))
            })?;
            let down_resource = ExpertResource::new(
                down_source_name,
                "qwen4-resident-mqv2-v1",
                vec![config.hidden_size, config.moe_intermediate_size],
                down_dtype,
                down_expert_bytes,
                down_row_stride,
                16,
                dtype_rotation_plan(down_dtype),
            )
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert resource: {error:?}"))
            })?;
            let resources =
                ExpertResources::fused(gate_up_resource, down_resource).map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("expert resources: {error:?}"))
                })?;
            metadata.push(
                ExpertMetadata::new(expert, 0, expert, resources).map_err(|error| {
                    Qwen4GpuForwardError::Dispatch(format!("expert metadata: {error:?}"))
                })?,
            );
        }
        let table = ExpertTable::new(metadata)
            .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("expert table: {error:?}")))?;
        let mut cache = table
            .prepare_binding(0, 1, gpu.device_id)
            .map_err(|error| {
                Qwen4GpuForwardError::Dispatch(format!("expert binding: {error:?}"))
            })?;

        let gate_ptrs = experts
            .iter()
            .map(|expert| expert.gate_up.tensor.buf.as_ptr() as usize)
            .flat_map(usize::to_ne_bytes)
            .collect::<Vec<_>>();
        let down_ptrs = experts
            .iter()
            .map(|expert| expert.down.tensor.buf.as_ptr() as usize)
            .flat_map(usize::to_ne_bytes)
            .collect::<Vec<_>>();
        let expert_gate_up_ptrs = match gpu.alloc_tensor(&[2 * config.num_experts], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => return Err(error.into()),
        };
        if let Err(error) = gpu.memcpy_htod_auto(&expert_gate_up_ptrs.buf, &gate_ptrs) {
            let _ = gpu.free_tensor(expert_gate_up_ptrs);
            return Err(error.into());
        }
        let expert_down_ptrs = match gpu.alloc_tensor(&[2 * config.num_experts], DType::F32) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(expert_gate_up_ptrs);
                return Err(error.into());
            }
        };
        if let Err(error) = gpu.memcpy_htod_auto(&expert_down_ptrs.buf, &down_ptrs) {
            let _ = gpu.free_tensor(expert_gate_up_ptrs);
            let _ = gpu.free_tensor(expert_down_ptrs);
            return Err(error.into());
        }
        let expert_views = ExpertViewSet { experts: &experts };
        if let Err(error) = cache.bind_live(
            &table,
            &expert_views,
            &expert_gate_up_ptrs,
            &expert_down_ptrs,
            None,
            None,
        ) {
            let _ = gpu.free_tensor(expert_gate_up_ptrs);
            let _ = gpu.free_tensor(expert_down_ptrs);
            return Err(Qwen4GpuForwardError::Dispatch(format!(
                "expert live binding: {error:?}"
            )));
        }
        Ok(Self {
            table,
            cache,
            router,
            shared_scalar,
            shared_gate,
            shared_up,
            shared_down,
            experts,
            expert_gate_up_ptrs,
            expert_down_ptrs,
        })
    }

    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let mut first = None;
        for tensor in [self.expert_gate_up_ptrs, self.expert_down_ptrs] {
            if let Err(error) = gpu.free_tensor(tensor) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
        first
    }
}

/// Scratch views required by one sealed Qwen4 MoE invocation.  The caller owns
/// these reusable buffers; the sealed expert table and live pointer resources
/// remain in [`Qwen4MoeLayerRuntime`].
pub(crate) struct Qwen4MoeScratch<'a> {
    pub(crate) router_logits: &'a GpuTensor,
    pub(crate) scalar_buf: &'a GpuTensor,
    pub(crate) x_rot_local: &'a GpuTensor,
    pub(crate) gate_up_buf: &'a GpuTensor,
    pub(crate) gate_buf: &'a GpuTensor,
    pub(crate) up_buf: &'a GpuTensor,
    pub(crate) ffn_hidden: &'a GpuTensor,
    pub(crate) ffn_out: &'a GpuTensor,
    pub(crate) gate_batch: &'a GpuTensor,
    pub(crate) up_batch: &'a GpuTensor,
    pub(crate) rot_batch: &'a GpuTensor,
    pub(crate) topk_indices: &'a GpuTensor,
    pub(crate) topk_weights: &'a GpuTensor,
    pub(crate) down_expanded: &'a GpuTensor,
}

pub(crate) fn execute_moe(
    gpu: &mut Gpu,
    config: &Qwen4Config,
    layer_index: usize,
    runtime: &Qwen4MoeLayerRuntime,
    input: &GpuTensor,
    output: &GpuTensor,
    scratch: Qwen4MoeScratch<'_>,
) -> Result<(), Qwen4GpuForwardError> {
    let ctx = DispatchCtx::new(gpu);
    gpu.hip.memset(&output.buf, 0, output.buf.size())?;
    let dtypes = MoeDtypes {
        router: runtime.router.dtype,
        shared_gate: runtime.shared_scalar.dtype,
        shared_expert_gate: runtime.shared_gate.dtype,
        shared_expert_up: runtime.shared_up.dtype,
        shared_expert_down: runtime.shared_down.dtype,
        experts_all_gate_up_mq4: runtime
            .experts
            .iter()
            .all(|expert| expert.gate_up.dtype == DType::MQ4G256V2),
        routed_gate_up: runtime
            .experts
            .first()
            .map(|expert| expert.gate_up.dtype)
            .ok_or_else(|| invalid("MoE expert table is empty"))?,
        routed_down: runtime
            .experts
            .first()
            .map(|expert| expert.down.dtype)
            .ok_or_else(|| invalid("MoE expert table is empty"))?,
        routed_has_mixed_experts: false,
        has_paro_shared: false,
        per_expert_gate_up: None,
        per_expert_down: None,
    };
    let params = MoeParams {
        dtypes,
        batch_size: 1,
        hidden: config.hidden_size,
        mi: config.moe_intermediate_size,
        smi: config.shared_expert_intermediate_size,
        k: config.num_experts_per_tok,
        n_exp: config.num_experts,
        norm_topk_prob: config.norm_topk_prob,
        x_rot_prerotated: false,
        defer_routed_combine: false,
        layer_idx: layer_index as u16,
        x_norm: input,
        x_residual: output,
        routed_out: None,
        skip_shared: false,
        router: runtime.router.dispatch_ref(),
        shared_expert_gate: runtime.shared_scalar.dispatch_ref(),
        shared_gate_w: runtime.shared_gate.dispatch_ref(),
        shared_up_w: runtime.shared_up.dispatch_ref(),
        shared_down_w: runtime.shared_down.dispatch_ref(),
        expert_gate_up_ptrs: &runtime.expert_gate_up_ptrs,
        expert_down_ptrs: &runtime.expert_down_ptrs,
        expert_down_awq_ptrs: None,
        expert_dtype_tags: None,
        routed_gate_up_k: config.hidden_size,
        routed_down_m: config.hidden_size,
        routed_down_k: config.moe_intermediate_size,
        routed_experts: runtime,
        routed_gate_up_paro: None,
        routed_down_paro: None,
        router_logits: scratch.router_logits,
        scalar_buf: scratch.scalar_buf,
        x_rot_local: scratch.x_rot_local,
        gate_up_buf: scratch.gate_up_buf,
        gate_buf: scratch.gate_buf,
        up_buf: scratch.up_buf,
        ffn_hidden: scratch.ffn_hidden,
        ffn_out: scratch.ffn_out,
        gate_batch: scratch.gate_batch,
        up_batch: scratch.up_batch,
        rot_batch: scratch.rot_batch,
        topk_indices: scratch.topk_indices,
        topk_weights: scratch.topk_weights,
        down_expanded: scratch.down_expanded,
    };
    let bound = BoundMoeExperts::from_cache(&runtime.table, &runtime.cache)
        .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("bound MoE experts: {error:?}")))?;
    let sealed = seal_decode(bound, &ctx, params)
        .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("seal Qwen4 MoE: {error:?}")))?;
    execute_steps(gpu, &ctx, &[Step::Moe(sealed)])
        .map_err(|error| Qwen4GpuForwardError::Dispatch(format!("execute Qwen4 MoE: {error:?}")))?;
    Ok(())
}

/// Device buffers shared by one forward object.  All allocations happen in
/// `new`; token/chunk execution only creates borrowed subviews.
pub struct Qwen4GpuForwardScratch {
    pub max_chunk: usize,
    pub token_ids: GpuTensor,
    pub embedding_rot: GpuTensor,
    pub embeddings: GpuTensor,
    pub streams: GpuTensor,
    pub hc_normalized: GpuTensor,
    pub hc_low: GpuTensor,
    pub hc_up: GpuTensor,
    pub hc_mixed: GpuTensor,
    pub hc_gates: GpuTensor,
    pub rotation: GpuTensor,
    pub projection: GpuTensor,
    pub projection2: GpuTensor,
    pub gdn_a: GpuTensor,
    pub gdn_b: GpuTensor,
    pub gdn_gate: GpuTensor,
    pub gdn_beta: GpuTensor,
    pub gdn_recurrent_output: GpuTensor,
    pub gdn_bf16: GpuTensor,
    pub gdn_z: GpuTensor,
    pub gdn_output: GpuTensor,
    pub qsa_index: GpuTensor,
    pub qsa_qgate: GpuTensor,
    pub qsa_k: GpuTensor,
    pub qsa_v: GpuTensor,
    pub qsa_output: GpuTensor,
    pub ple_staged: GpuTensor,
    pub ple_rows: GpuTensor,
    pub ple_key: GpuTensor,
    pub ple_value: GpuTensor,
    pub ple_query: GpuTensor,
    pub ple_gated: GpuTensor,
    pub ple_normed: GpuTensor,
    pub ple_output: GpuTensor,
    pub router_logits: GpuTensor,
    pub moe_x_rot: GpuTensor,
    pub moe_gate_up: GpuTensor,
    pub moe_gate: GpuTensor,
    pub moe_up: GpuTensor,
    pub moe_hidden: GpuTensor,
    pub moe_output: GpuTensor,
    pub moe_gate_batch: GpuTensor,
    pub moe_up_batch: GpuTensor,
    pub moe_rot_batch: GpuTensor,
    pub moe_topk_indices: GpuTensor,
    pub moe_topk_weights: GpuTensor,
    pub moe_down_expanded: GpuTensor,
    pub moe_scalar: GpuTensor,
    pub logits: GpuTensor,
    pub host_token_bytes: Vec<u8>,
    pub host_ple_bytes: Vec<u8>,
}

impl Qwen4GpuForwardScratch {
    pub fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_chunk: usize,
    ) -> Result<Self, Qwen4GpuForwardError> {
        if max_chunk == 0 {
            return Err(invalid("max_chunk is zero"));
        }
        let hidden = config.hidden_size;
        let wide = config.hc_count * hidden;
        let q_width = config.num_attention_heads * config.head_dim;
        let qsa_qgate = 2 * q_width;
        let qsa_index =
            (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
        let gdn_qk = config.linear_num_key_heads * config.linear_key_head_dim;
        let gdn_value = config.linear_num_value_heads * config.linear_value_head_dim;
        let gdn_qkv = 2 * gdn_qk + gdn_value;
        let max_projection = wide.max(qsa_qgate).max(gdn_qkv).max(hidden);
        let max_rotation = wide.max(hidden).max(config.moe_intermediate_size);
        let ple_channels = config.ple_embed_dim * config.hc_count;
        let max_experts = config.num_experts_per_tok;
        let max_logits = max_chunk
            .checked_mul(config.vocab_size)
            .ok_or_else(|| invalid("logit scratch overflow"))?;
        let max_ple_bytes = max_chunk
            .checked_mul(PLE_ROWS_PER_TOKEN)
            .and_then(|bytes| bytes.checked_mul(PLE_ROW_BYTES))
            .ok_or_else(|| invalid("PLE staging size overflow"))?;
        let mut allocated = Vec::new();
        let mut alloc = |shape: &[usize], dtype: DType| -> Result<(), Qwen4GpuForwardError> {
            let tensor = gpu.zeros(shape, dtype)?;
            allocated.push(tensor);
            Ok(())
        };
        let result = (|| {
            alloc(&[max_chunk * std::mem::size_of::<i32>()], DType::Raw)?;
            alloc(&[max_chunk * hidden], DType::F32)?;
            alloc(&[max_chunk * hidden], DType::F32)?;
            alloc(&[wide], DType::F32)?;
            alloc(&[wide], DType::F32)?;
            alloc(&[config.hc_lowrank], DType::F32)?;
            alloc(&[wide], DType::F32)?;
            alloc(&[hidden], DType::F32)?;
            alloc(&[config.hc_count], DType::F32)?;
            alloc(&[max_rotation], DType::F32)?;
            alloc(&[max_projection], DType::F32)?;
            alloc(&[max_projection], DType::F32)?;
            alloc(&[config.linear_num_value_heads], DType::F32)?;
            alloc(&[config.linear_num_value_heads], DType::F32)?;
            alloc(&[config.linear_num_value_heads], DType::F32)?;
            alloc(&[config.linear_num_value_heads], DType::F32)?;
            alloc(&[gdn_value], DType::F32)?;
            alloc(&[gdn_value], DType::BF16)?;
            alloc(&[gdn_value], DType::F32)?;
            alloc(&[gdn_value], DType::F32)?;
            alloc(&[qsa_index], DType::F32)?;
            alloc(&[qsa_qgate], DType::F32)?;
            alloc(&[config.num_key_value_heads * config.head_dim], DType::F32)?;
            alloc(&[config.num_key_value_heads * config.head_dim], DType::F32)?;
            alloc(&[q_width], DType::F32)?;
            alloc(
                &[max_chunk * PLE_ROWS_PER_TOKEN * (PLE_ROW_BYTES / 2)],
                DType::BF16,
            )?;
            alloc(&[max_chunk * hidden], DType::F32)?;
            alloc(&[ple_channels], DType::F32)?;
            alloc(&[hidden], DType::F32)?;
            alloc(&[ple_channels], DType::F32)?;
            alloc(&[ple_channels], DType::F32)?;
            alloc(&[ple_channels], DType::F32)?;
            alloc(&[ple_channels], DType::F32)?;
            alloc(&[config.num_experts], DType::F32)?;
            alloc(&[hidden], DType::F32)?;
            alloc(&[2 * config.moe_intermediate_size], DType::F32)?;
            alloc(&[config.moe_intermediate_size], DType::F32)?;
            alloc(&[config.moe_intermediate_size], DType::F32)?;
            alloc(&[config.shared_expert_intermediate_size], DType::F32)?;
            alloc(&[hidden], DType::F32)?;
            alloc(&[max_experts * config.moe_intermediate_size], DType::F32)?;
            alloc(&[max_experts * config.moe_intermediate_size], DType::F32)?;
            alloc(&[max_experts * config.moe_intermediate_size], DType::F32)?;
            alloc(&[max_experts], DType::F32)?;
            alloc(&[max_experts], DType::F32)?;
            alloc(&[max_experts * hidden + hidden.div_ceil(4)], DType::F32)?;
            alloc(&[config.shared_expert_intermediate_size.max(1)], DType::F32)?;
            alloc(&[max_logits], DType::F32)?;
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = result {
            for tensor in allocated {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(error);
        }
        let mut next = || allocated.remove(0);
        Ok(Self {
            max_chunk,
            token_ids: next(),
            embedding_rot: next(),
            embeddings: next(),
            streams: next(),
            hc_normalized: next(),
            hc_low: next(),
            hc_up: next(),
            hc_mixed: next(),
            hc_gates: next(),
            rotation: next(),
            projection: next(),
            projection2: next(),
            gdn_a: next(),
            gdn_b: next(),
            gdn_gate: next(),
            gdn_beta: next(),
            gdn_recurrent_output: next(),
            gdn_bf16: next(),
            gdn_z: next(),
            gdn_output: next(),
            qsa_index: next(),
            qsa_qgate: next(),
            qsa_k: next(),
            qsa_v: next(),
            qsa_output: next(),
            ple_staged: next(),
            ple_rows: next(),
            ple_key: next(),
            ple_value: next(),
            ple_query: next(),
            ple_gated: next(),
            ple_normed: next(),
            ple_output: next(),
            router_logits: next(),
            moe_x_rot: next(),
            moe_gate_up: next(),
            moe_gate: next(),
            moe_up: next(),
            moe_hidden: next(),
            moe_output: next(),
            moe_gate_batch: next(),
            moe_up_batch: next(),
            moe_rot_batch: next(),
            moe_topk_indices: next(),
            moe_topk_weights: next(),
            moe_down_expanded: next(),
            moe_scalar: next(),
            logits: next(),
            host_token_bytes: vec![0; max_chunk * std::mem::size_of::<i32>()],
            host_ple_bytes: vec![0; max_ple_bytes],
        })
    }

    fn free_gpu(self, gpu: &mut Gpu) -> Option<hip_bridge::HipError> {
        let tensors = [
            self.token_ids,
            self.embedding_rot,
            self.embeddings,
            self.streams,
            self.hc_normalized,
            self.hc_low,
            self.hc_up,
            self.hc_mixed,
            self.hc_gates,
            self.rotation,
            self.projection,
            self.projection2,
            self.gdn_a,
            self.gdn_b,
            self.gdn_gate,
            self.gdn_beta,
            self.gdn_recurrent_output,
            self.gdn_bf16,
            self.gdn_z,
            self.gdn_output,
            self.qsa_index,
            self.qsa_qgate,
            self.qsa_k,
            self.qsa_v,
            self.qsa_output,
            self.ple_staged,
            self.ple_rows,
            self.ple_key,
            self.ple_value,
            self.ple_query,
            self.ple_gated,
            self.ple_normed,
            self.ple_output,
            self.router_logits,
            self.moe_x_rot,
            self.moe_gate_up,
            self.moe_gate,
            self.moe_up,
            self.moe_hidden,
            self.moe_output,
            self.moe_gate_batch,
            self.moe_up_batch,
            self.moe_rot_batch,
            self.moe_topk_indices,
            self.moe_topk_weights,
            self.moe_down_expanded,
            self.moe_scalar,
            self.logits,
        ];
        let mut first = None;
        for tensor in tensors {
            if let Err(error) = gpu.free_tensor(tensor) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
        first
    }
}

/// Owns request-local PLE work until the forward attempt either commits or
/// aborts.  Immutable page-cache entries belong to [`PleRows`] and survive
/// `reset_epoch`; this guard only owns the exact ticket and completed lease for
/// the current epoch.
struct PleEpochGuard<'a> {
    rows: &'a PleRows,
    epoch: u64,
    ticket: Option<PlePrefetch>,
    lease: Option<PleRowLease>,
    armed: bool,
}

impl<'a> PleEpochGuard<'a> {
    fn new(rows: &'a PleRows, epoch: u64) -> Self {
        Self {
            rows,
            epoch,
            ticket: None,
            lease: None,
            armed: true,
        }
    }

    fn install_ticket(&mut self, ticket: PlePrefetch) {
        debug_assert!(self.ticket.is_none());
        self.ticket = Some(ticket);
    }

    fn install_lease(&mut self, lease: PleRowLease) {
        // `consume_at_layer1` marks the ticket consumed.  Drop that handle
        // before retaining the lease so the only live PLE owner is explicit.
        self.ticket.take();
        self.lease = Some(lease);
    }
    fn abort(&mut self) -> Result<u64, String> {
        if !self.armed {
            return Ok(self.rows.current_epoch());
        }
        // Disarm first: if cleanup itself reports an error, Drop must not
        // issue a second reset against a later epoch.
        self.armed = false;
        let mut errors = Vec::new();
        if let Some(ticket) = self.ticket.take() {
            if let Err(error) = self.rows.cancel(&ticket) {
                // consume_at_layer1 marks a ticket consumed before waiting;
                // source-read and cancellation errors therefore legitimately
                // report AlreadyConsumed during abort.
                if !matches!(
                    error,
                    PleRowsError::AlreadyConsumed(_)
                        | PleRowsError::Canceled
                        | PleRowsError::UnknownTicket(_)
                ) {
                    errors.push(format!("cancel epoch {} ticket: {error}", self.epoch));
                }
            }
            drop(ticket);
        }
        // A lease holds one of the bounded staging buffers.  It must be
        // returned before reset_epoch waits for readers/leases to drain.
        drop(self.lease.take());
        let next_epoch = match self.rows.reset_epoch(PLE_CLEANUP_TIMEOUT) {
            Ok(next) => Some(next),
            Err(error) => {
                errors.push(format!("drain epoch {}: {error}", self.epoch));
                None
            }
        };
        if errors.is_empty() {
            Ok(next_epoch.expect("successful PLE reset returns its next epoch"))
        } else {
            Err(errors.join("; "))
        }
    }

    fn complete(&mut self) {
        self.armed = false;
        drop(self.ticket.take());
        drop(self.lease.take());
    }
}

impl Drop for PleEpochGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            let _ = self.abort();
        }
    }
}

/// Reusable production forward owner.  Construct it once per assembled bundle
/// and use `forward_token`/`forward_chunk` repeatedly; both methods share the
/// exact same chunk implementation.
pub struct Qwen4GpuForward {
    pub scratch: Qwen4GpuForwardScratch,
    moe: Vec<Qwen4MoeLayerRuntime>,
    ple_epoch: u64,
}

impl Qwen4GpuForward {
    pub fn new(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        max_chunk: usize,
    ) -> Result<Self, Qwen4GpuForwardError> {
        if !gpu.arch_caps.is_gfx1151() {
            return Err(invalid("Qwen4 ordinary-HIP path requires gfx1151"));
        }
        let scratch = Qwen4GpuForwardScratch::new(gpu, &bundle.config, max_chunk)?;
        let mut moe = Vec::with_capacity(bundle.config.num_hidden_layers);
        let result = (|| {
            for layer in &bundle.weights.layer_refs {
                moe.push(Qwen4MoeLayerRuntime::new(
                    gpu,
                    &bundle.weights,
                    layer,
                    &bundle.config,
                )?);
            }
            Ok::<(), Qwen4GpuForwardError>(())
        })();
        if let Err(error) = result {
            for layer in moe {
                let _ = layer.free_gpu(gpu);
            }
            let _ = scratch.free_gpu(gpu);
            return Err(error);
        }
        Ok(Self {
            scratch,
            moe,
            ple_epoch: 0,
        })
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), hip_bridge::HipError> {
        let Qwen4GpuForward { scratch, moe, .. } = self;
        let mut first = scratch.free_gpu(gpu);
        for layer in moe {
            if let Some(error) = layer.free_gpu(gpu) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        }
        first.map_or(Ok(()), Err)
    }

    pub fn forward_token(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        token: u32,
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.forward_chunk_inner(
            bundle,
            gpu,
            std::slice::from_ref(&token),
            logits,
            top1,
            None,
        )
    }

    pub(crate) fn forward_token_with_wide_hidden(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        token: u32,
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        wide_hidden: &GpuTensor,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.forward_chunk_inner(
            bundle,
            gpu,
            std::slice::from_ref(&token),
            logits,
            top1,
            Some(wide_hidden),
        )
    }

    pub fn forward_chunk(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.forward_chunk_inner(bundle, gpu, tokens, logits, top1, None)
    }

    pub(crate) fn forward_chunk_with_wide_hidden(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        wide_hidden: &GpuTensor,
    ) -> Result<(), Qwen4GpuForwardError> {
        self.forward_chunk_inner(bundle, gpu, tokens, logits, top1, Some(wide_hidden))
    }

    fn forward_chunk_inner(
        &mut self,
        bundle: &mut Qwen4Bundle,
        gpu: &mut Gpu,
        tokens: &[u32],
        logits: &GpuTensor,
        top1: Option<&GpuTensor>,
        wide_hidden_capture: Option<&GpuTensor>,
    ) -> Result<(), Qwen4GpuForwardError> {
        let n = tokens.len();
        if n == 0 || n > self.scratch.max_chunk {
            return Err(invalid(format!(
                "chunk length {n} exceeds configured capacity {}",
                self.scratch.max_chunk
            )));
        }
        let expected_logits = n
            .checked_mul(bundle.config.vocab_size)
            .ok_or_else(|| invalid("logit shape overflow"))?;
        if logits.dtype != DType::F32 || logits.numel() != expected_logits {
            return Err(invalid(format!(
                "logits must be F32 with {expected_logits} elements"
            )));
        }
        if let Some(top1) = top1 {
            if top1.dtype != DType::Raw || top1.buf.size() < n * std::mem::size_of::<i32>() {
                return Err(invalid("top1 output must be Raw with one i32 per token"));
            }
        }
        if let Some(capture) = wide_hidden_capture.as_ref() {
            let expected = n
                .checked_mul(bundle.config.hc_count)
                .and_then(|v| v.checked_mul(bundle.config.hidden_size))
                .ok_or_else(|| invalid("wide hidden capture shape overflow"))?;
            if capture.dtype != DType::F32 || capture.numel() < expected {
                return Err(invalid(format!(
                    "wide hidden capture must be F32 with at least {expected} elements"
                )));
            }
        }
        let config = bundle.config.clone();
        for (index, token) in tokens.iter().copied().enumerate() {
            let bytes = &mut self.scratch.host_token_bytes[index * 4..index * 4 + 4];
            bytes.copy_from_slice(&(token as i32).to_ne_bytes());
        }
        gpu.memcpy_htod_auto(
            &self.scratch.token_ids.buf,
            &self.scratch.host_token_bytes[..n * std::mem::size_of::<i32>()],
        )?;
        let embedding = bundle.weights.resident(&bundle.weights.root.embedding)?;
        let embedding_rot = view(&self.scratch.embedding_rot, 0, n * config.hidden_size);
        let embeddings = view(&self.scratch.embeddings, 0, n * config.hidden_size);
        let ids = view(&self.scratch.token_ids, 0, n * std::mem::size_of::<i32>());
        gpu.embedding_lookup_mq4v2_batched(
            embedding,
            &embedding_rot,
            &embeddings,
            &ids,
            n,
            config.hidden_size,
        )?;

        self.ple_epoch = self.ple_epoch.wrapping_add(1).max(1);
        bundle
            .begin_ple_epoch(self.ple_epoch)
            .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
        let ple_rows_resource = &bundle.ple_rows;
        let mut ple = PleEpochGuard::new(ple_rows_resource, self.ple_epoch);
        let next_history = bundle.state.ple_history;
        let next_position = bundle.state.position;
        let attempt = (|| -> Result<(), Qwen4GpuForwardError> {
            let ticket = ple
                .rows
                .prefetch_before_layer0(self.ple_epoch, next_history, tokens)
                .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
            ple.install_ticket(ticket);
            let mut next_history = next_history;
            let mut next_position = next_position;
            let ple_rows = view(&self.scratch.ple_rows, 0, n * config.hidden_size);
            let staged = view(
                &self.scratch.ple_staged,
                0,
                n * PLE_ROWS_PER_TOKEN * (PLE_ROW_BYTES / 2),
            );
            for (token_index, _) in tokens.iter().enumerate() {
                // These are indices into the layer-type-specific state vectors,
                // not global chunk counters.  Every token starts at its first
                // GDN/QSA state slot; the device state itself carries time.
                let mut gdn_slot = 0usize;
                let mut qsa_slot = 0usize;
                let embedding_row = f32_view(
                    &embeddings,
                    token_index * config.hidden_size,
                    config.hidden_size,
                );
                let streams = &self.scratch.streams;
                for branch in 0..config.hc_count {
                    gpu.memcpy_dtod_at_auto(
                        &streams.buf,
                        branch * config.hidden_size * 4,
                        &embedding_row.buf,
                        0,
                        config.hidden_size * 4,
                    )?;
                }
                for layer_index in 0..config.num_hidden_layers {
                    if layer_index
                        == config
                            .ple_layer_ids
                            .first()
                            .and_then(|id| id.checked_sub(1))
                            .unwrap_or(usize::MAX)
                    {
                        if ple.lease.is_none() {
                            let ticket = ple.ticket.as_ref().ok_or_else(|| {
                                invalid("PLE layer reached without a prefetch ticket")
                            })?;
                            let lease = ple
                                .rows
                                .consume_at_layer1(ticket)
                                .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
                            ple.install_lease(lease);
                            let lease = ple.lease.as_ref().ok_or_else(|| {
                                invalid("PLE lease disappeared after consumption")
                            })?;
                            let upload_len = lease
                                .as_bytes()
                                .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?
                                .len();
                            lease
                                .stage_into(&mut self.scratch.host_ple_bytes[..upload_len])
                                .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
                            gpu.memcpy_htod_auto(
                                &staged.buf,
                                &self.scratch.host_ple_bytes[..upload_len],
                            )?;
                            lease
                                .validate_after_upload()
                                .map_err(|error| Qwen4GpuForwardError::Ple(error.to_string()))?;
                            gpu.qwen4_ple_gather_convert_bf16(&staged, &ple_rows, n)?;
                        }
                        self.apply_ple(
                            gpu,
                            &config,
                            &bundle.weights,
                            &mut bundle.state.ple_conv,
                            &ple_rows,
                            token_index,
                        )?;
                    }
                    let layer = &bundle.weights.layer_refs[layer_index];
                    self.hc_read(
                        gpu,
                        &config,
                        &bundle.weights,
                        &layer.attn_hyper,
                        streams,
                        &self.scratch.hc_normalized,
                        &self.scratch.hc_low,
                        &self.scratch.hc_up,
                        &self.scratch.hc_mixed,
                    )?;
                    let attn_input = &self.scratch.hc_mixed;
                    let attn_output = match layer.kind {
                        LayerType::LinearAttention => {
                            let state = bundle
                                .state
                                .gdn_mut(gdn_slot)
                                .ok_or_else(|| invalid("GDN state slot missing"))?;
                            let output = &self.scratch.gdn_output;
                            self.apply_gdn(
                                gpu,
                                &config,
                                &bundle.weights,
                                layer,
                                state,
                                next_position,
                                attn_input,
                                output,
                            )?;
                            gdn_slot += 1;
                            output
                        }
                        LayerType::FullAttention => {
                            let state = bundle
                                .state
                                .qsa_mut(qsa_slot)
                                .ok_or_else(|| invalid("QSA state slot missing"))?;
                            let output = &self.scratch.qsa_output;
                            self.apply_qsa(
                                gpu,
                                &config,
                                &bundle.weights,
                                layer,
                                state,
                                attn_input,
                                output,
                            )?;
                            qsa_slot += 1;
                            output
                        }
                    };
                    self.hc_write(
                        gpu,
                        &config,
                        &bundle.weights,
                        &layer.attn_hyper,
                        streams,
                        &self.scratch.hc_normalized,
                        attn_output,
                        streams,
                    )?;
                    self.hc_read(
                        gpu,
                        &config,
                        &bundle.weights,
                        &layer.mlp_hyper,
                        streams,
                        &self.scratch.hc_normalized,
                        &self.scratch.hc_low,
                        &self.scratch.hc_up,
                        &self.scratch.hc_mixed,
                    )?;
                    self.apply_moe(
                        gpu,
                        &config,
                        layer_index,
                        &self.moe[layer_index],
                        &self.scratch.hc_mixed,
                        &self.scratch.moe_output,
                    )?;
                    self.hc_write(
                        gpu,
                        &config,
                        &bundle.weights,
                        &layer.mlp_hyper,
                        streams,
                        &self.scratch.hc_normalized,
                        &self.scratch.moe_output,
                        streams,
                    )?;
                }
                // Capture the post-trunk 4H stream before final HC collapse.
                // Keep the source borrow immutable and copy into the caller's
                // model-owned destination; no drafter buffer aliases scratch.
                if let Some(capture) = wide_hidden_capture.as_ref() {
                    let width = config.hc_count * config.hidden_size;
                    let trunk_wide: &GpuTensor = &self.scratch.streams;
                    let destination = f32_view(capture, token_index * width, width);
                    gpu.copy_d2d(trunk_wide, &destination, trunk_wide.byte_size())?;
                }
                let final_hyper = &bundle.weights.root.final_hyper;
                self.hc_read(
                    gpu,
                    &config,
                    &bundle.weights,
                    final_hyper,
                    &self.scratch.streams,
                    &self.scratch.hc_normalized,
                    &self.scratch.hc_low,
                    &self.scratch.hc_up,
                    &self.scratch.hc_mixed,
                )?;
                let final_hidden = &self.scratch.hc_mixed;
                let lm_head = bundle.weights.resident(&bundle.weights.root.lm_head)?;
                let output_row =
                    f32_view(logits, token_index * config.vocab_size, config.vocab_size);
                self.gemv(
                    gpu,
                    lm_head,
                    final_hidden,
                    &self.scratch.rotation,
                    &output_row,
                    config.vocab_size,
                    config.hidden_size,
                )?;
                // Host-visible request state is staged only after this token's
                // complete device path succeeds.  The next token sees the
                // staged position while a failed token publishes nothing.
                next_history.push(tokens[token_index]);
                next_position = next_position.saturating_add(1);
            }
            if let Some(top1) = top1 {
                qwen4_argmax(
                    gpu,
                    &Qwen4Argmax {
                        logits,
                        indices: top1,
                        rows: n,
                        vocab: config.vocab_size,
                    },
                )?;
            }
            bundle.state.ple_history = next_history;
            bundle.state.position = next_position;
            Ok(())
        })();
        match attempt {
            Ok(()) => {
                ple.complete();
                Ok(())
            }
            Err(error) => match ple.abort() {
                Ok(next_epoch) => {
                    self.ple_epoch = next_epoch;
                    Err(error)
                }
                Err(cleanup) => {
                    self.ple_epoch = ple.rows.current_epoch();
                    Err(Qwen4GpuForwardError::Ple(format!(
                        "forward attempt failed: {error}; PLE cleanup failed: {cleanup}"
                    )))
                }
            },
        }
    }
    fn gemv(
        &self,
        gpu: &mut Gpu,
        weight: &GpuTensor,
        input: &GpuTensor,
        rotation: &GpuTensor,
        output: &GpuTensor,
        m: usize,
        k: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        dispatch_gemv(gpu, weight, input, rotation, output, m, k)?;
        Ok(())
    }

    fn hc_read(
        &self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
        weights: &Qwen4Weights,
        hyper: &HyperConnectionWeights,
        input: &GpuTensor,
        normalized: &GpuTensor,
        low: &GpuTensor,
        up: &GpuTensor,
        mixed: &GpuTensor,
    ) -> Result<(), Qwen4GpuForwardError> {
        let norm = weights.resident(&hyper.hc_norm)?;
        let down = weights.resident(&hyper.input_mix_down)?;
        let up_weight = weights.resident(&hyper.input_mix_up)?;
        qwen4_hc_norm(
            gpu,
            &Qwen4HcNorm {
                input,
                norm_weight: norm,
                normalized,
                branches: config.hc_count,
                hidden: config.hidden_size,
            },
        )?;
        self.gemv(
            gpu,
            down,
            normalized,
            &self.scratch.rotation,
            low,
            config.hc_lowrank,
            config.hc_count * config.hidden_size,
        )?;
        qwen4_scale(
            gpu,
            &Qwen4Scale {
                values: low,
                scale: 1.0 / config.hc_count as f32,
            },
        )?;
        gpu.silu_f32(low, low)?;
        self.gemv(
            gpu,
            up_weight,
            low,
            &self.scratch.rotation,
            up,
            config.hc_count * config.hidden_size,
            config.hc_lowrank,
        )?;
        qwen4_hc_read(
            gpu,
            &Qwen4HcRead {
                input,
                norm_weight: norm,
                low,
                up,
                normalized,
                mixed,
                branches: config.hc_count,
                hidden: config.hidden_size,
                rank: config.hc_lowrank,
            },
        )?;
        Ok(())
    }

    fn hc_write(
        &self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
        weights: &Qwen4Weights,
        hyper: &HyperConnectionWeights,
        input: &GpuTensor,
        normalized: &GpuTensor,
        mixed: &GpuTensor,
        output: &GpuTensor,
    ) -> Result<(), Qwen4GpuForwardError> {
        let norm = weights.resident(&hyper.hc_norm)?;
        let inject = weights.resident(&hyper.block_inject)?;
        qwen4_hc_norm(
            gpu,
            &Qwen4HcNorm {
                input,
                norm_weight: norm,
                normalized,
                branches: config.hc_count,
                hidden: config.hidden_size,
            },
        )?;
        let gates = &self.scratch.hc_gates;
        self.gemv(
            gpu,
            inject,
            normalized,
            &self.scratch.rotation,
            gates,
            config.hc_count,
            config.hc_count * config.hidden_size,
        )?;
        qwen4_hc_write(
            gpu,
            &Qwen4HcWrite {
                input,
                normalized,
                mixed,
                gates,
                output,
                branches: config.hc_count,
                hidden: config.hidden_size,
            },
        )?;
        Ok(())
    }

    fn apply_gdn(
        &self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
        weights: &Qwen4Weights,
        layer: &Qwen4LayerWeights,
        state: &mut GdnGpuState,
        position: usize,
        input: &GpuTensor,
        output: &GpuTensor,
    ) -> Result<(), Qwen4GpuForwardError> {
        let gdn = layer
            .gdn
            .as_ref()
            .ok_or_else(|| invalid("linear layer has no GDN weights"))?;
        let qk = config.linear_num_key_heads * config.linear_key_head_dim;
        let value = config.linear_num_value_heads * config.linear_value_head_dim;
        let qkv = 2 * qk + value;
        let qkv_weight = weights.resident(&gdn.qkv)?;
        let conv = weights.resident(&gdn.conv)?;
        self.gemv(
            gpu,
            qkv_weight,
            input,
            &self.scratch.rotation,
            &self.scratch.projection,
            qkv,
            config.hidden_size,
        )?;
        let history_rows = config.linear_conv_kernel_dim.saturating_sub(1);
        let cursor = if history_rows == 0 {
            0
        } else {
            position % history_rows
        };
        qwen4_gdn_conv(
            gpu,
            &Qwen4GdnConv {
                input: &self.scratch.projection,
                kernel: conv,
                history: &state.conv,
                output: &self.scratch.projection2,
                next_history: &state.conv,
                channels: qkv,
                history_rows,
                kernel_size: config.linear_conv_kernel_dim,
                cursor,
            },
        )?;
        let a_weight = weights.resident(&gdn.in_proj_a)?;
        let b_weight = weights.resident(&gdn.in_proj_b)?;
        self.gemv(
            gpu,
            a_weight,
            input,
            &self.scratch.rotation,
            &self.scratch.gdn_a,
            config.linear_num_value_heads,
            config.hidden_size,
        )?;
        self.gemv(
            gpu,
            b_weight,
            input,
            &self.scratch.rotation,
            &self.scratch.gdn_b,
            config.linear_num_value_heads,
            config.hidden_size,
        )?;
        let a_log = weights.resident(&gdn.a_log)?;
        let dt_bias = weights.resident(&gdn.dt_bias)?;
        qwen4_gdn_params(
            gpu,
            &Qwen4GdnParams {
                a: &self.scratch.gdn_a,
                b: &self.scratch.gdn_b,
                a_log,
                dt_bias,
                gate: &self.scratch.gdn_gate,
                beta: &self.scratch.gdn_beta,
            },
            config.linear_num_value_heads,
        )?;
        let q = view(&self.scratch.projection2, 0, qk);
        let k = view(&self.scratch.projection2, qk, qk);
        let v = view(&self.scratch.projection2, 2 * qk, value);
        qwen4_gdn_step(
            gpu,
            &Qwen4GdnStep {
                q: &q,
                k: &k,
                v: &v,
                gate: &self.scratch.gdn_gate,
                beta: &self.scratch.gdn_beta,
                state: &state.recurrent,
                output: &self.scratch.gdn_recurrent_output,
                key_heads: config.linear_num_key_heads,
                value_heads: config.linear_num_value_heads,
                key_dim: config.linear_key_head_dim,
                value_dim: config.linear_value_head_dim,
            },
        )?;
        qwen4_gdn_bf16_roundtrip(
            gpu,
            &Qwen4GdnBf16Roundtrip {
                input: &self.scratch.gdn_recurrent_output,
                scratch: &self.scratch.gdn_bf16,
                output: &self.scratch.gdn_recurrent_output,
                elements: value,
            },
        )?;
        let z_weight = weights.resident(&gdn.z)?;
        self.gemv(
            gpu,
            z_weight,
            input,
            &self.scratch.rotation,
            &self.scratch.gdn_z,
            value,
            config.hidden_size,
        )?;
        let norm = weights.resident(&gdn.norm)?;
        qwen4_gdn_gate(
            gpu,
            &Qwen4GdnGate {
                recurrent_output: &self.scratch.gdn_recurrent_output,
                z: &self.scratch.gdn_z,
                norm,
                output: &self.scratch.gdn_output,
                value_heads: config.linear_num_value_heads,
                value_dim: config.linear_value_head_dim,
            },
        )?;
        let out_weight = weights.resident(&gdn.output)?;
        self.gemv(
            gpu,
            out_weight,
            &self.scratch.gdn_output,
            &self.scratch.rotation,
            output,
            config.hidden_size,
            value,
        )?;
        Ok(())
    }

    fn apply_qsa(
        &self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
        weights: &Qwen4Weights,
        layer: &Qwen4LayerWeights,
        state: &mut QsaGpuState,
        input: &GpuTensor,
        output: &GpuTensor,
    ) -> Result<(), Qwen4GpuForwardError> {
        let qsa = layer
            .attention
            .as_ref()
            .ok_or_else(|| invalid("full layer has no QSA weights"))?;
        let index_dim = config.indexer_head_dim;
        let index_q_width = config.indexer_n_heads * index_dim;
        let index_width = (config.indexer_n_heads + config.indexer_kv_heads) * index_dim;
        let index_weight = weights.resident(&qsa.indexer_qk)?;
        self.gemv(
            gpu,
            index_weight,
            input,
            &self.scratch.rotation,
            &self.scratch.qsa_index,
            index_width,
            config.hidden_size,
        )?;
        let index_q = view(&self.scratch.qsa_index, 0, index_q_width);
        let index_k = view(
            &self.scratch.qsa_index,
            index_q_width,
            config.indexer_kv_heads * index_dim,
        );
        let index_q_norm = weights.resident(&qsa.indexer_q_norm)?;
        let index_k_norm = weights.resident(&qsa.indexer_k_norm)?;
        qwen4_qsa_norm_rope(
            gpu,
            &Qwen4QsaNormRope {
                values: &index_q,
                norm: index_q_norm,
                heads: config.indexer_n_heads,
                head_dim: index_dim,
                position: state.position,
                rotary_dim: index_dim.min(64),
            },
        )?;
        qwen4_qsa_norm_rope(
            gpu,
            &Qwen4QsaNormRope {
                values: &index_k,
                norm: index_k_norm,
                heads: config.indexer_kv_heads,
                head_dim: index_dim,
                position: 0,
                rotary_dim: index_dim.min(64),
            },
        )?;
        gpu.memcpy_dtod_at_auto(
            &state.raw_index_keys.buf,
            state.position * config.indexer_kv_heads * index_dim * 4,
            &index_k.buf,
            0,
            config.indexer_kv_heads * index_dim * 4,
        )?;

        let q_width = config.num_attention_heads * config.head_dim;
        let kv_width = config.num_key_value_heads * config.head_dim;
        let q_weight = weights.resident(&qsa.q)?;
        let k_weight = weights.resident(&qsa.k)?;
        let v_weight = weights.resident(&qsa.v)?;
        self.gemv(
            gpu,
            q_weight,
            input,
            &self.scratch.rotation,
            &self.scratch.qsa_qgate,
            2 * q_width,
            config.hidden_size,
        )?;
        self.gemv(
            gpu,
            k_weight,
            input,
            &self.scratch.rotation,
            &self.scratch.qsa_k,
            kv_width,
            config.hidden_size,
        )?;
        self.gemv(
            gpu,
            v_weight,
            input,
            &self.scratch.rotation,
            &self.scratch.qsa_v,
            kv_width,
            config.hidden_size,
        )?;
        let q_norm = weights.resident(&qsa.q_norm)?;
        let k_norm = weights.resident(&qsa.k_norm)?;
        let q_values = view(&self.scratch.qsa_qgate, 0, q_width);
        qwen4_qsa_norm_rope(
            gpu,
            &Qwen4QsaNormRope {
                values: &q_values,
                norm: q_norm,
                heads: config.num_attention_heads,
                head_dim: config.head_dim,
                position: state.position,
                rotary_dim: config.head_dim.min(64),
            },
        )?;
        qwen4_qsa_norm_rope(
            gpu,
            &Qwen4QsaNormRope {
                values: &self.scratch.qsa_k,
                norm: k_norm,
                heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                position: state.position,
                rotary_dim: config.head_dim.min(64),
            },
        )?;
        qwen4_qsa_cache_append(
            gpu,
            &Qwen4QsaCacheAppend {
                key: &self.scratch.qsa_k,
                value: &self.scratch.qsa_v,
                full_keys: &state.full_keys,
                full_values: &state.full_values,
                position: state.position,
                kv_width,
            },
        )?;
        let visible = state.position + 1;
        let complete = visible / config.indexer_compress_ratio;
        if complete > 0 {
            qwen4_qsa_pool_rope(
                gpu,
                &Qwen4QsaPoolRope {
                    raw_keys: &state.raw_index_keys,
                    pooled: &state.pooled_keys,
                    block_count: complete,
                    compress: config.indexer_compress_ratio,
                    index_dim: config.indexer_kv_heads * index_dim,
                },
            )?;
        }
        let budget_blocks = config.indexer_budget / config.indexer_compress_ratio;
        qwen4_qsa_select(
            gpu,
            &Qwen4QsaSelect {
                query: &index_q,
                pooled: &state.pooled_keys,
                selected: &state.selected_indices,
                block_count: complete,
                index_heads: config.indexer_n_heads,
                index_dim,
                budget_blocks,
                compress: config.indexer_compress_ratio,
                visible,
                capacity: state.selected_capacity,
            },
        )?;
        let selected = budget_blocks.min(complete) * config.indexer_compress_ratio + visible
            - complete * config.indexer_compress_ratio;
        let selected = selected.min(state.selected_capacity);
        qwen4_qsa_attention(
            gpu,
            &Qwen4QsaAttention {
                q_with_gate: &self.scratch.qsa_qgate,
                full_keys: &state.full_keys,
                full_values: &state.full_values,
                selected: &state.selected_indices,
                output: &self.scratch.qsa_output,
                n_heads: config.num_attention_heads,
                n_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                selected_len: selected,
                full_capacity: state.full_capacity,
            },
        )?;
        let out_weight = weights.resident(&qsa.output)?;
        self.gemv(
            gpu,
            out_weight,
            &self.scratch.qsa_output,
            &self.scratch.rotation,
            output,
            config.hidden_size,
            q_width,
        )?;
        state.full_len = visible;
        state.raw_len = visible;
        state.pooled_len = complete;
        state.selected_len = selected;
        state.position = visible;
        Ok(())
    }

    fn apply_ple(
        &self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
        weights: &Qwen4Weights,
        state: &mut GpuTensor,
        ple_rows: &GpuTensor,
        token_index: usize,
    ) -> Result<(), Qwen4GpuForwardError> {
        let layer = config
            .ple_layer_ids
            .first()
            .and_then(|id| id.checked_sub(1))
            .ok_or_else(|| invalid("PLE layer id missing"))?;
        let ple = weights
            .layer_refs
            .get(layer)
            .and_then(|layer| layer.ple.as_ref())
            .ok_or_else(|| invalid("PLE weights missing"))?;
        let channels = config.ple_embed_dim * config.hc_count;
        let row = f32_view(
            ple_rows,
            token_index * config.hidden_size,
            config.hidden_size,
        );
        let key_weight = weights.resident(&ple.key)?;
        let value_weight = weights.resident(&ple.value)?;
        self.gemv(
            gpu,
            key_weight,
            &row,
            &self.scratch.rotation,
            &self.scratch.ple_key,
            channels,
            config.hidden_size,
        )?;
        self.gemv(
            gpu,
            value_weight,
            &row,
            &self.scratch.rotation,
            &self.scratch.ple_value,
            config.hidden_size,
            config.hidden_size,
        )?;
        gpu.copy_d2d(
            &self.scratch.streams,
            &self.scratch.ple_query,
            self.scratch.ple_query.byte_size(),
        )?;
        let norm_key = weights.resident(&ple.norm_key)?;
        let norm_query = weights.resident(&ple.norm_query)?;
        gpu.qwen4_ple_gate_bf16(
            &self.scratch.ple_key,
            &self.scratch.ple_query,
            &self.scratch.ple_value,
            norm_key,
            norm_query,
            &self.scratch.ple_gated,
            1,
            config.hc_count,
            config.hidden_size,
            EPSILON,
        )?;
        let norm_conv = weights.resident(&ple.norm_conv)?;
        gpu.qwen4_ple_norm_bf16(
            &self.scratch.ple_gated,
            norm_conv,
            &self.scratch.ple_normed,
            1,
            config.hc_count,
            config.hidden_size,
            EPSILON,
        )?;
        let conv = weights.resident(&ple.conv)?;
        gpu.qwen4_ple_depthwise_conv_silu_add_bf16(
            &self.scratch.ple_gated,
            &self.scratch.ple_normed,
            conv,
            state,
            &self.scratch.ple_output,
            1,
            channels,
            config.ple_conv_kernel_size,
            3,
        )?;
        gpu.add_f32(
            &self.scratch.streams,
            &self.scratch.ple_output,
            &self.scratch.streams,
        )?;
        Ok(())
    }

    fn apply_moe(
        &self,
        gpu: &mut Gpu,
        config: &Qwen4Config,
        layer_index: usize,
        runtime: &Qwen4MoeLayerRuntime,
        input: &GpuTensor,
        output: &GpuTensor,
    ) -> Result<(), Qwen4GpuForwardError> {
        execute_moe(
            gpu,
            config,
            layer_index,
            runtime,
            input,
            output,
            Qwen4MoeScratch {
                router_logits: &self.scratch.router_logits,
                scalar_buf: &self.scratch.moe_scalar,
                x_rot_local: &self.scratch.moe_x_rot,
                gate_up_buf: &self.scratch.moe_gate_up,
                gate_buf: &self.scratch.moe_gate,
                up_buf: &self.scratch.moe_up,
                ffn_hidden: &self.scratch.moe_hidden,
                ffn_out: &self.scratch.moe_output,
                gate_batch: &self.scratch.moe_gate_batch,
                up_batch: &self.scratch.moe_up_batch,
                rot_batch: &self.scratch.moe_rot_batch,
                topk_indices: &self.scratch.moe_topk_indices,
                topk_weights: &self.scratch.moe_topk_weights,
                down_expanded: &self.scratch.moe_down_expanded,
            },
        )
    }
}
