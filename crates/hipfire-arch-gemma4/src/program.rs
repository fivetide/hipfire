// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Declarative Gemma 4 decoder-layer program.
//!
//! Each layer is `[SandwichAttention, SandwichMlp, PerLayerInput?, Scale?]`:
//! the per-layer-input branch exists on E-series checkpoints, the scale step
//! when the learned layer scalar is not 1. This module only binds resident
//! weights, KV storage and scratch to the shared
//! `hipfire_dispatch::pipeline::sandwich` operations; route selection and
//! executor bodies live in dispatch. The same program serves single-token
//! decode (`rows == 1`) and batched prefill / verify (`rows > 1`).

use crate::config::{Gemma4Config, LayerType, RopeType};
use crate::gemma4::{Gemma4State, LayerWeights, PerLayerBranchWeights};
use hipfire_dispatch::families::gemv::WeightRef;
use hipfire_dispatch::pipeline::sandwich::{
    Activation, PerLayerInputOp, Rope, RopeKind, SandwichAttentionOp, SandwichKv, SandwichMlpOp,
    SandwichStream, ScaleOp, SoftcapOp,
};
use hipfire_dispatch::pipeline::{GemvInput, Step};
use hipfire_dispatch::types::RotationPlan;
use hipfire_runtime::llama::{KvCache, KvCacheExt, WeightTensor};
use rdna_compute::GpuTensor;

/// The typed RoPE of one layer type.
pub(crate) fn layer_rope(cfg: &Gemma4Config, layer_type: LayerType) -> Rope {
    match layer_type {
        LayerType::Sliding => Rope {
            kind: RopeKind::RotateHalf,
            theta: cfg.sliding_rope_theta,
        },
        LayerType::Full => {
            let head_dim = cfg.full_head_dim;
            let rot_pairs = match cfg.full_rope_type {
                RopeType::Proportional => {
                    ((head_dim as f32) * cfg.full_partial_rotary_factor * 0.5) as usize
                }
                RopeType::Default => head_dim / 2,
            };
            Rope {
                kind: RopeKind::PartialHalved { rot_pairs },
                theta: cfg.full_rope_theta,
            }
        }
    }
}

/// Per-layer-input scratch; present on E-series checkpoints.
pub(crate) struct PleScratch<'a> {
    /// `[rows, n_layers * width]` per-layer inputs of every row.
    pub inputs: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub act: &'a GpuTensor,
    pub out: &'a GpuTensor,
}

/// Activation scratch of one forward, `rows` rows each.
pub(crate) struct LayerScratch<'a> {
    pub x: &'a GpuTensor,
    pub residual: &'a GpuTensor,
    pub normed: &'a GpuTensor,
    pub attn_rot: &'a GpuTensor,
    pub mlp_rot: &'a GpuTensor,
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub attn_out: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub act: &'a GpuTensor,
    pub mlp_out: &'a GpuTensor,
    pub ple: Option<PleScratch<'a>>,
}

impl<'a> LayerScratch<'a> {
    /// The resident single-row scratch of `state`.
    pub fn decode(state: &'a Gemma4State) -> Self {
        let ple = match (
            &state.ple_projection_all,
            &state.ple_gate,
            &state.ple_hidden,
            &state.ple_out,
        ) {
            (Some(inputs), Some(gate), Some(act), Some(out)) => Some(PleScratch {
                inputs,
                gate,
                act,
                out,
            }),
            _ => None,
        };
        Self {
            x: &state.x,
            residual: &state.residual,
            normed: &state.tmp,
            attn_rot: &state.tmp_rot,
            mlp_rot: &state.tmp_rot,
            q: &state.q,
            k: &state.k,
            v: &state.v,
            attn_out: &state.attn_out,
            gate: &state.gate_ffn,
            up: &state.up_ffn,
            act: &state.ffn_hidden,
            mlp_out: &state.ffn_out,
            ple,
        }
    }
}

/// Binding of one forward over `rows` consecutive positions from `position`.
pub(crate) struct ProgramBinding<'a> {
    pub cfg: &'a Gemma4Config,
    pub state: &'a Gemma4State,
    pub rows: usize,
    pub position: usize,
    /// Per-row i32 positions (`rows > 1`).
    pub positions: Option<&'a GpuTensor>,
    pub scratch: LayerScratch<'a>,
}

struct LayerRefs<'a> {
    input_norm: &'a GpuTensor,
    post_attn_norm: &'a GpuTensor,
    pre_ffn_norm: &'a GpuTensor,
    post_ffn_norm: &'a GpuTensor,
    layer_scalar: f32,
    q: &'a WeightTensor,
    k: &'a WeightTensor,
    v: Option<&'a WeightTensor>,
    o: &'a WeightTensor,
    q_norm: &'a GpuTensor,
    k_norm: &'a GpuTensor,
    gate: &'a WeightTensor,
    up: &'a WeightTensor,
    down: &'a WeightTensor,
    ffn_hidden_dim: usize,
    per_layer: Option<&'a PerLayerBranchWeights>,
}

fn layer_refs(layer: &LayerWeights) -> LayerRefs<'_> {
    match layer {
        LayerWeights::Sliding(l) => LayerRefs {
            input_norm: &l.input_layernorm,
            post_attn_norm: &l.post_attention_layernorm,
            pre_ffn_norm: &l.pre_feedforward_layernorm,
            post_ffn_norm: &l.post_feedforward_layernorm,
            layer_scalar: l.layer_scalar_host,
            q: &l.q_proj,
            k: &l.k_proj,
            v: Some(&l.v_proj),
            o: &l.o_proj,
            q_norm: &l.q_norm,
            k_norm: &l.k_norm,
            gate: &l.gate_proj,
            up: &l.up_proj,
            down: &l.down_proj,
            ffn_hidden_dim: l.ffn_hidden_dim,
            per_layer: l.per_layer.as_ref(),
        },
        LayerWeights::Full(l) => LayerRefs {
            input_norm: &l.input_layernorm,
            post_attn_norm: &l.post_attention_layernorm,
            pre_ffn_norm: &l.pre_feedforward_layernorm,
            post_ffn_norm: &l.post_feedforward_layernorm,
            layer_scalar: l.layer_scalar_host,
            q: &l.q_proj,
            k: &l.k_proj,
            v: l.v_proj.as_ref(),
            o: &l.o_proj,
            q_norm: &l.q_norm,
            k_norm: &l.k_norm,
            gate: &l.gate_proj,
            up: &l.up_proj,
            down: &l.down_proj,
            ffn_hidden_dim: l.ffn_hidden_dim,
            per_layer: l.per_layer.as_ref(),
        },
    }
}

impl<'a> ProgramBinding<'a> {
    fn stream(&self) -> SandwichStream<'a> {
        SandwichStream {
            rows: self.rows,
            hidden: self.cfg.dim,
            eps: self.cfg.norm_eps,
            x: self.scratch.x,
            residual: self.scratch.residual,
            normed: self.scratch.normed,
        }
    }

    /// Append the steps of `layer_idx` to `steps`.
    pub fn layer(
        &self,
        layer_idx: usize,
        layer: &'a LayerWeights,
        steps: &mut Vec<Step<'a>>,
    ) -> Result<(), String> {
        let cfg = self.cfg;
        let st = self.state;
        let sc = &self.scratch;
        let layer_type = cfg.layer_types[layer_idx];
        if !matches!(
            (layer_type, layer),
            (LayerType::Sliding, LayerWeights::Sliding(_))
                | (LayerType::Full, LayerWeights::Full(_))
        ) {
            return Err(format!("gemma4 layer {layer_idx} type/weights mismatch"));
        }
        let shared = match cfg.kv_shared_source_layer_idx(layer_idx) {
            Some(source) => Some(st.kv_slot_for_layer[source]),
            None if cfg.is_kv_shared_layer(layer_idx) => {
                return Err(format!(
                    "gemma4 layer {layer_idx}: missing same-type KV sharing source"
                ));
            }
            None => None,
        };
        let (kv, head_dim, n_kv_heads, window): (&KvCache, _, _, _) = match layer_type {
            LayerType::Sliding => (
                &st.kv_sliding,
                cfg.sliding_head_dim,
                cfg.sliding_n_kv_heads,
                cfg.sliding_window,
            ),
            LayerType::Full => (&st.kv_full, cfg.full_head_dim, cfg.full_n_kv_heads, 0),
        };
        let w = layer_refs(layer);
        let slot = shared.unwrap_or(st.kv_slot_for_layer[layer_idx]);
        let writes = shared.is_none();
        steps.push(Step::SandwichAttention(SandwichAttentionOp {
            stream: self.stream(),
            position: self.position,
            n_heads: cfg.n_heads,
            n_kv_heads,
            head_dim,
            input_norm: w.input_norm,
            wq: w.q.dispatch_ref(),
            wk: writes.then(|| w.k.dispatch_ref()),
            wv: w.v.filter(|_| writes).map(WeightTensor::dispatch_ref),
            q_norm: w.q_norm,
            k_norm: w.k_norm,
            v_norm: Some(&st.v_norm_ones),
            q_scale: (head_dim as f32).sqrt(),
            rope: layer_rope(cfg, layer_type),
            kv: SandwichKv {
                tier: kv.tier_inputs(),
                k_cache: &kv.k_gpu[slot],
                v_cache: &kv.v_gpu[slot],
                physical_cap: kv.physical_cap,
                givens_cos: kv.givens_cos.as_ref(),
                givens_sin: kv.givens_sin.as_ref(),
                window,
                write: writes,
            },
            pos_buf: &st.pos_buf,
            positions: self.positions,
            wo: w.o.dispatch_ref(),
            post_norm: w.post_attn_norm,
            x_rot: sc.attn_rot,
            q: sc.q,
            k: sc.k,
            v: sc.v,
            attn_out: sc.attn_out,
            flash_partials: &st.q8_flash_partials,
        }));
        steps.push(Step::SandwichMlp(SandwichMlpOp {
            stream: self.stream(),
            pre_norm: w.pre_ffn_norm,
            w_gate: w.gate.dispatch_ref(),
            w_up: w.up.dispatch_ref(),
            w_down: w.down.dispatch_ref(),
            activation: Activation::GeluTanh,
            hidden_dim: w.ffn_hidden_dim,
            post_norm: w.post_ffn_norm,
            x_rot: sc.mlp_rot,
            gate: sc.gate,
            up: sc.up,
            act: sc.act,
            out: sc.mlp_out,
        }));
        if let Some(ple) = w.per_layer.filter(|_| cfg.hidden_size_per_layer_input != 0) {
            let s = sc
                .ple
                .as_ref()
                .ok_or_else(|| format!("gemma4 layer {layer_idx}: missing PLE scratch"))?;
            steps.push(Step::PerLayerInput(PerLayerInputOp {
                stream: self.stream(),
                layer: layer_idx,
                layer_width: cfg.hidden_size_per_layer_input,
                n_layers: cfg.n_layers,
                inputs: s.inputs,
                w_gate: ple.input_gate.dispatch_ref(),
                w_proj: ple.projection.dispatch_ref(),
                post_norm: &ple.post_input_norm,
                gate: s.gate,
                act: s.act,
                out: s.out,
            }));
        }
        if w.layer_scalar != 1.0 {
            steps.push(Step::Scale(ScaleOp {
                x: sc.x,
                factor: w.layer_scalar,
            }));
        }
        Ok(())
    }
}

/// Final norm, tied LM head and logit soft cap of one row `x` into `logits`.
/// `lm_head` is the caller-held `weights.lm_head.dispatch_ref()`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn head<'a>(
    cfg: &Gemma4Config,
    final_norm: &'a GpuTensor,
    lm_head: &'a WeightRef<'a>,
    x: &'a GpuTensor,
    normed: &'a GpuTensor,
    logits: &'a GpuTensor,
    steps: &mut Vec<Step<'a>>,
) {
    steps.push(Step::RmsnormAutomatic {
        x,
        norm_weight: final_norm,
        x_plain: normed,
        out: normed,
        awq_scale: None,
        k: cfg.dim,
        eps: cfg.norm_eps,
        rotation: RotationPlan::None,
    });
    steps.push(Step::Gemv {
        w: lm_head,
        input: GemvInput::Raw(normed),
        out: logits,
    });
    if cfg.final_logit_softcapping > 0.0 {
        steps.push(Step::Softcap(SoftcapOp {
            logits,
            n: cfg.vocab_size,
            cap: cfg.final_logit_softcapping,
        }));
    }
}
