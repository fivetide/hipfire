// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon-1-mini forward composition (`forward_token` / `forward_rows`). Owner: slice I.
//!
//! One layer schedule serves decode (`rows == 1`) and bounded prefill chunks
//! (`rows == N`): every stage is a module call over `rows` rows, so a chunk is
//! genuinely batched (projections, mHC, RDA, MLA, dense/MoE) and a single token
//! is the `rows == 1` instance of the same route.
//!
//! ```text
//! embed rows -> broadcast to the 4 streams (PLAN §1.2)
//! for layer in 0..L:
//!     attention site:  route(OLD streams) -> aggregate+RMS(input_layernorm)
//!                      -> mixer (RDA | MLA) -> postmix(OLD streams, branch)
//!     FFN site:        route(OLD streams) -> aggregate+RMS(post_attention_layernorm)
//!                      -> dense (layer 0) | MoE -> postmix
//! head: mean over streams -> RMS(model.norm) -> lm_head (BF16-rounded), last row only
//! ```
//!
//! All activations are F32 planes holding BF16-rounded values: every GEMM output
//! is rounded through `projection::OutRound::Bf16`; the stream-mix stages, the
//! RDA module, the latent-attention module and the MoE module apply their own
//! documented boundaries.
//!
//! Position bookkeeping: the MLA kernels read the slot's *device* `seq_len`
//! (no host-baked lengths), RDA is position-free (no RoPE anywhere). The
//! counter is advanced exactly once per call, after every layer consumed it.
//!
//! Failure semantics: an `Err` after the first layer may leave RDA state and
//! latent rows partially updated while `n_tokens` is unchanged; the caller must
//! `reset` the slot before reusing it.
//!
//! The final streams of the last call always live in `scratch.stream_a`
//! (50 ping-pong sites: an even count), which is what [`readout_logits`] reads.

use hipfire_dispatch::context::DispatchCtx;
use hipfire_dispatch::pipeline::latent_attention::{
    execute_latent_attention, ExpandedMirror as DMirror, LatentAttentionOp, LatentCache, LatentGeometry,
    RowRange,
};
use hipfire_dispatch::pipeline::stream_mix::{
    execute_stream_mix, StreamChart, StreamMixOp, StreamMixStage, StreamNorm,
};
use hipfire_runtime::llama::EmbeddingFormat;
use rdna_compute::{Gpu, GpuTensor};

use crate::config::{LayerKind, RaonConfig};
use crate::mhc::{broadcast_embedding, final_mean_norm, RaonMhcShape, RaonMhcWork};
use crate::mla::{raon_mla_scale, RAON_MLA_EPS};
use crate::moe::{
    forward_dense, forward_moe, RaonDenseRefs, RaonDenseScratch, RaonMoeDims, RaonMoeLayerRefs,
    RaonMoeScratch,
};
use crate::projection::{project, OutRound};
use crate::rda::{
    rda_forward_rows, RaonRdaLayer, RdaGeometry, RdaLayerState, RdaProjected, RdaScratch,
};
use crate::state::{RaonPrefillScratch, RaonState, EXPANDED_MIRROR_MAX_ROWS, MLA_TILE_ROWS};
use crate::weights::{
    MhcSiteWeights, MlaWeights, RaonFfn, RaonLayerWeights, RaonMixer, RaonWeights, RdaWeights,
};

// ───────────────────────── module scratch ─────────────────────────

/// Module-owned work buffers of the forward (mHC controls, RDA, MoE, dense
/// FFN), sized for `rows` rows like the [`RaonPrefillScratch`] they accompany.
/// Owned once per bundle; reused by every layer and every call.
pub struct RaonWork {
    rows: usize,
    mhc_shape: RaonMhcShape,
    mhc: RaonMhcWork,
    rda_geom: RdaGeometry,
    rda: RdaScratch,
    moe_dims: RaonMoeDims,
    moe: RaonMoeScratch,
    dense: RaonDenseScratch,
}

impl RaonWork {
    /// Allocate the module scratch for at most `rows` rows.
    pub fn new(gpu: &mut Gpu, cfg: &RaonConfig, rows: usize) -> Result<Self, String> {
        if rows == 0 {
            return Err("raon work: rows == 0".into());
        }
        let mhc_shape = RaonMhcShape {
            streams: cfg.mhc_streams,
            width: cfg.hidden,
            post_factor: cfg.mhc_post_factor,
            selector_eps: cfg.mhc_selector_eps,
            ..RaonMhcShape::raon_mini()
        };
        let lower_bound = cfg.rda_gate_lower_bound.ok_or_else(|| {
            "raon1_mini: RDA gate without a lower bound (-softplus form) is not implemented".to_string()
        })?;
        let rda_geom = RdaGeometry {
            n_heads: cfg.n_heads,
            head_dim: cfg.head_dim,
            conv_kernel: cfg.rda_conv_kernel,
            rms_eps: cfg.rms_eps,
            norm_eps: cfg.qk_l2_eps,
            gate_lower_bound: lower_bound,
        };
        let moe_dims = RaonMoeDims {
            hidden: cfg.hidden,
            n_experts: cfg.n_experts,
            top_k: cfg.top_k,
            logical_mi: cfg.moe_inter,
            down_k: cfg.moe_down_k,
            dense_mi: cfg.dense_inter,
            gate_beta: cfg.situ_beta,
            value_beta: cfg.situ_linear_beta,
            router_eps: cfg.router_eps,
            router_scale: cfg.routed_scale,
        };
        moe_dims.validate()?;
        let mhc = RaonMhcWork::new(gpu, &mhc_shape, rows)?;
        let rda = match RdaScratch::new(gpu, &rda_geom, rows) {
            Ok(r) => r,
            Err(e) => {
                let _ = mhc.free(gpu);
                return Err(e);
            }
        };
        let moe = match RaonMoeScratch::new(gpu, &moe_dims, rows) {
            Ok(m) => m,
            Err(e) => {
                let _ = mhc.free(gpu);
                free_rda_scratch(gpu, rda);
                return Err(e);
            }
        };
        let dense = match RaonDenseScratch::new(gpu, &moe_dims, rows) {
            Ok(d) => d,
            Err(e) => {
                let _ = mhc.free(gpu);
                free_rda_scratch(gpu, rda);
                moe.free(gpu);
                return Err(e);
            }
        };
        Ok(Self {
            rows,
            mhc_shape,
            mhc,
            rda_geom,
            rda,
            moe_dims,
            moe,
            dense,
        })
    }

    /// Rows this scratch supports.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Return every device buffer.
    pub fn free(self, gpu: &mut Gpu) {
        if let Err(e) = self.mhc.free(gpu) {
            eprintln!("raon1_mini: mhc work free: {e}");
        }
        free_rda_scratch(gpu, self.rda);
        self.moe.free(gpu);
        self.dense.free(gpu);
    }
}

fn free_rda_scratch(gpu: &mut Gpu, r: RdaScratch) {
    for t in [r.qconv, r.kconv, r.vconv, r.q, r.k, r.vprime, r.log_alpha, r.beta, r.y] {
        let _ = gpu.free_tensor(t);
    }
}

// ───────────────────────── taps ─────────────────────────

/// A point in the layer schedule at which the stream state can be observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TapSite {
    /// After the embedding broadcast.
    Embedding,
    /// After the attention-site postmix of this layer.
    Attn(usize),
    /// After the FFN-site postmix of this layer.
    Ffn(usize),
}

/// Observer of the stream state (`[rows, streams, hidden]` F32 holding BF16
/// values). Oracle only: a tap downloads and is not graph-capturable.
pub type LayerTap<'a> = &'a mut dyn FnMut(&mut Gpu, TapSite, &GpuTensor, usize) -> Result<(), String>;

fn fire(
    tap: &mut Option<LayerTap<'_>>,
    gpu: &mut Gpu,
    site: TapSite,
    streams: &GpuTensor,
    rows: usize,
) -> Result<(), String> {
    if let Some(t) = tap.as_mut() {
        (**t)(gpu, site, streams, rows)?;
    }
    Ok(())
}

// ───────────────────────── mHC site bindings ─────────────────────────

fn mix_op<'a>(shape: &RaonMhcShape, rows: usize, stage: StreamMixStage<'a>) -> StreamMixOp<'a> {
    StreamMixOp {
        rows,
        streams: shape.streams,
        width: shape.width,
        rounding: shape.rounding,
        stage,
    }
}

fn run_mix(gpu: &mut Gpu, op: &StreamMixOp<'_>) -> Result<(), String> {
    execute_stream_mix(gpu, op).map_err(|e| e.to_string())
}

/// Controls of one site from the OLD streams (selector/params are the loaded
/// `[17, n*hidden]` F32 matrix and the F32 parameter vector of the site).
fn mhc_route(
    gpu: &mut Gpu,
    work: &RaonWork,
    site: &MhcSiteWeights,
    rows: usize,
    streams: &GpuTensor,
) -> Result<(), String> {
    let shape = &work.mhc_shape;
    let op = mix_op(
        shape,
        rows,
        StreamMixStage::Route {
            streams,
            chart: StreamChart::HelmertBounded {
                selector: &site.selector.buf,
                params: &site.params,
                post_factor: shape.post_factor,
                selector_eps: shape.selector_eps,
            },
            selector_out: work.mhc.selector_out(),
            controls: work.mhc.controls(),
        },
    );
    run_mix(gpu, &op)
}

fn mhc_aggregate(
    gpu: &mut Gpu,
    work: &RaonWork,
    rows: usize,
    streams: &GpuTensor,
    norm: StreamNorm<'_>,
    out: &GpuTensor,
) -> Result<(), String> {
    let op = mix_op(
        &work.mhc_shape,
        rows,
        StreamMixStage::Aggregate {
            streams,
            controls: work.mhc.controls(),
            norm: Some(norm),
            out,
        },
    );
    run_mix(gpu, &op)
}

fn mhc_postmix(
    gpu: &mut Gpu,
    work: &RaonWork,
    rows: usize,
    streams: &GpuTensor,
    branch: &GpuTensor,
    streams_out: &GpuTensor,
) -> Result<(), String> {
    let op = mix_op(
        &work.mhc_shape,
        rows,
        StreamMixStage::Postmix {
            streams,
            branch,
            controls: work.mhc.controls(),
            streams_out,
        },
    );
    run_mix(gpu, &op)
}

// ───────────────────────── mixers ─────────────────────────

/// RDA layer: projections (BF16-rounded) -> `rda_forward_rows` -> `o` projection.
/// Reads `scratch.branch_in`, writes `scratch.branch_out`.
fn rda_mixer(
    gpu: &mut Gpu,
    w: &RdaWeights,
    ri: usize,
    s: &RaonState,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    rows: usize,
) -> Result<(), String> {
    let x = &scratch.branch_in;
    let b = OutRound::Bf16;
    project(gpu, &w.q_proj, x, &scratch.rda_q_raw, rows, b)?;
    project(gpu, &w.k_proj, x, &scratch.rda_k_raw, rows, b)?;
    project(gpu, &w.v_proj, x, &scratch.rda_v_raw, rows, b)?;
    project(gpu, &w.g_proj, x, &scratch.rda_z, rows, b)?;
    project(gpu, &w.b_proj, x, &scratch.rda_b, rows, b)?;
    project(gpu, &w.rda_w_proj, x, &scratch.rda_delta, rows, b)?;
    project(gpu, &w.f_a_proj, x, &scratch.rda_f_lat, rows, b)?;
    project(gpu, &w.f_b_proj, &scratch.rda_f_lat, &scratch.rda_f, rows, b)?;

    let layer = RaonRdaLayer {
        conv_w_q: &w.q_conv,
        conv_w_k: &w.k_conv,
        conv_w_v: &w.v_conv,
        a_log: &w.a_log,
        dt_bias: &w.dt_bias,
        o_norm_gamma: &w.o_norm,
    };
    let st = RdaLayerState {
        state: &s.rda[ri],
        hist_q: &s.conv_q[ri],
        hist_k: &s.conv_k[ri],
        hist_v: &s.conv_v[ri],
        slot: 0,
        state_slot_stride: work.rda_geom.state_elems(),
    };
    let proj = RdaProjected {
        q_raw: &scratch.rda_q_raw,
        k_raw: &scratch.rda_k_raw,
        v_raw: &scratch.rda_v_raw,
        z: &scratch.rda_z,
        b: &scratch.rda_b,
        delta: &scratch.rda_delta,
        f: &scratch.rda_f,
    };
    rda_forward_rows(gpu, &work.rda_geom, &layer, &st, &proj, &work.rda, &scratch.rda_gated, rows)?;
    project(gpu, &w.o_proj, &scratch.rda_gated, &scratch.branch_out, rows, b)
}

fn latent_geometry(cfg: &RaonConfig) -> LatentGeometry {
    LatentGeometry {
        n_heads: cfg.n_heads,
        nope: cfg.qk_nope,
        tail: cfg.qk_rope,
        v_dim: cfg.v_head,
        latent: cfg.kv_lora,
    }
}

fn run_latent(gpu: &mut Gpu, op: LatentAttentionOp<'_>) -> Result<(), String> {
    execute_latent_attention(gpu, &op).map_err(|e| e.to_string())
}

/// Expand cached latent rows `range` of one MLA layer into the K/V mirror
/// through the layer's own (quantized) `kv_b` projection:
/// gather -> `kv_b` projection (exact F32) -> BF16 scatter. `range` spans at most
/// [`MLA_TILE_ROWS`] rows.
#[allow(clippy::too_many_arguments)]
fn mla_expand(
    gpu: &mut Gpu,
    cfg: &RaonConfig,
    w: &MlaWeights,
    scratch: &RaonPrefillScratch,
    cache: LatentCache<'_>,
    mirror: DMirror<'_>,
    range: RowRange,
    count: usize,
) -> Result<(), String> {
    let geometry = latent_geometry(cfg);
    run_latent(
        gpu,
        LatentAttentionOp::GatherLatent {
            geometry,
            cache,
            range,
            out: &scratch.mla_latent_plane,
        },
    )?;
    project(
        gpu,
        &w.kv_b_proj,
        &scratch.mla_latent_plane,
        &scratch.mla_kv_tile,
        count,
        OutRound::Exact,
    )?;
    run_latent(
        gpu,
        LatentAttentionOp::Decompress {
            geometry,
            cache,
            mirror,
            expansion: hipfire_dispatch::pipeline::latent_attention::KvExpansion::External {
                projected: &scratch.mla_kv_tile,
            },
            range,
        },
    )
}

/// MLA layer: PLAN §1.6 with the expanded-mirror attention. Reads
/// `scratch.branch_in`, writes `scratch.branch_out`.
#[allow(clippy::too_many_arguments)]
fn mla_mixer(
    gpu: &mut Gpu,
    cfg: &RaonConfig,
    w: &MlaWeights,
    mi: usize,
    s: &RaonState,
    scratch: &RaonPrefillScratch,
    rows: usize,
    start: usize,
) -> Result<(), String> {
    let x = &scratch.branch_in;
    let b = OutRound::Bf16;
    let geometry = latent_geometry(cfg);
    let mirror = s
        .expanded_mirror()
        .ok_or("raon1_mini: MLA attention needs an admitted expanded K/V mirror")?;
    let cache = LatentCache {
        rows: &s.mla_cache[mi],
        capacity_rows: s.capacity_rows(),
        pos: &s.seq_len,
        host_pos: Some(start),
    };
    let dmirror = DMirror {
        knope: &mirror.knope[mi],
        vmir: &mirror.vmir[mi],
        capacity_rows: mirror.rows,
    };

    project(gpu, &w.q_a_proj, x, &scratch.mla_qa, rows, b)?;
    project(gpu, &w.kv_a_proj, x, &scratch.mla_kv_a, rows, b)?;
    project(gpu, &w.g_proj, x, &scratch.mla_g, rows, b)?;

    run_latent(
        gpu,
        LatentAttentionOp::NormRows {
            x: &scratch.mla_qa,
            gamma: &w.q_a_norm,
            out: &scratch.mla_qa_norm,
            rows,
            width: cfg.q_lora,
            eps: RAON_MLA_EPS,
        },
    )?;
    project(gpu, &w.q_b_proj, &scratch.mla_qa_norm, &scratch.mla_q, rows, b)?;
    run_latent(
        gpu,
        LatentAttentionOp::SplitKv {
            geometry,
            cache,
            kv_a: &scratch.mla_kv_a,
            gamma: &w.kv_a_norm,
            rows,
            eps: RAON_MLA_EPS,
        },
    )?;
    mla_expand(
        gpu,
        cfg,
        w,
        scratch,
        cache,
        dmirror,
        RowRange::NewRows { count: rows },
        rows,
    )?;
    run_latent(
        gpu,
        LatentAttentionOp::Attention {
            geometry,
            cache,
            mirror: dmirror,
            q: &scratch.mla_q,
            y: &scratch.mla_attn,
            rows,
            scale: raon_mla_scale(&geometry),
        },
    )?;
    run_latent(
        gpu,
        LatentAttentionOp::Gate {
            y: &scratch.mla_attn,
            g: &scratch.mla_g,
            out: &scratch.mla_attn,
            elements: rows * cfg.n_heads * cfg.v_head,
        },
    )?;
    project(gpu, &w.o_proj, &scratch.mla_attn, &scratch.branch_out, rows, b)
}

// ───────────────────────── FFN ─────────────────────────

fn ffn_block(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    work: &RaonWork,
    layer: &RaonLayerWeights,
    scratch: &RaonPrefillScratch,
    rows: usize,
) -> Result<(), String> {
    match &layer.ffn {
        RaonFfn::Dense(d) => {
            let refs = RaonDenseRefs::from_parts(&d.gate, &d.up, &d.down)?;
            forward_dense(
                gpu,
                ctx,
                &work.moe_dims,
                &refs,
                rows,
                &scratch.branch_in,
                &work.dense,
                &scratch.ffn_out,
            )
        }
        RaonFfn::Moe(m) => {
            let refs = RaonMoeLayerRefs::from_parts(
                &m.router,
                &m.e_bias,
                &m.shared.gate,
                &m.shared.up,
                &m.shared.down,
                &m.expert_gate_up_ptrs,
                &m.expert_down_ptrs,
                m.gate_up_awq.as_ref(),
                m.expert_down_awq_ptrs.as_ref(),
            );
            forward_moe(
                gpu,
                ctx,
                &work.moe_dims,
                &refs,
                rows,
                &scratch.branch_in,
                &work.moe,
                &scratch.ffn_out,
            )
        }
    }
}

// ───────────────────────── mirror admission ─────────────────────────

/// Make sure `s` can attend over `end` rows: map the latent caches ahead and
/// size the expanded K/V mirror (growing it re-expands the live prefix from the
/// latent cache through each layer's `kv_b`). Outside capture only.
fn ensure_context(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &RaonPrefillScratch,
    gpu: &mut Gpu,
    end: usize,
) -> Result<(), String> {
    s.ensure_capacity(gpu, end)?;
    if cfg.n_mla() == 0 {
        return Ok(());
    }
    let have = s.expanded_mirror().map_or(0, |m| m.rows);
    if end <= have {
        return Ok(());
    }
    let cap = EXPANDED_MIRROR_MAX_ROWS.min(s.max_seq());
    if end > cap {
        return Err(format!(
            "raon1_mini: context of {end} rows exceeds the {cap} rows the reference MLA attention can read \
             (expanded K/V mirror limit {EXPANDED_MIRROR_MAX_ROWS}, max_seq {}); a tiled/absorbed MLA route is not admitted",
            s.max_seq()
        ));
    }
    let target = end.next_multiple_of(1024).min(cap).max(end);
    let live = s.n_tokens();
    s.admit_expanded_mirror(gpu, target)?;
    if live == 0 {
        return Ok(());
    }
    let mirror = s.expanded_mirror().expect("mirror just admitted");
    for (mi, layer_idx) in cfg.mla_layers().into_iter().enumerate() {
        let RaonMixer::Mla(mw) = &w.layers[layer_idx].mixer else {
            return Err(format!("raon1_mini: layer {layer_idx} is not an MLA layer"));
        };
        let cache = LatentCache {
            rows: &s.mla_cache[mi],
            capacity_rows: s.capacity_rows(),
            pos: &s.seq_len,
            host_pos: None,
        };
        let dmirror = DMirror {
            knope: &mirror.knope[mi],
            vmir: &mirror.vmir[mi],
            capacity_rows: mirror.rows,
        };
        let mut done = 0usize;
        while done < live {
            let count = MLA_TILE_ROWS.min(scratch.rows).min(live - done);
            mla_expand(
                gpu,
                cfg,
                mw,
                scratch,
                cache,
                dmirror,
                RowRange::Absolute { start: done, count },
                count,
            )?;
            done += count;
        }
    }
    Ok(())
}

// ───────────────────────── embedding / head ─────────────────────────

fn embed_rows(
    cfg: &RaonConfig,
    w: &RaonWeights,
    scratch: &RaonPrefillScratch,
    work: &RaonWork,
    gpu: &mut Gpu,
    tokens: &[u32],
) -> Result<(), String> {
    let rows = tokens.len();
    let mut bytes = Vec::with_capacity(rows * 4);
    for &t in tokens {
        if t as usize >= cfg.vocab {
            return Err(format!("raon1_mini: token id {t} outside vocab {}", cfg.vocab));
        }
        bytes.extend_from_slice(&t.to_le_bytes());
    }
    gpu.memcpy_htod_auto(&scratch.token_ids.buf, &bytes)
        .map_err(|e| format!("raon1_mini: upload token ids: {e:?}"))?;
    let res = match w.embed_format {
        EmbeddingFormat::Q8_0 => {
            gpu.embedding_lookup_q8_batched(&w.embed, &scratch.branch_out, &scratch.token_ids, rows, cfg.hidden)
        }
        EmbeddingFormat::F32 => {
            gpu.embedding_lookup_f32_batched(&w.embed, &scratch.branch_out, &scratch.token_ids, rows, cfg.hidden)
        }
        other => {
            return Err(format!("raon1_mini: embedding format {other:?} is not routed (expected Q8F16 or F32)"));
        }
    };
    res.map_err(|e| format!("raon1_mini: embedding lookup: {e:?}"))?;
    broadcast_embedding(gpu, &work.mhc_shape, rows, &scratch.branch_out, cfg.hidden, &scratch.stream_a)
}

/// Final mean + RMSNorm + lm_head for rows `row0..row0+count` of
/// `scratch.stream_a` into `out` (`[count, vocab]` F32, BF16-valued).
/// Uses `scratch.branch_in` as the normalized-hidden staging plane.
pub fn readout_logits(
    cfg: &RaonConfig,
    w: &RaonWeights,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    row0: usize,
    count: usize,
    out: &GpuTensor,
) -> Result<(), String> {
    if count == 0 || row0 + count > scratch.rows || count > work.rows {
        return Err(format!(
            "raon1_mini: readout of rows {row0}..{} outside the {} scratch rows",
            row0 + count,
            scratch.rows
        ));
    }
    let width = cfg.mhc_streams * cfg.hidden;
    let view = scratch.stream_a.sub_offset(row0 * width, count * width);
    let norm = StreamNorm {
        gamma: &w.final_norm,
        eps: cfg.rms_eps,
    };
    final_mean_norm(gpu, &work.mhc_shape, count, &view, norm, &scratch.branch_in)?;
    project(gpu, &w.lm_head, &scratch.branch_in, out, count, OutRound::Bf16)
}

// ───────────────────────── the forward ─────────────────────────

#[allow(clippy::too_many_arguments)]
fn forward_impl(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    want_logits: bool,
    mut tap: Option<LayerTap<'_>>,
) -> Result<(), String> {
    let rows = tokens.len();
    if rows == 0 || rows > scratch.rows || rows > work.rows {
        return Err(format!(
            "raon1_mini: {rows} rows outside 1..={} (scratch rows {}, work rows {})",
            scratch.rows.min(work.rows),
            scratch.rows,
            work.rows
        ));
    }
    if w.layers.len() != cfg.n_layers {
        return Err(format!(
            "raon1_mini: weights hold {} layers, config {}",
            w.layers.len(),
            cfg.n_layers
        ));
    }
    let start = start_pos as usize;
    if start != s.n_tokens() {
        return Err(format!(
            "raon1_mini: position {start} does not match the slot's consumed tokens {}",
            s.n_tokens()
        ));
    }
    let end = start + rows;
    ensure_context(cfg, w, s, scratch, gpu, end)?;

    embed_rows(cfg, w, scratch, work, gpu, tokens)?;
    fire(&mut tap, gpu, TapSite::Embedding, &scratch.stream_a, rows)?;

    let mut ctx = DispatchCtx::new(gpu);
    ctx.batch_size = rows;
    let eps = cfg.rms_eps;

    // Ping-pong: `cur` holds the OLD streams of a site, `oth` receives the new.
    let mut cur_is_a = true;
    for (l, layer) in w.layers.iter().enumerate() {
        let kind = cfg.layer_kinds[l];

        // ── attention site ──
        {
            let (cur, oth) = if cur_is_a {
                (&scratch.stream_a, &scratch.stream_b)
            } else {
                (&scratch.stream_b, &scratch.stream_a)
            };
            mhc_route(gpu, work, &layer.attn_mhc, rows, cur)?;
            mhc_aggregate(
                gpu,
                work,
                rows,
                cur,
                StreamNorm { gamma: &layer.input_norm, eps },
                &scratch.branch_in,
            )?;
            match (kind, &layer.mixer) {
                (LayerKind::Rda, RaonMixer::Rda(rw)) => {
                    let ri = cfg.rda_index(l).ok_or_else(|| format!("raon1_mini: layer {l} has no RDA slot"))?;
                    rda_mixer(gpu, rw, ri, s, scratch, work, rows)
                        .map_err(|e| format!("raon1_mini L{l} rda: {e}"))?;
                }
                (LayerKind::Mla, RaonMixer::Mla(mw)) => {
                    let mi = cfg.mla_index(l).ok_or_else(|| format!("raon1_mini: layer {l} has no MLA slot"))?;
                    mla_mixer(gpu, cfg, mw, mi, s, scratch, rows, start)
                        .map_err(|e| format!("raon1_mini L{l} mla: {e}"))?;
                }
                _ => return Err(format!("raon1_mini: layer {l} weights do not match its configured mixer")),
            }
            mhc_postmix(gpu, work, rows, cur, &scratch.branch_out, oth)?;
        }
        cur_is_a = !cur_is_a;
        fire(
            &mut tap,
            gpu,
            TapSite::Attn(l),
            if cur_is_a { &scratch.stream_a } else { &scratch.stream_b },
            rows,
        )?;

        // ── FFN site ──
        {
            let (cur, oth) = if cur_is_a {
                (&scratch.stream_a, &scratch.stream_b)
            } else {
                (&scratch.stream_b, &scratch.stream_a)
            };
            mhc_route(gpu, work, &layer.ffn_mhc, rows, cur)?;
            mhc_aggregate(
                gpu,
                work,
                rows,
                cur,
                StreamNorm { gamma: &layer.post_attn_norm, eps },
                &scratch.branch_in,
            )?;
            ffn_block(gpu, &ctx, work, layer, scratch, rows).map_err(|e| format!("raon1_mini L{l} ffn: {e}"))?;
            mhc_postmix(gpu, work, rows, cur, &scratch.ffn_out, oth)?;
        }
        cur_is_a = !cur_is_a;
        fire(
            &mut tap,
            gpu,
            TapSite::Ffn(l),
            if cur_is_a { &scratch.stream_a } else { &scratch.stream_b },
            rows,
        )?;
    }
    debug_assert!(cur_is_a, "an even number of sites leaves the result in stream_a");
    if !cur_is_a {
        return Err("raon1_mini: odd number of mHC sites (layer count) — result not in stream_a".into());
    }

    if want_logits {
        let last = rows - 1;
        let width = cfg.mhc_streams * cfg.hidden;
        let view = scratch.stream_a.sub_offset(last * width, width);
        let norm = StreamNorm {
            gamma: &w.final_norm,
            eps,
        };
        final_mean_norm(gpu, &work.mhc_shape, 1, &view, norm, &scratch.final_norm)?;
        project(gpu, &w.lm_head, &scratch.final_norm, &scratch.logits, 1, OutRound::Bf16)?;
    }

    // One advance per call, after every MLA layer consumed the old length.
    s.set_n_tokens(gpu, end)
}

/// One token at position `pos` (must equal `s.n_tokens()`); advances the slot by
/// one. With `want_logits`, `scratch.logits` holds this token's `[vocab]` logits.
#[allow(clippy::too_many_arguments)]
pub fn forward_token(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    token: u32,
    pos: u32,
    want_logits: bool,
) -> Result<(), String> {
    forward_token_tapped(cfg, w, s, scratch, work, gpu, token, pos, want_logits, None)
}

/// [`forward_token`] with an optional stream observer.
#[allow(clippy::too_many_arguments)]
pub fn forward_token_tapped(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    token: u32,
    pos: u32,
    want_logits: bool,
    tap: Option<LayerTap<'_>>,
) -> Result<(), String> {
    forward_impl(cfg, w, s, scratch, work, gpu, &[token], pos, want_logits, tap)
}

/// `tokens.len()` (1..=`scratch.rows`) consecutive tokens starting at
/// `start_pos` (must equal `s.n_tokens()`) through one batched pass: every
/// projection, mHC site, RDA/MLA layer and FFN runs once over all rows. With
/// `want_logits`, `scratch.logits` holds the LAST row's logits. Advances the
/// slot by `tokens.len()`.
#[allow(clippy::too_many_arguments)]
pub fn forward_rows(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    want_logits: bool,
) -> Result<(), String> {
    forward_rows_tapped(cfg, w, s, scratch, work, gpu, tokens, start_pos, want_logits, None)
}

/// [`forward_rows`] with an optional stream observer.
#[allow(clippy::too_many_arguments)]
pub fn forward_rows_tapped(
    cfg: &RaonConfig,
    w: &RaonWeights,
    s: &mut RaonState,
    scratch: &mut RaonPrefillScratch,
    work: &mut RaonWork,
    gpu: &mut Gpu,
    tokens: &[u32],
    start_pos: u32,
    want_logits: bool,
    tap: Option<LayerTap<'_>>,
) -> Result<(), String> {
    forward_impl(cfg, w, s, scratch, work, gpu, tokens, start_pos, want_logits, tap)
}
