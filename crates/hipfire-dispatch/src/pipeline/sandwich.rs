// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Sandwich-norm decoder layer operations.
//!
//! A sandwich-norm decoder normalizes each sublayer's input AND its output
//! before the residual add (`x = residual + post_norm(f(pre_norm(x)))`).
//! Architecture crates declare one attention and one MLP operation per layer
//! (plus an optional per-layer-input branch and layer scale) and bind their
//! resident weights, KV storage and fixed scratch; every dtype-, tier-, arch-
//! and fusion-gated route choice lives here.
//!
//! Executors take `rows`: `rows == 1` is single-token decode, `rows > 1` a
//! batch of consecutive positions (prefill or speculative verify) laid out
//! row-major. The bodies are the former Gemma 4 eager decode and batched
//! verify arms moved verbatim, so every route issues the same launches.

use crate::context::DispatchCtx;
use crate::families::attention::{AttentionFamily, AttnParams};
use crate::families::fused_qkv::{FusedQkvFamily, FusedQkvParams};
use crate::families::gemm::{GemmFamily, GemmParams};
use crate::families::gemv::{GemvFamily, GemvParams, WeightRef};
use crate::families::kv_tier::{KvTierInputs, KvTierPlan};
use crate::types::{DispatchError, GemvVariant, KernelKey};
use hip_bridge::DeviceBuffer;
use rdna_compute::{DType, Gpu, GpuTensor};
use std::sync::LazyLock;

fn hip(e: impl std::fmt::Display) -> DispatchError {
    DispatchError::Hip(e.to_string())
}

fn hip_dbg(e: impl std::fmt::Debug) -> DispatchError {
    DispatchError::Hip(format!("{e:?}"))
}

/// Developer route switches. Each fused route is on unless its variable is
/// `0`/`off`/`false`. `HIPFIRE_GEMMA4_EAGLE=1` (greedy EAGLE) additionally
/// turns off the fusions that are not byte-identical to their unfused form,
/// so batched verify and single-row decode share one arithmetic.
struct Routes {
    attn_norm: bool,
    q8_qk: bool,
    qk_norm_rope: bool,
    post_norm: bool,
    gate_up: bool,
    /// Batched Q8 GEMMs use the unchunked scalar kernel (decode's arithmetic).
    strict_q8_gemm: bool,
}

static ROUTES: LazyLock<Routes> = LazyLock::new(|| {
    let on = |name: &str| {
        !matches!(
            hipfire_config::developer_var(name).ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    };
    let strict = hipfire_config::developer_var("HIPFIRE_GEMMA4_EAGLE")
        .ok()
        .as_deref()
        == Some("1");
    Routes {
        attn_norm: on("HIPFIRE_GEMMA4_FUSED_ATTN_NORM"),
        q8_qk: on("HIPFIRE_GEMMA4_FUSED_QK"),
        qk_norm_rope: !strict && on("HIPFIRE_GEMMA4_FUSED_QK_ROPE"),
        post_norm: !strict && on("HIPFIRE_GEMMA4_FUSED_POSTNORM"),
        gate_up: on("HIPFIRE_GEMMA4_FUSED_FFN"),
        strict_q8_gemm: strict,
    }
});

static GEMV: LazyLock<GemvFamily> = LazyLock::new(GemvFamily::new);
static GEMM: LazyLock<GemmFamily> = LazyLock::new(GemmFamily::new);
static FUSED_QKV: LazyLock<FusedQkvFamily> = LazyLock::new(FusedQkvFamily::new);
static ATTENTION: LazyLock<AttentionFamily> = LazyLock::new(AttentionFamily::new);

/// Arches whose batched sandwich prefill admits the exact fused and wide
/// routes below. Elsewhere batched rows take the per-row fallbacks.
fn batched_fusion_arch(arch: &str) -> bool {
    matches!(arch, "gfx1100" | "gfx1201")
}

fn gemv(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w: &WeightRef,
    x: &GpuTensor,
    y: &GpuTensor,
) -> Result<(), DispatchError> {
    GEMV.run_auto(ctx, gpu, w, x, y)
}

fn gemv_prerotated(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    w: &WeightRef,
    x_rot: &GpuTensor,
    y: &GpuTensor,
) -> Result<(), DispatchError> {
    GEMV.run(
        ctx,
        gpu,
        &GemvParams {
            w,
            x: x_rot,
            y,
            variant: GemvVariant::Prerotated,
            residual: None,
            gate: None,
            up: None,
        },
    )
}

/// `y[rows, m] = x[rows, k] · Wᵀ`. MagnumQuant weights rotate `x` into
/// `x_rot` (`[rows, k]`) first.
pub fn gemm_rows(
    gpu: &mut Gpu,
    w: &WeightRef,
    x: &GpuTensor,
    y: &GpuTensor,
    x_rot: &GpuTensor,
    rows: usize,
) -> Result<(), DispatchError> {
    match w.dtype {
        DType::F32 => gpu
            .gemm_f32_batched(w.buf, x, y, w.m, w.k, rows)
            .map_err(hip),
        DType::Q8_0 if ROUTES.strict_q8_gemm => gpu
            .gemm_q8_0_batched(w.buf, x, y, w.m, w.k, rows)
            .map_err(hip),
        DType::Q8_0 => gpu
            .gemm_q8_0_batched_chunked(w.buf, x, y, w.m, w.k, rows)
            .map_err(hip),
        DType::MQ4G256 | DType::HFQ4G256 => {
            crate::pipeline::batched::rotate_x_mq_batched_for(gpu, w, x, x_rot, w.k, rows)
                .map_err(hip)?;
            gpu.gemm_hfq4g256_batched_lmhead(w.buf, x_rot, y, w.m, w.k, rows)
                .map_err(hip)
        }
        DType::MQ6G256 => {
            crate::pipeline::batched::rotate_x_mq_batched_for(gpu, w, x, x_rot, w.k, rows)
                .map_err(hip)?;
            gpu.gemm_mq6g256_batched_lmhead(w.buf, x_rot, y, w.m, w.k, rows)
                .map_err(hip)
        }
        other => Err(DispatchError::Hip(format!(
            "sandwich: dtype {other:?} has no batched projection kernel"
        ))),
    }
}

/// Weight formats [`gemm_rows`] can project with `rows > 1`.
pub fn supports_batched_projection(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::F32 | DType::Q8_0 | DType::MQ4G256 | DType::HFQ4G256 | DType::MQ6G256
    )
}

fn fused_projection(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    key: KernelKey,
    weights: &[&WeightRef],
    x: &GpuTensor,
    outputs: &[&GpuTensor],
    rows: usize,
) -> Result<(), DispatchError> {
    let bufs: Vec<&GpuTensor> = weights.iter().map(|w| w.buf).collect();
    let m: Vec<usize> = weights.iter().map(|w| w.m).collect();
    FUSED_QKV.run(
        ctx,
        gpu,
        &FusedQkvParams {
            kind: key,
            weights: &bufs,
            x,
            outputs,
            m: &m,
            k: weights[0].k,
            rot_scratch: &[],
            batch_size: Some(rows),
        },
    )
}

/// gfx1100 opt-in: Q8 projections sharing one input run as one fused WMMA GEMM.
fn q8_fused_prefill(gpu: &Gpu, weights: &[&WeightRef]) -> bool {
    gpu.arch == "gfx1100"
        && gpu.flags.gemma4_q8_fused_prefill
        && weights
            .iter()
            .all(|w| w.dtype == DType::Q8_0 && w.k == weights[0].k)
        && weights[0].k.is_multiple_of(32)
}

fn scale(gpu: &mut Gpu, x: &GpuTensor, factor: f32) -> Result<(), DispatchError> {
    rdna_compute::tensor_ops::scale_f32(
        gpu,
        &rdna_compute::tensor_ops::ScaleF32 {
            values: x,
            scale: factor,
        },
    )
    .map_err(hip)
}

/// Residual stream and the fixed scratch every sandwich sublayer shares.
/// With `rows > 1` each tensor holds `rows` row-major rows.
#[derive(Clone, Copy)]
pub struct SandwichStream<'a> {
    pub rows: usize,
    pub hidden: usize,
    pub eps: f32,
    /// Residual stream; updated in place.
    pub x: &'a GpuTensor,
    /// Holds `x` at sublayer entry.
    pub residual: &'a GpuTensor,
    /// Normed activation / sublayer output scratch.
    pub normed: &'a GpuTensor,
}

impl SandwichStream<'_> {
    fn bytes(&self) -> usize {
        self.rows * self.hidden * 4
    }

    fn save_residual(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if self.rows == 1 {
            gpu.memcpy_dtod_auto(&self.residual.buf, &self.x.buf, self.bytes())
        } else {
            gpu.hip
                .memcpy_dtod_at(&self.residual.buf, 0, &self.x.buf, 0, self.bytes())
        }
        .map_err(hip)
    }

    fn restore_residual(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        if self.rows == 1 {
            gpu.memcpy_dtod_auto(&self.x.buf, &self.residual.buf, self.bytes())
        } else {
            gpu.hip
                .memcpy_dtod_at(&self.x.buf, 0, &self.residual.buf, 0, self.bytes())
        }
        .map_err(hip)
    }

    fn norm(
        &self,
        gpu: &mut Gpu,
        x: &GpuTensor,
        w: &GpuTensor,
        out: &GpuTensor,
    ) -> Result<(), DispatchError> {
        if self.rows == 1 {
            gpu.rmsnorm_f32(x, w, out, self.eps)
        } else {
            gpu.rmsnorm_batched(x, w, out, self.rows, self.hidden, self.eps)
        }
        .map_err(hip)
    }

    /// `x = residual + post_norm(out)`; `out` may alias `normed`.
    fn post_norm_residual(
        &self,
        gpu: &mut Gpu,
        out: &GpuTensor,
        post_norm: &GpuTensor,
    ) -> Result<(), DispatchError> {
        if self.rows == 1 && ROUTES.post_norm {
            return gpu
                .rmsnorm_residual_add_f32(out, post_norm, self.residual, self.x, self.eps)
                .map_err(hip);
        }
        self.norm(gpu, out, post_norm, self.normed)?;
        self.restore_residual(gpu)?;
        gpu.add_inplace_f32(self.x, self.normed).map_err(hip)
    }
}

/// Rotary position embedding applied to Q and K.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeKind {
    /// Every channel pair of the head rotates.
    RotateHalf,
    /// The first `rot_pairs` pairs of each half rotate; the rest pass through.
    PartialHalved { rot_pairs: usize },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rope {
    pub kind: RopeKind,
    pub theta: f32,
}

impl Rope {
    fn rot_pairs(&self, head_dim: usize) -> usize {
        match self.kind {
            RopeKind::RotateHalf => head_dim / 2,
            RopeKind::PartialHalved { rot_pairs } => rot_pairs,
        }
    }
}

/// The K/V cache one attention sublayer reads, and whether it writes the
/// current rows first. `write == false` reads another layer's populated cache
/// (KV sharing, draft heads).
pub struct SandwichKv<'a> {
    pub tier: KvTierInputs,
    pub k_cache: &'a GpuTensor,
    pub v_cache: &'a GpuTensor,
    pub physical_cap: usize,
    pub givens_cos: Option<&'a GpuTensor>,
    pub givens_sin: Option<&'a GpuTensor>,
    /// Sliding window; `0` = full causal.
    pub window: usize,
    pub write: bool,
}

/// Attention sublayer:
/// `x = residual + post_norm(o(attend(rope(q_norm(q)), rope(k_norm(k)), v_norm(v))))`
/// over `input_norm(x)`.
///
/// `wk == None` projects only Q and attends a cache it does not write.
/// `wv == None` with `wk` present takes V from the pre-norm K (K=V).
/// Row `r` sits at absolute position `position + r`.
pub struct SandwichAttentionOp<'a> {
    pub stream: SandwichStream<'a>,
    pub position: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub input_norm: &'a GpuTensor,
    pub wq: WeightRef<'a>,
    pub wk: Option<WeightRef<'a>>,
    pub wv: Option<WeightRef<'a>>,
    pub q_norm: &'a GpuTensor,
    pub k_norm: &'a GpuTensor,
    /// Weight-less V RMSNorm, bound as a ones vector of `head_dim`.
    pub v_norm: Option<&'a GpuTensor>,
    /// Q multiplier applied after q_norm (sets the softmax scale against the
    /// kernel's `1/sqrt(head_dim)`).
    pub q_scale: f32,
    pub rope: Rope,
    pub kv: SandwichKv<'a>,
    /// Device position scalar (`rows == 1`).
    pub pos_buf: &'a DeviceBuffer,
    /// Per-row i32 positions (`rows > 1`).
    pub positions: Option<&'a GpuTensor>,
    pub wo: WeightRef<'a>,
    pub post_norm: &'a GpuTensor,
    /// FWHT rotation scratch, `[rows, max(hidden, n_heads*head_dim)]`.
    pub x_rot: &'a GpuTensor,
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub attn_out: &'a GpuTensor,
    pub flash_partials: &'a GpuTensor,
}

impl SandwichAttentionOp<'_> {
    fn projections(&self) -> [Option<(&WeightRef<'_>, &GpuTensor)>; 3] {
        [
            Some((&self.wq, self.q)),
            self.wk.as_ref().map(|w| (w, self.k)),
            self.wv.as_ref().map(|w| (w, self.v)),
        ]
    }

    fn kv_bytes(&self) -> usize {
        self.stream.rows * self.n_kv_heads * self.head_dim * 4
    }

    /// `input_norm(x)` → Q, K, V projections (V from K when K=V).
    fn project(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        let s = &self.stream;
        if s.rows > 1 {
            return self.project_rows(gpu, ctx);
        }
        let all_mq4 = self
            .projections()
            .iter()
            .flatten()
            .all(|(w, _)| w.dtype == DType::MQ4G256);
        if ROUTES.attn_norm && all_mq4 {
            gpu.fused_rmsnorm_rotate_mq(s.x, self.input_norm, self.x_rot, s.hidden, s.eps)
                .map_err(hip)?;
            for (w, out) in self.projections().iter().flatten() {
                gemv_prerotated(gpu, ctx, w, self.x_rot, out)?;
            }
        } else {
            s.norm(gpu, s.x, self.input_norm, s.normed)?;
            match &self.wk {
                Some(wk)
                    if ROUTES.q8_qk && self.wq.dtype == DType::Q8_0 && wk.dtype == DType::Q8_0 =>
                {
                    gpu.fused_gate_up_q8_0(
                        self.wq.buf,
                        wk.buf,
                        s.normed,
                        self.q,
                        self.k,
                        self.wq.m,
                        wk.m,
                        self.wq.k,
                    )
                    .map_err(hip)?;
                }
                Some(wk) => {
                    gemv(gpu, ctx, &self.wq, s.normed, self.q)?;
                    gemv(gpu, ctx, wk, s.normed, self.k)?;
                }
                None => gemv(gpu, ctx, &self.wq, s.normed, self.q)?,
            }
            if let Some(wv) = &self.wv {
                gemv(gpu, ctx, wv, s.normed, self.v)?;
            }
        }
        if self.wk.is_some() && self.wv.is_none() {
            gpu.memcpy_dtod_auto(&self.v.buf, &self.k.buf, self.kv_bytes())
                .map_err(hip)?;
        }
        Ok(())
    }

    fn project_rows(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        let s = &self.stream;
        let rows = s.rows;
        s.norm(gpu, s.x, self.input_norm, s.normed)?;
        let mut v_projected = false;
        match (&self.wk, &self.wv) {
            (Some(wk), Some(wv)) if q8_fused_prefill(gpu, &[&self.wq, wk, wv]) => {
                fused_projection(
                    gpu,
                    ctx,
                    KernelKey::FusedQkvQ8_0,
                    &[&self.wq, wk, wv],
                    s.normed,
                    &[self.q, self.k, self.v],
                    rows,
                )?;
                v_projected = true;
            }
            (Some(wk), None) if q8_fused_prefill(gpu, &[&self.wq, wk]) => {
                fused_projection(
                    gpu,
                    ctx,
                    KernelKey::FusedGateUpQ8_0,
                    &[&self.wq, wk],
                    s.normed,
                    &[self.q, self.k],
                    rows,
                )?;
            }
            (wk, wv) => {
                gemm_rows(gpu, &self.wq, s.normed, self.q, self.x_rot, rows)?;
                if let Some(wk) = wk {
                    gemm_rows(gpu, wk, s.normed, self.k, self.x_rot, rows)?;
                }
                if let (Some(_), Some(wv)) = (wk, wv) {
                    gemm_rows(gpu, wv, s.normed, self.v, self.x_rot, rows)?;
                    v_projected = true;
                }
            }
        }
        if self.wk.is_some() && !v_projected {
            gpu.hip
                .memcpy_dtod_at(&self.v.buf, 0, &self.k.buf, 0, self.kv_bytes())
                .map_err(hip)?;
        }
        Ok(())
    }

    /// Per-head q_norm/k_norm, weight-less v_norm, Q prescale, RoPE.
    fn position_encode(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let eps = self.stream.eps;
        let rows = self.stream.rows;
        let (n_heads, n_kv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);
        let writes_k = self.wk.is_some();
        if writes_k {
            if let Some(ones) = self.v_norm {
                gpu.rmsnorm_batched(self.v, ones, self.v, rows * n_kv, hd, eps)
                    .map_err(hip)?;
            }
        }
        if rows == 1 && writes_k && ROUTES.qk_norm_rope {
            return gpu
                .fused_gemma4_qk_norm_rope_f32(
                    self.q,
                    self.k,
                    self.q_norm,
                    self.k_norm,
                    self.pos_buf,
                    n_heads,
                    n_kv,
                    hd,
                    self.rope.rot_pairs(hd),
                    self.q_scale,
                    self.rope.theta,
                    eps,
                )
                .map_err(hip);
        }
        gpu.rmsnorm_batched(self.q, self.q_norm, self.q, rows * n_heads, hd, eps)
            .map_err(hip)?;
        if writes_k {
            gpu.rmsnorm_batched(self.k, self.k_norm, self.k, rows * n_kv, hd, eps)
                .map_err(hip)?;
        }
        scale(gpu, self.q, self.q_scale)?;
        let rope_kv = if writes_k { n_kv } else { 0 };
        let theta = self.rope.theta;
        match (self.rope.kind, rows) {
            (RopeKind::RotateHalf, 1) => {
                gpu.rope_f32(self.q, self.k, self.pos_buf, n_heads, rope_kv, hd, theta)
            }
            (RopeKind::PartialHalved { rot_pairs }, 1) => gpu.rope_partial_halved_f32(
                self.q,
                self.k,
                self.pos_buf,
                n_heads,
                rope_kv,
                hd,
                rot_pairs,
                theta,
            ),
            (RopeKind::RotateHalf, _) => gpu.rope_batched_f32(
                self.q,
                self.k,
                self.positions()?,
                n_heads,
                rope_kv,
                hd,
                theta,
                rows,
            ),
            (RopeKind::PartialHalved { rot_pairs }, _) => gpu.rope_partial_halved_f32_batched(
                self.q,
                self.k,
                self.positions()?,
                n_heads,
                rope_kv,
                hd,
                rot_pairs,
                theta,
                rows,
            ),
        }
        .map_err(hip)
    }

    fn positions(&self) -> Result<&GpuTensor, DispatchError> {
        self.positions.ok_or_else(|| {
            DispatchError::Hip("sandwich attention: rows > 1 needs per-row positions".into())
        })
    }

    fn attend(&self, gpu: &mut Gpu, ctx: &DispatchCtx) -> Result<(), DispatchError> {
        if self.stream.rows > 1 {
            return self.attend_rows(gpu);
        }
        let kv = &self.kv;
        let plan = KvTierPlan::derive(KvTierInputs {
            pos: self.position,
            batch_size: 1,
            q8_windowed: true,
            window: kv.window as i32,
            ..kv.tier
        })
        .map_err(hip)?;
        let io = AttnParams {
            q: self.q,
            k: self.k,
            v: self.v,
            k_cache: kv.k_cache,
            v_cache: kv.v_cache,
            k_scales: None,
            v_scales: None,
            pos_buf: self.pos_buf,
            pos: self.position,
            positions: None,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            physical_cap: kv.physical_cap,
            batch_size: 1,
            max_ctx_len: 0,
            flash_partials: Some(self.flash_partials),
            givens_cos: kv.givens_cos,
            givens_sin: kv.givens_sin,
            tree_bias: None,
            block_start: 0,
            block_cols: 0,
            output_gate: None,
            output_awq_scale: None,
            output: self.attn_out,
        };
        if kv.write {
            ATTENTION.run_attention(ctx, gpu, &plan, &io)
        } else {
            ATTENTION.run_attend_only(ctx, gpu, &plan, &io)
        }
    }

    /// Causal attention for `rows` consecutive positions over a Q8 cache.
    fn attend_rows(&self, gpu: &mut Gpu) -> Result<(), DispatchError> {
        let kv = &self.kv;
        if !kv.tier.quant_q8 {
            return Err(DispatchError::Hip(
                "sandwich attention: rows > 1 needs a Q8 KV cache".into(),
            ));
        }
        let rows = self.stream.rows;
        let positions = self.positions()?;
        let (n_heads, n_kv, hd) = (self.n_heads, self.n_kv_heads, self.head_dim);
        let seq_len = self.position + rows;
        if kv.write {
            gpu.kv_cache_write_q8_0_batched(kv.k_cache, self.k, positions, n_kv, hd, rows)
                .map_err(hip)?;
            gpu.kv_cache_write_q8_0_batched(kv.v_cache, self.v, positions, n_kv, hd, rows)
                .map_err(hip)?;
        }
        // Prefill is not tree verification: the native causal kernels, and the
        // windowed tile kernel for sliding layers.
        if kv.window > 0 {
            gpu.attention_flash_q8_0_batched_masked_windowed(
                self.q,
                kv.k_cache,
                kv.v_cache,
                self.attn_out,
                positions,
                n_heads,
                n_kv,
                hd,
                kv.physical_cap,
                seq_len,
                rows,
                self.flash_partials,
                None,
                0,
                0,
                kv.window as i32,
            )
        } else if seq_len > 8_192 {
            gpu.attention_flash_q8_0_batched_masked(
                self.q,
                kv.k_cache,
                kv.v_cache,
                self.attn_out,
                positions,
                n_heads,
                n_kv,
                hd,
                kv.physical_cap,
                seq_len,
                rows,
                self.flash_partials,
                None,
                0,
                0,
            )
        } else {
            gpu.attention_q8_0_kv_batched_masked(
                self.q,
                kv.k_cache,
                kv.v_cache,
                self.attn_out,
                positions,
                n_heads,
                n_kv,
                hd,
                kv.physical_cap,
                seq_len,
                rows,
                None,
                0,
                0,
            )
        }
        .map_err(hip)
    }
}

pub fn execute_sandwich_attention(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SandwichAttentionOp<'_>,
) -> Result<(), DispatchError> {
    if op.wk.is_none() && op.kv.write {
        return Err(DispatchError::Hip(
            "sandwich attention: a Q-only layer cannot write its KV row".into(),
        ));
    }
    let s = op.stream;
    s.save_residual(gpu)?;
    op.project(gpu, ctx)?;
    op.position_encode(gpu)?;
    op.attend(gpu, ctx)?;
    if s.rows == 1 {
        gemv(gpu, ctx, &op.wo, op.attn_out, s.normed)?;
    } else {
        gemm_rows(gpu, &op.wo, op.attn_out, s.normed, op.x_rot, s.rows)?;
    }
    s.post_norm_residual(gpu, s.normed, op.post_norm)
}

/// Gated MLP activation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Activation {
    /// `gelu_pytorch_tanh(gate) * up`.
    GeluTanh,
}

/// MLP sublayer: `x = residual + post_norm(down(act(gate(n)) * up(n)))`,
/// `n = pre_norm(x)`.
pub struct SandwichMlpOp<'a> {
    pub stream: SandwichStream<'a>,
    pub pre_norm: &'a GpuTensor,
    pub w_gate: WeightRef<'a>,
    pub w_up: WeightRef<'a>,
    pub w_down: WeightRef<'a>,
    pub activation: Activation,
    pub hidden_dim: usize,
    pub post_norm: &'a GpuTensor,
    /// FWHT rotation scratch, `[rows, max(hidden, hidden_dim)]`.
    pub x_rot: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub act: &'a GpuTensor,
    pub out: &'a GpuTensor,
}

pub fn execute_sandwich_mlp(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &SandwichMlpOp<'_>,
) -> Result<(), DispatchError> {
    let s = op.stream;
    let rows = s.rows;
    s.save_residual(gpu)?;
    if rows > 1 {
        s.norm(gpu, s.x, op.pre_norm, s.normed)?;
        if q8_fused_prefill(gpu, &[&op.w_gate, &op.w_up]) {
            fused_projection(
                gpu,
                ctx,
                KernelKey::FusedGateUpQ8_0,
                &[&op.w_gate, &op.w_up],
                s.normed,
                &[op.gate, op.up],
                rows,
            )?;
        } else {
            gemm_rows(gpu, &op.w_gate, s.normed, op.gate, op.x_rot, rows)?;
            gemm_rows(gpu, &op.w_up, s.normed, op.up, op.x_rot, rows)?;
        }
    } else if ROUTES.gate_up && op.w_gate.dtype == DType::MQ4G256 && op.w_up.dtype == DType::MQ4G256
    {
        gpu.fused_rmsnorm_rotate_mq(s.x, op.pre_norm, op.x_rot, s.hidden, s.eps)
            .map_err(hip)?;
        gpu.fused_gate_up_hfq4g256(
            op.w_gate.buf,
            op.w_up.buf,
            op.x_rot,
            op.gate,
            op.up,
            op.w_gate.m,
            op.w_up.m,
            op.w_gate.k,
        )
        .map_err(hip)?;
    } else {
        s.norm(gpu, s.x, op.pre_norm, s.normed)?;
        gemv(gpu, ctx, &op.w_gate, s.normed, op.gate)?;
        gemv(gpu, ctx, &op.w_up, s.normed, op.up)?;
    }
    match op.activation {
        Activation::GeluTanh => {
            gpu.gelu_tanh_f32(op.gate, op.act, rows * op.hidden_dim)
                .map_err(hip)?;
            gpu.mul_f32(op.act, op.up, op.act).map_err(hip)?;
        }
    }
    if rows == 1 {
        gemv(gpu, ctx, &op.w_down, op.act, op.out)?;
    } else {
        gemm_rows(gpu, &op.w_down, op.act, op.out, op.x_rot, rows)?;
    }
    s.post_norm_residual(gpu, op.out, op.post_norm)
}

/// Per-layer-input branch:
/// `x = residual + post_norm(proj(gelu(gate(x)) * inputs[row, layer]))`.
/// `inputs` holds `layer_width`-wide slices for `n_layers` layers per row.
pub struct PerLayerInputOp<'a> {
    pub stream: SandwichStream<'a>,
    pub layer: usize,
    pub layer_width: usize,
    pub n_layers: usize,
    pub inputs: &'a GpuTensor,
    pub w_gate: WeightRef<'a>,
    pub w_proj: WeightRef<'a>,
    pub post_norm: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub act: &'a GpuTensor,
    pub out: &'a GpuTensor,
}

/// Q8 projections whose batched GEMM reproduces the single-row GEMV exactly.
fn exact_wide_q8_gemm(dtype: DType, k: usize) -> bool {
    dtype == DType::Q8_0 && k > 0 && k <= 1536 && k.is_multiple_of(32)
}

pub fn execute_per_layer_input(
    gpu: &mut Gpu,
    ctx: &DispatchCtx,
    op: &PerLayerInputOp<'_>,
) -> Result<(), DispatchError> {
    let s = op.stream;
    let (rows, width, hidden) = (s.rows, op.layer_width, s.hidden);
    let packed = op.n_layers * width;
    s.save_residual(gpu)?;
    if rows == 1 {
        let layer_input = op.inputs.sub_offset(op.layer * width, width);
        gemv(gpu, ctx, &op.w_gate, s.x, op.gate)?;
        gpu.gelu_tanh_f32(op.gate, op.act, width).map_err(hip)?;
        gpu.mul_f32(op.act, &layer_input, op.act).map_err(hip)?;
        gemv(gpu, ctx, &op.w_proj, op.act, op.out)?;
    } else {
        let exact = batched_fusion_arch(&gpu.arch) && gpu.flags.gemma4_ple_branch_batched_prefill;
        let wide = |gpu: &mut Gpu, w: &WeightRef, x: &GpuTensor, y: &GpuTensor| {
            GEMM.run_key(
                KernelKey::GemmQ8_0BatchedWideExact,
                ctx,
                gpu,
                &GemmParams {
                    w,
                    x,
                    y,
                    batch_size: rows,
                },
            )
        };
        if exact && exact_wide_q8_gemm(op.w_gate.dtype, op.w_gate.k) {
            wide(gpu, &op.w_gate, s.x, op.gate)?;
        } else {
            for row in 0..rows {
                let x_row = s.x.sub_offset(row * hidden, hidden);
                let gate_row = op.gate.sub_offset(row * width, width);
                gemv(gpu, ctx, &op.w_gate, &x_row, &gate_row)?;
            }
        }
        if batched_fusion_arch(&gpu.arch) && gpu.flags.gemma4_ple_activation_fused_prefill {
            gpu.gemma4_ple_gelu_mul_strided_f32(
                op.gate, op.inputs, op.act, rows, width, packed, op.layer,
            )
            .map_err(hip_dbg)?;
        } else {
            gpu.gelu_tanh_f32(op.gate, op.act, rows * width)
                .map_err(hip)?;
            for row in 0..rows {
                let act_row = op.act.sub_offset(row * width, width);
                let layer_input = op.inputs.sub_offset(row * packed + op.layer * width, width);
                gpu.mul_f32(&act_row, &layer_input, &act_row).map_err(hip)?;
            }
        }
        if exact && exact_wide_q8_gemm(op.w_proj.dtype, op.w_proj.k) {
            wide(gpu, &op.w_proj, op.act, op.out)?;
        } else {
            for row in 0..rows {
                let act_row = op.act.sub_offset(row * width, width);
                let out_row = op.out.sub_offset(row * hidden, hidden);
                gemv(gpu, ctx, &op.w_proj, &act_row, &out_row)?;
            }
        }
    }
    s.norm(gpu, op.out, op.post_norm, s.normed)?;
    s.restore_residual(gpu)?;
    gpu.add_inplace_f32(s.x, s.normed).map_err(hip)
}

/// In-place `x *= factor`.
pub struct ScaleOp<'a> {
    pub x: &'a GpuTensor,
    pub factor: f32,
}

pub fn execute_scale(gpu: &mut Gpu, op: &ScaleOp<'_>) -> Result<(), DispatchError> {
    scale(gpu, op.x, op.factor)
}

/// In-place final-logit soft cap: `x = tanh(x / cap) * cap`.
pub struct SoftcapOp<'a> {
    pub logits: &'a GpuTensor,
    pub n: usize,
    pub cap: f32,
}

pub fn execute_softcap(gpu: &mut Gpu, op: &SoftcapOp<'_>) -> Result<(), DispatchError> {
    gpu.logit_softcap_f32(op.logits, op.n, op.cap).map_err(hip)
}

#[cfg(test)]
mod tests {
    use super::{exact_wide_q8_gemm, supports_batched_projection};
    use rdna_compute::DType;

    #[test]
    fn batched_projection_formats_are_explicit() {
        for dtype in [
            DType::F32,
            DType::Q8_0,
            DType::MQ4G256,
            DType::HFQ4G256,
            DType::MQ6G256,
        ] {
            assert!(supports_batched_projection(dtype));
        }
        for dtype in [
            DType::HFQ4G128,
            DType::HFQ6G256,
            DType::HFQ2G256,
            DType::HFQ3G256,
            DType::MQ3G256,
        ] {
            assert!(!supports_batched_projection(dtype));
        }
    }

    #[test]
    fn exact_wide_q8_gemm_matches_the_wide_gemv_boundary() {
        assert!(exact_wide_q8_gemm(DType::Q8_0, 1536));
        assert!(!exact_wide_q8_gemm(DType::Q8_0, 1535));
        assert!(!exact_wide_q8_gemm(DType::Q8_0, 1537));
        assert!(!exact_wide_q8_gemm(DType::Q8_0, 0));
        assert!(!exact_wide_q8_gemm(DType::F32, 256));
    }
}
