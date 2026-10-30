// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `LatentAttentionOp` / `execute_latent_attention`: the generic latent
//! attention pipeline (split latent/tail, latent row cache, bounded expanded
//! K/V mirrors, unequal QK/V attention with causal offsets, BF16 output gate).
//!
//! One call executes one stage; the caller sequences the stages around its own
//! projections (the query projection and any quantized expansion projector are
//! ordinary projections and stay outside this op):
//!
//! ```text
//! NormRows(q_a out) -> [caller: q = q_b * qlat]
//! SplitKv(kv_a out) -> latent cache row pos+i
//! Decompress(NewRows) | GatherLatent -> [caller projector] -> Decompress(External)
//! Attention(q, mirror) -> Gate(y, g)  -> [caller: o_proj]
//! ```
//!
//! The op borrows persistent buffers and owns nothing. Positions/lengths are a
//! device-resident i32 scalar (rows already cached) that no stage mutates, so
//! every stage is graph-capture safe and performs no host readback/allocation.
//! Decode is the `rows == 1` instance of the same route.

use rdna_compute::{Gpu, GpuTensor};

use crate::types::DispatchError;

/// Independent attention geometry. Nothing here is checkpoint-specific.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LatentGeometry {
    pub n_heads: usize,
    /// Per-head key channels expanded from the latent.
    pub nope: usize,
    /// Cached channels shared by all heads (plain extra QK channels).
    pub tail: usize,
    /// Per-head value channels (independent of `nope + tail`).
    pub v_dim: usize,
    /// Normalized cached latent channels.
    pub latent: usize,
}

impl LatentGeometry {
    /// Query/key width per head.
    pub fn qk_dim(&self) -> usize {
        self.nope + self.tail
    }
    /// Cache row width (`latent + tail`).
    pub fn row_width(&self) -> usize {
        self.latent + self.tail
    }
    /// Expansion projection output width (`n_heads * (nope + v_dim)`).
    pub fn expanded_width(&self) -> usize {
        self.n_heads * (self.nope + self.v_dim)
    }
    /// Default score scale `(nope + tail)^-0.5`.
    pub fn default_scale(&self) -> f32 {
        (self.qk_dim() as f32).powf(-0.5)
    }
}

/// BF16 latent row cache for one layer/slot, row-major `[capacity_rows, latent+tail]`.
#[derive(Clone, Copy)]
pub struct LatentCache<'a> {
    pub rows: &'a GpuTensor,
    pub capacity_rows: usize,
    /// Device scalar: number of rows already cached before this call.
    pub pos: &'a GpuTensor,
    /// Optional host-known copy of `*pos` for eager callers; enables a
    /// capacity refusal before launch. Capture callers pass `None` and own the
    /// length accounting.
    pub host_pos: Option<usize>,
}

/// Bounded expanded BF16 mirrors, head-major with row stride `capacity_rows`
/// (the admitted mirror rows, independent of the latent cache's mapped
/// capacity): `knope[h][t][nope]`, `vmir[h][t][v_dim]`, valid for `t < pos`.
/// Sized by the caller; never an unconditional max-sequence allocation.
#[derive(Clone, Copy)]
pub struct ExpandedMirror<'a> {
    pub knope: &'a GpuTensor,
    pub vmir: &'a GpuTensor,
    pub capacity_rows: usize,
}

/// Which cache rows a stage covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowRange {
    /// Rows `*pos + 0 .. *pos + count` (the rows written this call).
    NewRows { count: usize },
    /// Absolute rows `start .. start + count`.
    Absolute { start: usize, count: usize },
}

/// How cached latent rows become expanded K_nope/V rows.
#[derive(Clone, Copy)]
pub enum KvExpansion<'a> {
    /// Built-in deterministic dense projector. `weight` is
    /// `[n_heads*(nope+v_dim), latent]` BF16 or F32 (row `h*(nope+v)+e`).
    Dense { weight: &'a GpuTensor },
    /// The caller already ran its own projector on a `GatherLatent` plane;
    /// `projected` is `[count, n_heads*(nope+v_dim)]` F32. This stage only
    /// rounds to BF16 and scatters into the mirrors.
    External { projected: &'a GpuTensor },
}

/// One stage of the latent attention pipeline.
pub enum LatentAttentionOp<'a> {
    /// RMS over the full row, BF16-rounded F32 result (`out` may alias `x`).
    NormRows {
        x: &'a GpuTensor,
        gamma: &'a GpuTensor,
        out: &'a GpuTensor,
        rows: usize,
        width: usize,
        eps: f32,
    },
    /// Normalize the latent channels only, append the raw tail, write BF16 row
    /// `*pos + i` of the cache.
    SplitKv {
        geometry: LatentGeometry,
        cache: LatentCache<'a>,
        kv_a: &'a GpuTensor,
        gamma: &'a GpuTensor,
        rows: usize,
        eps: f32,
    },
    /// Cache latent rows → F32 `[count, latent]` plane for an external projector.
    GatherLatent {
        geometry: LatentGeometry,
        cache: LatentCache<'a>,
        range: RowRange,
        out: &'a GpuTensor,
    },
    /// Cache latent rows → expanded BF16 K_nope/V mirror rows.
    Decompress {
        geometry: LatentGeometry,
        cache: LatentCache<'a>,
        mirror: ExpandedMirror<'a>,
        expansion: KvExpansion<'a>,
        range: RowRange,
    },
    /// Serial reference attention (query row `i` sees keys `j <= *pos + i`).
    Attention {
        geometry: LatentGeometry,
        cache: LatentCache<'a>,
        mirror: ExpandedMirror<'a>,
        /// `[rows, n_heads, nope+tail]` F32 (BF16-valued).
        q: &'a GpuTensor,
        /// `[rows, n_heads, v_dim]` F32 (BF16-valued).
        y: &'a GpuTensor,
        rows: usize,
        scale: f32,
    },
    /// `out = bf16(y * bf16(sigmoid(g)))` (`out` may alias `y`).
    Gate {
        y: &'a GpuTensor,
        g: &'a GpuTensor,
        out: &'a GpuTensor,
        elements: usize,
    },
}

fn refuse(msg: String) -> DispatchError {
    DispatchError::Hip(format!("latent_attention: {msg}"))
}

fn hip(e: hip_bridge::HipError) -> DispatchError {
    DispatchError::Hip(e.to_string())
}

fn check_geometry(g: &LatentGeometry) -> Result<(), DispatchError> {
    if g.n_heads == 0 || g.latent == 0 || g.v_dim == 0 || g.nope + g.tail == 0 {
        return Err(refuse(format!("degenerate geometry {g:?}")));
    }
    Ok(())
}

/// Refuse when the host-known live length exceeds the mirror's admitted rows
/// (device-side kernels additionally clamp, never read/write past the stride).
fn check_mirror(m: &ExpandedMirror<'_>, end: Option<usize>) -> Result<(), DispatchError> {
    if m.capacity_rows == 0 {
        return Err(refuse("expanded mirror capacity is zero".into()));
    }
    if let Some(e) = end {
        if e > m.capacity_rows {
            return Err(refuse(format!(
                "expanded mirror holds {} rows, live length needs {e}",
                m.capacity_rows
            )));
        }
    }
    Ok(())
}

fn check_cache(c: &LatentCache<'_>, rows: usize) -> Result<(), DispatchError> {
    if c.capacity_rows == 0 {
        return Err(refuse("latent cache capacity is zero".into()));
    }
    if let Some(p) = c.host_pos {
        if p.checked_add(rows).map_or(true, |e| e > c.capacity_rows) {
            return Err(refuse(format!(
                "latent cache overflow: pos {p} + {rows} rows > capacity {}",
                c.capacity_rows
            )));
        }
    }
    Ok(())
}

/// Resolve a [`RowRange`] to `(use_device_base, row0, count)`.
fn resolve_range(
    c: &LatentCache<'_>,
    range: RowRange,
) -> Result<(bool, usize, usize), DispatchError> {
    match range {
        RowRange::NewRows { count } => {
            check_cache(c, count)?;
            Ok((true, 0, count))
        }
        RowRange::Absolute { start, count } => {
            if start.checked_add(count).map_or(true, |e| e > c.capacity_rows) {
                return Err(refuse(format!(
                    "row range {start}+{count} exceeds cache capacity {}",
                    c.capacity_rows
                )));
            }
            Ok((false, start, count))
        }
    }
}

/// Execute one latent-attention stage.
pub fn execute_latent_attention(
    gpu: &mut Gpu,
    op: &LatentAttentionOp<'_>,
) -> Result<(), DispatchError> {
    match op {
        LatentAttentionOp::NormRows { x, gamma, out, rows, width, eps } => gpu
            .latent_attention_rms_rows(x, gamma, out, *rows, *width, *eps)
            .map_err(hip),
        LatentAttentionOp::SplitKv { geometry, cache, kv_a, gamma, rows, eps } => {
            check_geometry(geometry)?;
            check_cache(cache, *rows)?;
            gpu.latent_attention_split_kv(
                kv_a,
                gamma,
                cache.rows,
                cache.pos,
                *rows,
                geometry.latent,
                geometry.tail,
                *eps,
                cache.capacity_rows,
            )
            .map_err(hip)
        }
        LatentAttentionOp::GatherLatent { geometry, cache, range, out } => {
            check_geometry(geometry)?;
            let (dev_base, row0, count) = resolve_range(cache, *range)?;
            gpu.latent_attention_gather_latent(
                cache.rows,
                dev_base.then_some(cache.pos),
                out,
                row0,
                count,
                geometry.latent,
                geometry.tail,
                cache.capacity_rows,
            )
            .map_err(hip)
        }
        LatentAttentionOp::Decompress { geometry, cache, mirror, expansion, range } => {
            check_geometry(geometry)?;
            let mirror_end = match range {
                RowRange::Absolute { start, count } => start.checked_add(*count),
                RowRange::NewRows { count } => cache.host_pos.map(|p| p + *count),
            };
            check_mirror(mirror, mirror_end)?;
            let (dev_base, row0, count) = resolve_range(cache, *range)?;
            let base = dev_base.then_some(cache.pos);
            match expansion {
                KvExpansion::Dense { weight } => gpu
                    .latent_attention_decompress_dense(
                        cache.rows,
                        base,
                        weight,
                        mirror.knope,
                        mirror.vmir,
                        row0,
                        count,
                        geometry.n_heads,
                        geometry.nope,
                        geometry.v_dim,
                        geometry.latent,
                        geometry.tail,
                        cache.capacity_rows,
                        mirror.capacity_rows,
                    )
                    .map_err(hip),
                KvExpansion::External { projected } => gpu
                    .latent_attention_scatter_expanded(
                        projected,
                        mirror.knope,
                        mirror.vmir,
                        base,
                        row0,
                        count,
                        geometry.n_heads,
                        geometry.nope,
                        geometry.v_dim,
                        mirror.capacity_rows,
                    )
                    .map_err(hip),
            }
        }
        LatentAttentionOp::Attention { geometry, cache, mirror, q, y, rows, scale } => {
            check_geometry(geometry)?;
            check_cache(cache, *rows)?;
            check_mirror(mirror, cache.host_pos.map(|p| p + *rows))?;
            gpu.latent_attention_serial(
                q,
                cache.rows,
                mirror.knope,
                mirror.vmir,
                y,
                cache.pos,
                *rows,
                geometry.n_heads,
                geometry.nope,
                geometry.tail,
                geometry.v_dim,
                geometry.latent,
                cache.capacity_rows,
                mirror.capacity_rows,
                *scale,
            )
            .map_err(hip)
        }
        LatentAttentionOp::Gate { y, g, out, elements } => gpu
            .latent_attention_gate(y, g, out, *elements)
            .map_err(hip),
    }
}
