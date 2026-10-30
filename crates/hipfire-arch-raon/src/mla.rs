// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Raon MLA (NoPE, q576/kv512+64) binding onto the generic `LatentAttentionOp`.
//!
//! Math (PLAN §1.6, reference `modeling_raon1_mini.py:381-469`):
//!
//! ```text
//! qlat = RMS576(Wq_a x, eps 1e-6)               q = Wq_b qlat  -> [40, 128+64]
//! [kv512, kpe64] = Wkv_a x ;  c = RMS512(kv512) (the 64 tail channels are NOT normalized)
//! cache row = bf16(c) || kpe64                  ([T, 576] per MLA layer, BF16)
//! [knope128, v128] per head = Wkv_b c ;  key_h = knope_h || kpe (shared over heads)
//! y_h = softmax(q_h . key_h / sqrt(192)) v_h    (no RoPE of any kind, causal j <= prefix+i)
//! y = y * sigmoid(Wg x)  (sigmoid AND product in BF16) ;  o = Wo y
//! ```
//!
//! This file owns only the Raon constants, checkpoint tensor naming, the
//! mirror-admission arithmetic and the stage sequencing. The five
//! projections (`q_a`, `q_b`, `kv_a`, `g`, `o`) are ordinary projections owned
//! by `projection.rs`; the caller runs them and injects `q_b` through a
//! closure because it sits between the latent norm and the attention.

use hipfire_dispatch::pipeline::latent_attention::{
    execute_latent_attention, ExpandedMirror, KvExpansion, LatentAttentionOp, LatentCache,
    LatentGeometry, RowRange,
};
use hipfire_dispatch::types::DispatchError;
use rdna_compute::{Gpu, GpuTensor};

/// Released geometry (config.json `q_lora_rank`, `kv_lora_rank`, `qk_*`, `v_head_dim`).
pub const RAON_MLA_HEADS: usize = 40;
pub const RAON_MLA_Q_LORA: usize = 576;
pub const RAON_MLA_KV_LATENT: usize = 512;
/// The 64 "rope" channels: plain unrotated shared key/query channels.
pub const RAON_MLA_TAIL: usize = 64;
pub const RAON_MLA_NOPE: usize = 128;
pub const RAON_MLA_V: usize = 128;
/// `mla_latent_norm_eps`.
pub const RAON_MLA_EPS: f32 = 1e-6;

/// Zero-based MLA layer indices of the released checkpoint (config lists are
/// one-based: `[4,8,12,16,20,24,25]`).
pub const RAON_MLA_LAYERS: [usize; 7] = [3, 7, 11, 15, 19, 23, 24];

/// Generic geometry carrying the released Raon values.
pub const fn raon_mla_geometry() -> LatentGeometry {
    LatentGeometry {
        n_heads: RAON_MLA_HEADS,
        nope: RAON_MLA_NOPE,
        tail: RAON_MLA_TAIL,
        v_dim: RAON_MLA_V,
        latent: RAON_MLA_KV_LATENT,
    }
}

/// Score scale `(nope+tail)^-0.5` = `192^-0.5` (never `1/sqrt(576)`).
pub fn raon_mla_scale(g: &LatentGeometry) -> f32 {
    g.default_scale()
}

/// Checkpoint tensor names of one MLA layer's attention mixer
/// (`model.safetensors.index.json`). Shapes: `q_a [576,2560]`, `q_b [7680,576]`,
/// `kv_a [576,2560]`, `kv_b [10240,512]`, `g [5120,2560]`, `o [2560,5120]`,
/// norms `[576]` / `[512]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlaTensorNames {
    pub q_a: String,
    pub q_a_norm: String,
    pub q_b: String,
    pub kv_a: String,
    pub kv_a_norm: String,
    pub kv_b: String,
    pub g: String,
    pub o: String,
}

impl MlaTensorNames {
    pub fn for_layer(layer: usize) -> Self {
        let p = |s: &str| format!("model.layers.{layer}.self_attn.{s}.weight");
        Self {
            q_a: p("q_a_proj"),
            q_a_norm: p("q_a_layernorm"),
            q_b: p("q_b_proj"),
            kv_a: p("kv_a_proj_with_mqa"),
            kv_a_norm: p("kv_a_layernorm"),
            kv_b: p("kv_b_proj"),
            g: p("g_proj"),
            o: p("o_proj"),
        }
    }
}

/// Persistent per-layer operands of the latent route (weights borrowed).
pub struct RaonMlaWeights<'a> {
    /// `[576]` q latent norm gamma (F32 or BF16).
    pub q_a_norm: &'a GpuTensor,
    /// `[512]` kv latent norm gamma (F32 or BF16).
    pub kv_a_norm: &'a GpuTensor,
    /// How cached latents expand to K_nope/V (dense kernel or external projector).
    pub kv_b: KvExpansion<'a>,
}

/// Per-layer, per-slot mutable operands (all caller-allocated, bounded).
pub struct RaonMlaState<'a> {
    pub cache: LatentCache<'a>,
    pub mirror: ExpandedMirror<'a>,
}

/// Scratch planes for one call of `rows` rows (F32, BF16-valued).
pub struct RaonMlaScratch<'a> {
    /// `[rows, 576]` normalized q latent (input of the `q_b` projection).
    pub qlat: &'a GpuTensor,
    /// `[rows, 40*192]` output of the `q_b` projection.
    pub q: &'a GpuTensor,
    /// `[rows, 40*128]` attention output (gated in place).
    pub y: &'a GpuTensor,
}

/// Bytes of the BF16 latent cache for `rows` rows of one layer/slot.
pub const fn latent_cache_bytes(rows: usize) -> usize {
    rows * (RAON_MLA_KV_LATENT + RAON_MLA_TAIL) * 2
}

/// Bytes per token row of the expanded BF16 K_nope+V mirror (all heads).
pub const fn mirror_bytes_per_row() -> usize {
    RAON_MLA_HEADS * (RAON_MLA_NOPE + RAON_MLA_V) * 2
}

/// Rows of expanded mirror to admit for a layer/slot: `needed_rows` when it
/// fits `budget_bytes`, otherwise `None` (route not admitted; the caller keeps
/// latent-only storage and must use a tiled/absorbed route). Never sized from
/// the model's maximum sequence unconditionally.
pub fn admitted_mirror_rows(needed_rows: usize, budget_bytes: usize) -> Option<usize> {
    let need = needed_rows.checked_mul(mirror_bytes_per_row())?;
    (need <= budget_bytes).then_some(needed_rows)
}

/// Run the Raon MLA core for `rows` rows starting at the device position held
/// in `state.cache.pos`:
///
/// 1. `qlat = RMS576(q_a_out)`; 2. `project_q_b(gpu, qlat, q)`;
/// 3. split `kv_a_out` into the latent cache (norm over 512 only);
/// 4. decompress the new rows into the mirror; 5. serial attention; 6. BF16
///    sigmoid gate with `g_out`.
///
/// On return `scratch.y` holds the gated `[rows, 5120]` context, ready for the
/// `o` projection. The position scalar is not advanced here (the layer
/// schedule owner advances it once per call after all MLA layers consumed it).
pub fn raon_mla_core(
    gpu: &mut Gpu,
    w: &RaonMlaWeights<'_>,
    st: &RaonMlaState<'_>,
    scratch: &RaonMlaScratch<'_>,
    q_a_out: &GpuTensor,
    kv_a_out: &GpuTensor,
    g_out: &GpuTensor,
    rows: usize,
    project_q_b: &mut dyn FnMut(&mut Gpu, &GpuTensor, &GpuTensor) -> Result<(), DispatchError>,
) -> Result<(), DispatchError> {
    let geometry = raon_mla_geometry();
    execute_latent_attention(
        gpu,
        &LatentAttentionOp::NormRows {
            x: q_a_out,
            gamma: w.q_a_norm,
            out: scratch.qlat,
            rows,
            width: RAON_MLA_Q_LORA,
            eps: RAON_MLA_EPS,
        },
    )?;
    project_q_b(gpu, scratch.qlat, scratch.q)?;
    execute_latent_attention(
        gpu,
        &LatentAttentionOp::SplitKv {
            geometry,
            cache: st.cache,
            kv_a: kv_a_out,
            gamma: w.kv_a_norm,
            rows,
            eps: RAON_MLA_EPS,
        },
    )?;
    match w.kv_b {
        KvExpansion::Dense { .. } => {
            execute_latent_attention(
                gpu,
                &LatentAttentionOp::Decompress {
                    geometry,
                    cache: st.cache,
                    mirror: st.mirror,
                    expansion: w.kv_b,
                    range: RowRange::NewRows { count: rows },
                },
            )?;
        }
        KvExpansion::External { .. } => {
            return Err(DispatchError::Hip(
                "raon_mla_core: external kv_b projector needs the caller to run \
                 GatherLatent -> projector -> Decompress(External) itself"
                    .into(),
            ));
        }
    }
    execute_latent_attention(
        gpu,
        &LatentAttentionOp::Attention {
            geometry,
            cache: st.cache,
            mirror: st.mirror,
            q: scratch.q,
            y: scratch.y,
            rows,
            scale: raon_mla_scale(&geometry),
        },
    )?;
    execute_latent_attention(
        gpu,
        &LatentAttentionOp::Gate {
            y: scratch.y,
            g: g_out,
            out: scratch.y,
            elements: rows * RAON_MLA_HEADS * RAON_MLA_V,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_match_checkpoint_index() {
        let n = MlaTensorNames::for_layer(3);
        assert_eq!(n.q_a, "model.layers.3.self_attn.q_a_proj.weight");
        assert_eq!(n.kv_a, "model.layers.3.self_attn.kv_a_proj_with_mqa.weight");
        assert_eq!(n.kv_a_norm, "model.layers.3.self_attn.kv_a_layernorm.weight");
        assert_eq!(n.kv_b, "model.layers.3.self_attn.kv_b_proj.weight");
        assert_eq!(n.g, "model.layers.3.self_attn.g_proj.weight");
        assert_eq!(n.o, "model.layers.3.self_attn.o_proj.weight");
    }

    #[test]
    fn geometry_is_192_128_with_192_scale() {
        let g = raon_mla_geometry();
        assert_eq!(g.qk_dim(), 192);
        assert_eq!(g.v_dim, 128);
        assert_eq!(g.row_width(), 576);
        assert_eq!(g.expanded_width(), 10240);
        let s = raon_mla_scale(&g);
        assert!((s - 192f32.powf(-0.5)).abs() < 1e-9);
        assert!((s - 576f32.powf(-0.5)).abs() > 1e-3);
    }

    #[test]
    fn mirror_admission_is_bounded() {
        assert_eq!(mirror_bytes_per_row(), 40 * 256 * 2);
        assert_eq!(latent_cache_bytes(1), 1152);
        assert_eq!(admitted_mirror_rows(1024, 1024 * mirror_bytes_per_row()), Some(1024));
        assert_eq!(admitted_mirror_rows(1025, 1024 * mirror_bytes_per_row()), None);
        assert_eq!(admitted_mirror_rows(usize::MAX, usize::MAX), None);
    }
}
