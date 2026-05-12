//! Imatrix GGUF parser: load llama.cpp importance-matrix files and remap
//! tensor names to the safetensors convention hipfire uses internally.
//!
//! Imatrix files are plain GGUF files with `general.type = "imatrix"`.
//! Each quantised tensor contributes two auxiliary tensors:
//!   `<base_name>.in_sum2`  — sum of squared activations per column (F32)
//!   `<base_name>.counts`   — number of rows accumulated             (F32)
//!
//! After normalisation `importance[j] = in_sum2[j] / counts[row_of_j]`
//! gives the mean squared activation for column j, which measures how
//! much each weight column is "used" during calibration.  High-importance
//! columns should be quantised more carefully.
//!
//! MoE models store expert-stacked tensors (`ffn_gate_exps`, `ffn_up_exps`,
//! `ffn_down_exps`).  `imatrix_expand_moe` slices these into per-expert
//! entries so downstream code can look up importance by the exact
//! safetensors tensor name.

use std::collections::HashMap;
use std::io;
use std::path::Path;

use crate::gguf_input::{GgufFile, MetaValue};

// ─── Public types ────────────────────────────────────────────────────────────

/// Per-tensor importance vectors loaded from a llama.cpp imatrix GGUF file.
///
/// Keys use the **safetensors / HuggingFace** name convention so callers can
/// look up importance directly by the name used during quantisation.
pub struct ImatrixData {
    /// Per-tensor importance vectors, keyed by safetensors-style name.
    /// Each `Vec<f32>` has one entry per weight column (the "input" dimension).
    pub tensors: HashMap<String, Vec<f32>>,
    /// Calibration dataset names recorded in the imatrix file.
    pub datasets: Vec<String>,
    /// Number of calibration chunks accumulated.
    pub chunk_count: u32,
    /// Tokens per calibration chunk.
    pub chunk_size: u32,
}

// ─── Public entry point ──────────────────────────────────────────────────────

/// Load a llama.cpp imatrix GGUF file and return normalised importance vectors.
///
/// The function:
/// 1. Opens the file via [`GgufFile::open`].
/// 2. Verifies `general.type == "imatrix"`.
/// 3. Reads `imatrix.chunk_count`, `imatrix.chunk_size`, `imatrix.datasets`.
/// 4. For every `*.in_sum2` tensor, finds the matching `*.counts` tensor,
///    normalises, and remaps the name to the safetensors convention.
/// 5. Expands MoE stacked tensors (`*_exps`) into per-expert entries.
/// 6. Skips tensors with NaN/Inf values (with a warning).
pub fn load_imatrix(path: &Path) -> io::Result<ImatrixData> {
    let gguf = GgufFile::open(path)?;

    // ── Verify file type ──────────────────────────────────────────────────
    match gguf.meta_str("general.type") {
        Some(t) if t == "imatrix" => {}
        Some(other) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected general.type=imatrix, got {other:?}"),
            ));
        }
        None => {
            // Some older imatrix files omit the key — tolerate it.
            eprintln!("warning: imatrix file missing general.type; assuming imatrix");
        }
    }

    // ── Metadata ──────────────────────────────────────────────────────────
    let chunk_count = gguf.meta_u32("imatrix.chunk_count").unwrap_or(0);
    let chunk_size = gguf.meta_u32("imatrix.chunk_size").unwrap_or(0);

    let datasets: Vec<String> = match gguf.meta("imatrix.datasets") {
        Some(MetaValue::Array(arr)) => arr
            .iter()
            .filter_map(|v| {
                if let MetaValue::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect(),
        Some(MetaValue::String(s)) => vec![s.clone()],
        _ => vec![],
    };

    // ── Build name → TensorInfo index maps ───────────────────────────────
    // We need O(1) lookup for matching .in_sum2 ↔ .counts pairs.
    let tensor_index: HashMap<&str, usize> = gguf
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.as_str(), i))
        .collect();

    // ── Collect base names from .in_sum2 tensors ─────────────────────────
    let base_names: Vec<String> = gguf
        .tensors
        .iter()
        .filter_map(|t| t.name.strip_suffix(".in_sum2").map(|b| b.to_owned()))
        .collect();

    let mut result: HashMap<String, Vec<f32>> = HashMap::new();
    let mut skipped = 0usize;
    // Accumulate gate/up halves for post-loop fusion into gate_up_proj.
    let mut gate_up_halves: Vec<(String, &'static str, Vec<Vec<f32>>)> = Vec::new();

    for base in &base_names {
        let sum2_key = format!("{base}.in_sum2");
        let counts_key = format!("{base}.counts");

        let sum2_idx = match tensor_index.get(sum2_key.as_str()) {
            Some(&i) => i,
            None => {
                skipped += 1;
                continue;
            }
        };
        let counts_idx = match tensor_index.get(counts_key.as_str()) {
            Some(&i) => i,
            None => {
                eprintln!("warning: imatrix tensor {base} has .in_sum2 but no .counts — skipping");
                skipped += 1;
                continue;
            }
        };

        let sum2_info = &gguf.tensors[sum2_idx];
        let counts_info = &gguf.tensors[counts_idx];

        // Number of columns (input features) in the weight matrix.
        let n_cols = sum2_info.numel();
        // counts may be scalar (1 value shared across all rows) or a vector
        // with one entry per row. For column-importance we need counts per
        // row; typically it's a single scalar or a 1-D vector of length
        // `n_mats` (one count per expert / matrix).
        let n_counts = counts_info.numel();

        let sum2_bytes = gguf.tensor_data(sum2_info);
        let counts_bytes = gguf.tensor_data(counts_info);

        let sum2 = bytes_to_f32_vec(sum2_bytes);
        let counts = bytes_to_f32_vec(counts_bytes);

        if sum2.len() != n_cols {
            eprintln!(
                "warning: imatrix {base} sum2 length mismatch ({} vs {n_cols}) — skipping",
                sum2.len()
            );
            skipped += 1;
            continue;
        }

        // counts is either:
        //   • scalar (1): single count for the whole tensor
        //   • vector (n_mats): one count per stacked matrix (MoE experts)
        //   • vector (n_cols): one count per column (rare but valid)
        let n_mats = if n_counts == 1 {
            1usize
        } else {
            // Treat as n_mats; each expert block has n_cols/n_mats columns.
            n_counts
        };

        // Normalise: importance[j] = sum2[j] / counts[expert_of_j]
        let cols_per_mat = if n_mats > 1 { n_cols / n_mats } else { n_cols };
        let mut importance = Vec::with_capacity(n_cols);
        let mut has_bad = false;

        for j in 0..n_cols {
            let c_idx = if n_mats > 1 { j / cols_per_mat } else { 0 };
            let c = counts[c_idx.min(counts.len() - 1)];
            let v = if c == 0.0 { 0.0 } else { sum2[j] / c };
            if v.is_nan() || v.is_infinite() {
                has_bad = true;
                break;
            }
            importance.push(v);
        }

        if has_bad {
            eprintln!("warning: imatrix {base} contains NaN/Inf after normalisation — skipping");
            skipped += 1;
            continue;
        }

        // ── Name remapping ────────────────────────────────────────────────
        if n_mats > 1 {
            // MoE stacked tensor — expand into per-expert entries.
            match imatrix_expand_moe(base, &importance, n_mats) {
                Some(MoeExpansion::Direct(entries)) => {
                    for (name, vec) in entries {
                        result.insert(name, vec);
                    }
                }
                Some(MoeExpansion::GateUpHalf { layer_idx, half, experts }) => {
                    // Stash gate/up halves for post-loop fusion.
                    gate_up_halves.push((layer_idx, half, experts));
                }
                None => {
                    eprintln!("warning: imatrix MoE tensor {base} not remapped (unknown projection) — skipping");
                    skipped += 1;
                }
            }
        } else {
            match imatrix_remap_name(base) {
                Some(mapped) => {
                    result.insert(mapped, importance);
                }
                None => {
                    skipped += 1;
                }
            }
        }
    }

    // ── Fuse gate + up halves into gate_up_proj ──────────────────────────
    // llama.cpp stores ffn_gate_exps (K cols) and ffn_up_exps (K cols)
    // separately, but hipfire fuses them into gate_up_proj (2K cols).
    // Concatenate: gate columns [0..K) ++ up columns [K..2K).
    {
        // Group by layer_idx
        let mut by_layer: HashMap<String, (Option<Vec<Vec<f32>>>, Option<Vec<Vec<f32>>>)> = HashMap::new();
        for (layer_idx, half, experts) in gate_up_halves {
            let entry = by_layer.entry(layer_idx).or_insert((None, None));
            match half {
                "gate" => entry.0 = Some(experts),
                "up" => entry.1 = Some(experts),
                _ => {}
            }
        }

        for (layer_idx, (gate_opt, up_opt)) in &by_layer {
            match (gate_opt, up_opt) {
                (Some(gate), Some(up)) => {
                    // Both halves present — fuse per expert
                    let n_experts = gate.len().min(up.len());
                    for x in 0..n_experts {
                        let mut fused = gate[x].clone();
                        fused.extend_from_slice(&up[x]);
                        let name = format!(
                            "model.layers.{layer_idx}.mlp.experts.{x}.gate_up_proj.weight"
                        );
                        result.insert(name, fused);
                    }
                }
                (Some(half), None) | (None, Some(half)) => {
                    // Only one half — use it as-is (partial coverage is
                    // better than none; the column count mismatch will
                    // cause the per-tensor validation to skip it anyway
                    // if the model expects 2K columns).
                    let n_experts = half.len();
                    for x in 0..n_experts {
                        let name = format!(
                            "model.layers.{layer_idx}.mlp.experts.{x}.gate_up_proj.weight"
                        );
                        result.insert(name, half[x].clone());
                    }
                    eprintln!("warning: imatrix layer {layer_idx} has only one half of gate_up — partial coverage");
                }
                (None, None) => {}
            }
        }
    }

    if skipped > 0 {
        eprintln!("imatrix: skipped {skipped} tensors (unmapped, invalid, or incomplete)");
    }

    eprintln!(
        "imatrix: loaded {} tensors from {} (chunk_count={chunk_count}, chunk_size={chunk_size})",
        result.len(),
        path.display(),
    );

    Ok(ImatrixData { tensors: result, datasets, chunk_count, chunk_size })
}

// ─── Name remapping ──────────────────────────────────────────────────────────

/// Remap a GGUF imatrix base name (no `.in_sum2` suffix) to the safetensors
/// tensor name convention used inside hipfire.
///
/// Delegates to [`super::gguf_to_safetensors_name`] for dense tensors.
/// Returns `None` if the name is not recognised.
pub fn imatrix_remap_name(base: &str) -> Option<String> {
    // imatrix base names may or may not already have ".weight" suffix.
    let gguf_name = if base.ends_with(".weight") {
        base.to_string()
    } else {
        format!("{base}.weight")
    };

    // Handle Qwen3.5 DeltaNet/Mamba tensor names BEFORE the standard mapper,
    // because gguf_to_safetensors_name's catch-all "other" arm would map
    // e.g. "attn_qkv" → "model.layers.N.attn_qkv.weight" (wrong name).
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        if let Some(dot) = rest.find('.') {
            let layer_idx = &rest[..dot];
            if let Some(slot) = rest[dot + 1..].strip_suffix(".weight") {
                let translated = match slot {
                    // DeltaNet/Mamba linear attention
                    "attn_qkv" => Some("linear_attn.in_proj_qkv"),
                    "attn_gate" => Some("linear_attn.in_proj_z"),
                    "ssm_alpha" => Some("linear_attn.in_proj_a"),
                    "ssm_beta" => Some("linear_attn.in_proj_b"),
                    "ssm_out" => Some("linear_attn.out_proj"),
                    "ssm_conv1d" => Some("linear_attn.conv1d"),
                    // MoE router + shared expert
                    "ffn_gate_inp" => Some("mlp.gate"),
                    "ffn_gate_inp_shexp" => Some("mlp.shared_expert_gate"),
                    "ffn_gate_shexp" => Some("mlp.shared_expert.gate_proj"),
                    "ffn_up_shexp" => Some("mlp.shared_expert.up_proj"),
                    "ffn_down_shexp" => Some("mlp.shared_expert.down_proj"),
                    _ => None,
                };
                if let Some(t) = translated {
                    return Some(format!("model.layers.{layer_idx}.{t}.weight"));
                }
            }
        }
    }

    // Standard mapping for dense attention + FFN tensors
    super::gguf_to_safetensors_name(&gguf_name)
}

// ─── MoE expansion ───────────────────────────────────────────────────────────

/// Expand a MoE stacked imatrix tensor into per-expert importance entries.
///
/// llama.cpp stores MoE expert weights as a single tensor stacking all
/// `n_experts` matrices along the first dimension.  The imatrix follows
/// the same layout: a flat vector of `n_experts * cols_per_expert` values.
///
/// The function:
/// 1. Extracts the layer index from `blk.{N}.` prefix.
/// 2. Maps the GGUF projection name to the HF name:
///    - `ffn_gate_exps` / `ffn_up_exps` → `gate_up_proj`  (fused in HF layout)
///    - `ffn_down_exps`                  → `down_proj`
/// 3. Slices `importance` into `n_experts` chunks and returns them with names
///    `model.layers.{N}.mlp.experts.{X}.{proj}.weight`.
///
/// Returns an empty Vec if the tensor name pattern is not recognised.
/// Result of expanding a MoE expert imatrix tensor.
pub enum MoeExpansion {
    /// Direct per-expert entries (e.g. ffn_down_exps → down_proj per expert).
    Direct(Vec<(String, Vec<f32>)>),
    /// Half of a fused gate_up_proj — needs to be merged with the other half.
    /// (layer_idx, "gate"|"up", per-expert importance slices)
    GateUpHalf {
        layer_idx: String,
        half: &'static str, // "gate" or "up"
        experts: Vec<Vec<f32>>,
    },
}

pub fn imatrix_expand_moe(
    base: &str,
    importance: &[f32],
    n_mats: usize,
) -> Option<MoeExpansion> {
    // Parse "blk.{N}.{proj}" prefix.
    let rest = base.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let layer_idx = &rest[..dot];
    let proj_key = &rest[dot + 1..]; // e.g. "ffn_gate_exps"

    let total = importance.len();
    if total % n_mats != 0 {
        eprintln!(
            "warning: imatrix MoE {base} length {total} not divisible by n_mats={n_mats} — skipping"
        );
        return None;
    }
    let cols_per_expert = total / n_mats;

    let expert_slices: Vec<Vec<f32>> = (0..n_mats)
        .map(|x| importance[x * cols_per_expert..(x + 1) * cols_per_expert].to_vec())
        .collect();

    match proj_key {
        "ffn_gate_exps" => Some(MoeExpansion::GateUpHalf {
            layer_idx: layer_idx.to_string(),
            half: "gate",
            experts: expert_slices,
        }),
        "ffn_up_exps" => Some(MoeExpansion::GateUpHalf {
            layer_idx: layer_idx.to_string(),
            half: "up",
            experts: expert_slices,
        }),
        "ffn_down_exps" => {
            let entries = expert_slices.into_iter().enumerate()
                .map(|(x, imp)| {
                    let name = format!(
                        "model.layers.{layer_idx}.mlp.experts.{x}.down_proj.weight"
                    );
                    (name, imp)
                })
                .collect();
            Some(MoeExpansion::Direct(entries))
        }
        _ => None,
    }
}

impl ImatrixData {
    /// Look up importance data for a tensor, trying common name variants.
    ///
    /// Qwen3.5 safetensors use `model.language_model.layers.N.` while the
    /// imatrix (from GGUF convention) maps to `model.layers.N.`.  This
    /// helper tries the exact name first, then strips the
    /// `model.language_model.` or `language_model.model.` prefix.
    pub fn get(&self, name: &str) -> Option<&Vec<f32>> {
        if let Some(v) = self.tensors.get(name) {
            return Some(v);
        }
        // Strip "model.language_model." → "model."
        if let Some(rest) = name.strip_prefix("model.language_model.") {
            let short = format!("model.{rest}");
            if let Some(v) = self.tensors.get(&short) {
                return Some(v);
            }
        }
        // Strip "language_model.model." → "model."
        if let Some(rest) = name.strip_prefix("language_model.model.") {
            let short = format!("model.{rest}");
            if let Some(v) = self.tensors.get(&short) {
                return Some(v);
            }
        }
        None
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Reinterpret a byte slice as a little-endian f32 slice.
fn bytes_to_f32_vec(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let arr: [u8; 4] = bytes[i * 4..i * 4 + 4].try_into().unwrap();
        out.push(f32::from_le_bytes(arr));
    }
    out
}

// ─── FP4 imatrix scale strategy ──────────────────────────────────────────────

use crate::strategy::{ScaleStrategy, MinMaxScale};
use crate::formats::hfp4::{E2M1_LUT, e2m1_round};

/// FP4 scale strategy using imatrix importance data.
/// Delegates `solve_affine` to `MinMaxScale` (no benefit for affine formats).
/// Overrides FP4 row scale with importance-weighted outlier clipping.
/// Block exponent stays at default (the row scale optimization is the primary
/// mechanism producing the -19.2% KLD improvement; block exponent tweaking
/// was marginal and requires block-offset information the trait does not carry).
pub struct ImatrixFp4Scale<'a> {
    pub importance: &'a [f32],
}

impl ScaleStrategy for ImatrixFp4Scale<'_> {
    fn solve_affine(&self, group: &[f32], n_levels: u32) -> (f32, f32) {
        MinMaxScale.solve_affine(group, n_levels)
    }

    fn solve_fp4_row_scale(&self, row: &[f32]) -> f32 {
        hfp4_optimal_row_scale(row, self.importance)
    }

    fn solve_fp4_block_e(&self, block: &[f32], inv_row_scale: f32, default_e: u8, block_idx: usize) -> u8 {
        let start = block_idx * 32;
        let end = (start + 32).min(self.importance.len());
        if start >= self.importance.len() {
            return default_e;
        }
        let im = &self.importance[start..end];
        hfp4_optimal_block_e(block, im, inv_row_scale, default_e)
    }
}

/// Find optimal row_scale_a by trying a few candidate scales that clip
/// low-importance outliers. Uses an importance-weighted percentile approach
/// instead of brute-force error computation for speed (O(K log K) sort
/// dominates, not O(K x candidates x K) error recomputation).
fn hfp4_optimal_row_scale(row: &[f32], im: &[f32]) -> f32 {
    // Collect (abs_magnitude, importance) pairs, sort by magnitude descending
    let mut mag_imp: Vec<(f32, f32)> = row.iter().zip(im.iter())
        .map(|(&x, &w)| (x.abs(), w))
        .collect();
    mag_imp.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let max_abs = mag_imp[0].0;
    if max_abs <= 0.0 { return 1.0; }

    // Walk down sorted magnitudes. Stop when cumulative clipped importance
    // exceeds 5% of total, or we've clipped 0.5% of elements.
    let total_imp: f32 = mag_imp.iter().map(|&(_, w)| w).sum();
    let imp_budget = 0.05 * total_imp;
    let max_clip = (row.len() / 200).max(1).min(8);

    let mut cum_imp = 0.0f32;
    let mut best_idx = 0usize; // 0 = no clipping

    for i in 0..max_clip.min(mag_imp.len()) {
        cum_imp += mag_imp[i].1;
        if cum_imp > imp_budget { break; }
        // Accept clipping element i if its importance is below the per-element mean
        if mag_imp[i].1 < total_imp / row.len() as f32 {
            best_idx = i + 1;
        }
    }

    // Scale from the first unclipped element
    let scale_max = if best_idx < mag_imp.len() && mag_imp[best_idx].0 > 0.0 {
        mag_imp[best_idx].0
    } else {
        max_abs
    };

    scale_max / 6.0
}

/// Try the default block exponent and one step tighter. Keep whichever
/// gives lower importance-weighted error within the block.
#[allow(dead_code)] // retained for future block-level optimization
fn hfp4_optimal_block_e(block: &[f32], im: &[f32], inv_row_scale: f32, default_e: u8) -> u8 {
    if default_e == 0 { return 0; }

    let err_default = hfp4_block_weighted_error(block, im, inv_row_scale, default_e);

    // Try one step tighter (lower exponent = finer granularity, but clips more)
    let tighter_e = default_e - 1;
    let err_tighter = hfp4_block_weighted_error(block, im, inv_row_scale, tighter_e);

    if err_tighter < err_default { tighter_e } else { default_e }
}

#[allow(dead_code)] // retained for future block-level optimization
fn hfp4_block_weighted_error(block: &[f32], im: &[f32], inv_row_scale: f32, block_e: u8) -> f32 {
    let bsf = ((block_e as i32 - 127) as f32).exp2();
    let inv_bs = if bsf > 0.0 { 1.0 / bsf } else { 0.0 };
    let row_scale = 1.0 / inv_row_scale;

    let mut err = 0.0f32;
    for j in 0..32 {
        let normalized = block[j] * inv_row_scale * inv_bs;
        let q = e2m1_round(normalized);
        let recon = row_scale * bsf * E2M1_LUT[q as usize];
        let e = (recon - block[j]) * (recon - block[j]);
        err += im[j] * e;
    }
    err
}

// ─── PromotionStrategy impl ──────────────────────────────────────────────────

/// Imatrix-driven promotion: uses measured importance data to decide which
/// tensors get promoted, while keeping the same budget as K-map alternating mode.
pub struct ImatrixPromotion<'a> {
    pub imatrix: &'a ImatrixData,
    pub budget_mode: u8,
}

impl crate::strategy::PromotionStrategy for ImatrixPromotion<'_> {
    fn plan(&self, tensors: &[&str], n_layers: usize, is_moe: bool) -> std::collections::HashMap<String, crate::strategy::QuantLevel> {
        imatrix_promotion_plan(tensors, n_layers, is_moe, self.imatrix)
    }
}

// ─── Tensor promotion planning ───────────────────────────────────────────────

/// Build an imatrix-ranked tensor promotion plan that matches K-map's file-size
/// budget (same number of `Promote6` tensors as K-map alternating mode would
/// promote), but selects WHICH tensors to promote based on measured importance
/// rather than positional heuristics.
///
/// Algorithm:
/// 1. Run K-map alternating mode (mode=1) for every tensor to obtain structural
///    assignments (`F16`, `Q8`) and the promotion count baseline.
/// 2. Tensors assigned `F16` or `Q8` by K-map are kept as-is (structural).
/// 3. For the remaining "base-eligible" tensors (those K-map would assign
///    `Promote6` or `Base`), compute aggregate importance as `mean(imatrix[j])`
///    across all columns.  Tensors absent from the imatrix receive importance
///    0.0 (lowest priority).
/// 4. Sort base-eligible tensors descending by importance.
/// 5. Promote the top `n_kmap_promoted` tensors to `Promote6`; assign the rest
///    `Base`.
/// 6. Return the combined map (structural + ranked).
pub fn imatrix_promotion_plan(
    all_tensors: &[&str],
    n_layers: usize,
    is_moe: bool,
    imatrix: &ImatrixData,
) -> HashMap<String, crate::strategy::QuantLevel> {
    // Step 1 — run K-map alternating mode for every tensor.
    let kmap_levels: Vec<(&&str, crate::strategy::QuantLevel)> = all_tensors
        .iter()
        .map(|name| (name, crate::strategy::kmap_resolve_mode(name, n_layers, is_moe, 1)))
        .collect();

    // Step 2 — split into structural and base-eligible groups.
    let mut plan: HashMap<String, crate::strategy::QuantLevel> = HashMap::new();
    let mut base_eligible: Vec<&str> = Vec::new();
    let mut n_kmap_promoted: usize = 0;

    for (name, level) in &kmap_levels {
        match level {
            crate::strategy::QuantLevel::F16 | crate::strategy::QuantLevel::Q8 => {
                // Structural — keep verbatim.
                plan.insert(name.to_string(), *level);
            }
            crate::strategy::QuantLevel::Promote6 => {
                n_kmap_promoted += 1;
                base_eligible.push(name);
            }
            crate::strategy::QuantLevel::Base => {
                base_eligible.push(name);
            }
        }
    }

    // Step 3 — compute per-tensor aggregate importance.
    let mut scored: Vec<(&str, f32)> = base_eligible
        .iter()
        .map(|&name| {
            let importance = match imatrix.lookup(name) {
                Some(cols) if !cols.is_empty() => {
                    cols.iter().copied().sum::<f32>() / cols.len() as f32
                }
                _ => 0.0,
            };
            (name, importance)
        })
        .collect();

    // Step 4 — sort descending by importance (stable for determinism).
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Steps 5+6 — promote top N, rest to Base.
    for (i, (name, _)) in scored.iter().enumerate() {
        let level = if i < n_kmap_promoted {
            crate::strategy::QuantLevel::Promote6
        } else {
            crate::strategy::QuantLevel::Base
        };
        plan.insert(name.to_string(), level);
    }

    plan
}

// ─── ImatrixData methods ─────────────────────────────────────────────────────

impl ImatrixData {
    /// Look up importance data for a tensor, trying common name variants.
    ///
    /// Qwen3.5 safetensors use `model.language_model.layers.N.` while the
    /// imatrix (from GGUF convention) maps to `model.layers.N.`.  This
    /// helper tries the exact name first, then strips known prefixes.
    pub fn lookup(&self, name: &str) -> Option<&Vec<f32>> {
        if let Some(v) = self.tensors.get(name) {
            return Some(v);
        }
        // Strip "model.language_model." → "model."
        if let Some(rest) = name.strip_prefix("model.language_model.") {
            let short = format!("model.{rest}");
            if let Some(v) = self.tensors.get(&short) {
                return Some(v);
            }
        }
        // Strip "language_model.model." → "model."
        if let Some(rest) = name.strip_prefix("language_model.model.") {
            let short = format!("model.{rest}");
            if let Some(v) = self.tensors.get(&short) {
                return Some(v);
            }
        }
        None
    }

    /// Check coverage against a list of model tensor names.
    /// Prints a warning if fewer than 80% of 2D weight tensors are covered.
    pub fn check_coverage(&self, model_tensors: &[&str]) {
        let weight_tensors: Vec<&&str> = model_tensors
            .iter()
            .filter(|n| n.contains(".weight") && !n.contains("norm") && !n.contains("bias"))
            .collect();
        let matched = weight_tensors
            .iter()
            .filter(|n| self.lookup(***n).is_some())
            .count();
        let total = weight_tensors.len();
        if total > 0 && (matched as f64 / total as f64) < 0.8 {
            eprintln!(
                "WARNING: imatrix covers only {matched}/{total} weight tensors ({:.0}%). \
                 This may indicate a model mismatch or incomplete calibration.",
                100.0 * matched as f64 / total as f64
            );
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::{QuantLevel, kmap_resolve_mode, promotion_diff};

    #[test]
    fn imatrix_promotion_budget_matches_kmap() {
        // Create a mock imatrix for an 8-layer dense model.
        // K-map alt promotes edge layers (0,1,6,7) FFN only.
        let mut tensors = HashMap::new();
        for i in 0..8 {
            // Give higher importance to middle layers (opposite of K-map's edge preference)
            let imp = (4.0 - (i as f32 - 3.5).abs()) / 4.0;
            tensors.insert(
                format!("model.layers.{i}.mlp.gate_proj.weight"),
                vec![imp; 256],
            );
            tensors.insert(
                format!("model.layers.{i}.mlp.down_proj.weight"),
                vec![imp; 256],
            );
            tensors.insert(
                format!("model.layers.{i}.self_attn.q_proj.weight"),
                vec![imp * 0.5; 256],
            );
        }
        let imatrix = ImatrixData {
            tensors,
            datasets: vec!["test".to_string()],
            chunk_count: 100,
            chunk_size: 512,
        };

        let all_names: Vec<String> = (0..8).flat_map(|i| vec![
            format!("model.layers.{i}.mlp.gate_proj.weight"),
            format!("model.layers.{i}.mlp.down_proj.weight"),
            format!("model.layers.{i}.self_attn.q_proj.weight"),
        ]).collect();
        let all_refs: Vec<&str> = all_names.iter().map(|s| s.as_str()).collect();

        let plan = imatrix_promotion_plan(&all_refs, 8, false, &imatrix);

        // Count promoted tensors
        let n_promoted = plan.values()
            .filter(|&&v| v == QuantLevel::Promote6)
            .count();

        // K-map alt count
        let kmap_promoted: usize = all_refs.iter()
            .filter(|&&n| kmap_resolve_mode(n, 8, false, 1) == QuantLevel::Promote6)
            .count();

        // Budget must match
        assert_eq!(n_promoted, kmap_promoted,
            "imatrix promoted {n_promoted} but kmap would promote {kmap_promoted}");

        // But the WHICH tensors differ — imatrix prefers middle layers (higher importance)
        let kmap_plan: HashMap<String, QuantLevel> = all_refs.iter()
            .map(|&n| (n.to_string(), kmap_resolve_mode(n, 8, false, 1)))
            .collect();
        let diffs = promotion_diff(&plan, &kmap_plan);
        assert!(diffs > 0, "plans should differ since importance peaks at middle layers");
    }

    /// Dense tensor name remapping via imatrix_remap_name.
    #[test]
    fn imatrix_remap_dense_names() {
        // Standard attention projections.
        assert_eq!(
            imatrix_remap_name("blk.0.attn_q"),
            Some("model.layers.0.self_attn.q_proj.weight".to_string())
        );
        assert_eq!(
            imatrix_remap_name("blk.12.attn_k"),
            Some("model.layers.12.self_attn.k_proj.weight".to_string())
        );
        assert_eq!(
            imatrix_remap_name("blk.3.attn_v"),
            Some("model.layers.3.self_attn.v_proj.weight".to_string())
        );
        assert_eq!(
            imatrix_remap_name("blk.7.attn_output"),
            Some("model.layers.7.self_attn.o_proj.weight".to_string())
        );
        // FFN projections.
        assert_eq!(
            imatrix_remap_name("blk.1.ffn_gate"),
            Some("model.layers.1.mlp.gate_proj.weight".to_string())
        );
        assert_eq!(
            imatrix_remap_name("blk.1.ffn_up"),
            Some("model.layers.1.mlp.up_proj.weight".to_string())
        );
        assert_eq!(
            imatrix_remap_name("blk.1.ffn_down"),
            Some("model.layers.1.mlp.down_proj.weight".to_string())
        );
        // Top-level tensors.
        assert_eq!(
            imatrix_remap_name("token_embd"),
            Some("model.embed_tokens.weight".to_string())
        );
        assert_eq!(imatrix_remap_name("output"), Some("lm_head.weight".to_string()));
        // Unknown name.
        assert_eq!(imatrix_remap_name("some.unknown.tensor"), None);
    }

    /// MoE gate expert expansion returns GateUpHalf.
    #[test]
    fn imatrix_expand_moe_gate() {
        let importance: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let result = imatrix_expand_moe("blk.5.ffn_gate_exps", &importance, 3);
        match result {
            Some(MoeExpansion::GateUpHalf { layer_idx, half, experts }) => {
                assert_eq!(layer_idx, "5");
                assert_eq!(half, "gate");
                assert_eq!(experts.len(), 3);
                assert_eq!(experts[0], vec![0.0, 1.0, 2.0, 3.0]);
                assert_eq!(experts[2], vec![8.0, 9.0, 10.0, 11.0]);
            }
            other => panic!("expected GateUpHalf, got {:?}", other.is_some()),
        }
    }

    /// MoE down_proj expert expansion returns Direct.
    #[test]
    fn imatrix_expand_moe_down() {
        let importance: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let result = imatrix_expand_moe("blk.10.ffn_down_exps", &importance, 2);
        match result {
            Some(MoeExpansion::Direct(entries)) => {
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0].0, "model.layers.10.mlp.experts.0.down_proj.weight");
                assert_eq!(entries[0].1, vec![0.0, 0.5, 1.0, 1.5]);
                assert_eq!(entries[1].0, "model.layers.10.mlp.experts.1.down_proj.weight");
                assert_eq!(entries[1].1, vec![2.0, 2.5, 3.0, 3.5]);
            }
            other => panic!("expected Direct, got {:?}", other.is_some()),
        }
    }

    /// Unknown MoE projection returns None.
    #[test]
    fn imatrix_expand_moe_bad_name() {
        let importance: Vec<f32> = vec![1.0; 8];
        assert!(imatrix_expand_moe("blk.0.ffn_unknown_exps", &importance, 2).is_none());
    }

    /// Non-blk prefix returns None.
    #[test]
    fn imatrix_expand_moe_non_blk() {
        let importance: Vec<f32> = vec![1.0; 4];
        assert!(imatrix_expand_moe("output.ffn_gate_exps", &importance, 1).is_none());
    }

}
