//! Quantization strategy traits and K-map promotion infrastructure.
//!
//! This module defines the abstractions for scale solving and tensor
//! promotion decisions, plus the concrete K-map implementation that
//! assigns per-tensor precision levels based on name patterns and
//! positional heuristics.

use std::collections::HashMap;

// ─── Scale strategy ─────────────────────────────────────────────────────────

/// Strategy for computing quantization scale and zero-point for a group of
/// floating-point values.
pub trait ScaleStrategy {
    /// Solve for affine quantization parameters (scale, zero) that map
    /// `group` values into `n_levels` integer bins.
    fn solve_affine(&self, group: &[f32], n_levels: u32) -> (f32, f32);

    /// Compute the FP4 row-level scale (E2M1 format).
    /// Default: `max_abs / 6.0` (the max representable E2M1 magnitude).
    fn solve_fp4_row_scale(&self, row: &[f32]) -> f32 {
        let max_abs = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if max_abs > 0.0 { max_abs / 6.0 } else { 1.0 }
    }

    /// Compute the FP4 block-level exponent (UE8M0 format).
    /// `block_idx` is the 0-based block index within the row (block = row[idx*32..(idx+1)*32]).
    /// Default: pass through the caller's `default_e`.
    fn solve_fp4_block_e(&self, _block: &[f32], _inv_row_scale: f32, default_e: u8, _block_idx: usize) -> u8 {
        default_e
    }
}

/// Min/max scale strategy: maps the full [min, max] range to `n_levels` bins.
pub struct MinMaxScale;

impl ScaleStrategy for MinMaxScale {
    fn solve_affine(&self, group: &[f32], n_levels: u32) -> (f32, f32) {
        let min = group.iter().fold(f32::INFINITY, |m, &v| m.min(v));
        let max = group.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
        let range = max - min;
        let scale = if range > 0.0 { range / (n_levels - 1) as f32 } else { 1.0 };
        (scale, min)
    }
}

// ─── Promotion strategy ─────────────────────────────────────────────────────

/// Per-tensor precision level assigned by the K-map pre-pass.
/// Determines whether a tensor gets the base format, a 6-bit promotion,
/// Q8, or F16. See docs/superpowers/specs/2026-05-08-mixed-quant-kmap-design.md.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantLevel {
    /// Store as F16 (norms, biases, 1D tensors).
    F16,
    /// Store as Q8_F16 (embeddings, lm_head, MoE routers).
    Q8,
    /// Promote to 6-bit variant of the base format (edge layers, MoE expert FFN).
    Promote6,
    /// Use the base format as-is.
    Base,
}

/// Strategy for deciding which tensors get promoted to higher precision.
pub trait PromotionStrategy {
    /// Return a map of tensor name -> `QuantLevel` for each tensor in the model.
    fn plan(&self, tensors: &[&str], n_layers: usize, is_moe: bool) -> HashMap<String, QuantLevel>;
}

/// No-op promotion: returns an empty map (everything stays at base).
pub struct NoPromotion;

impl PromotionStrategy for NoPromotion {
    fn plan(&self, _: &[&str], _: usize, _: bool) -> HashMap<String, QuantLevel> {
        HashMap::new()
    }
}

/// K-map promotion: uses name-pattern heuristics and positional rules to
/// assign precision levels.
pub struct KmapPromotion {
    /// K-map mode: 0 = full, 1 = alternating, 2 = typed.
    pub mode: u8,
}

impl PromotionStrategy for KmapPromotion {
    fn plan(&self, tensors: &[&str], n_layers: usize, is_moe: bool) -> HashMap<String, QuantLevel> {
        let mut map = HashMap::new();
        for &name in tensors {
            let level = kmap_resolve_mode(name, n_layers, is_moe, self.mode);
            map.insert(name.to_string(), level);
        }
        map
    }
}

// ─── K-map infrastructure ───────────────────────────────────────────────────

/// Stride for alternating-mode promotion: edge layers always promoted,
/// plus every Nth middle layer. 3 was chosen empirically — promotes ~40%
/// of middle layers, matching llama.cpp Q4_K_M's budget-allocation pattern.
/// On MoE 3.6-35B-A3B: stride=3 gives PPL 8K=19.96 at 21.8 GB vs full
/// K-map PPL 8K=20.07 at 27.7 GB.
pub const ALTERNATING_STRIDE: usize = 3;

/// Extract layer index from a tensor name.
/// Handles both safetensors (`layers.{N}.`) and GGUF (`blk.{N}.`) patterns.
/// Uses unanchored search to handle any prefix (model.layers, model.language_model.layers, etc.).
pub fn parse_layer_idx(name: &str) -> Option<usize> {
    // Try safetensors pattern: "layers.{N}."
    if let Some(pos) = name.find("layers.") {
        let after = &name[pos + 7..]; // skip "layers."
        if let Some(dot) = after.find('.') {
            if let Ok(idx) = after[..dot].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    // Try GGUF pattern: "blk.{N}."
    if let Some(pos) = name.find("blk.") {
        let after = &name[pos + 4..]; // skip "blk."
        if let Some(dot) = after.find('.') {
            if let Ok(idx) = after[..dot].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

/// llama.cpp-style alternating promotion: edge layers always promoted,
/// middle layers promoted every `stride` layers.
pub fn is_positional_promote(idx: usize, n_layers: usize, stride: usize) -> bool {
    if n_layers == 0 || stride == 0 { return false; }
    if idx < 2 || idx >= n_layers.saturating_sub(2) { return true; }
    (idx - 2) % stride == 0
}

/// Resolve the quantization level for a tensor based on its name, the model's
/// layer count, whether the model is MoE, and the K-map mode (full = mode 0).
pub fn kmap_resolve(name: &str, n_layers: usize, is_moe: bool) -> QuantLevel {
    kmap_resolve_mode(name, n_layers, is_moe, 0)
}

/// Resolve the quantization level for a tensor based on its name, the model's
/// layer count, whether the model is MoE, and the K-map mode.
///
/// `kmap_mode`: 0 = full (all candidates promoted), 1 = alternating
/// (experts + ffn_down every 3rd middle layer, edge layers always),
/// 2 = typed (ffn_down + attn_v everywhere).
///
/// Note: In the safetensors path, norms/biases are filtered by `should_quantize()`
/// before this function is called. Rules 1-2 exist for the GGUF path and completeness.
pub fn kmap_resolve_mode(name: &str, n_layers: usize, is_moe: bool, kmap_mode: u8) -> QuantLevel {
    // Rule 1: norms, biases, 1D (GGUF path mainly)
    if name.contains("norm") || name.contains("bias") {
        return QuantLevel::F16;
    }

    // Rule 2: embeddings, lm_head, output projection
    if name.contains("embed_tokens") || name.contains("token_embd")
        || name.contains("lm_head") || name.ends_with("output.weight")
    {
        return QuantLevel::Q8;
    }

    // Rule 3: MoE routers
    if is_moe
        && (name.ends_with("mlp.gate.weight")
            || name.contains("shared_expert_gate"))
    {
        return QuantLevel::Q8;
    }

    // Rule 4: MoE expert FFN weights
    if is_moe && name.contains("mlp.experts.") {
        if kmap_mode == 1 {
            // Alternating: promote expert groups only in positional layers
            if let Some(idx) = parse_layer_idx(name) {
                if is_positional_promote(idx, n_layers, ALTERNATING_STRIDE) {
                    return QuantLevel::Promote6;
                }
                return QuantLevel::Base;
            }
        }
        return QuantLevel::Promote6;
    }

    // Mode 2 (typed): promote ffn_down and attn_v in all layers.
    if kmap_mode == 2 {
        let is_down = name.contains("down_proj") || name.contains("ffn_down");
        let is_v = name.contains("v_proj") || name.contains("attn_v");
        if is_down || is_v {
            return QuantLevel::Promote6;
        }
        if n_layers > 0 {
            if let Some(idx) = parse_layer_idx(name) {
                if idx < 2 || idx >= n_layers.saturating_sub(2) {
                    return QuantLevel::Promote6;
                }
            }
        }
        return QuantLevel::Base;
    }

    // Mode 1 (alternating): ffn_down in edge + every 3rd middle layer.
    // Edge-layer rule mirrors mode 0 below: attn+FFN for MoE (full promotion
    // gives -19.8% PPL on 3.6-35B-A3B), FFN only for dense (attn promotion
    // regresses PPL +3.1% on 27B). Bench: asym4 KV, ctx=8192, wikitext-2-test.
    // See ppl_kmap_20260508.md.
    if kmap_mode == 1 {
        let is_down = name.contains("down_proj") || name.contains("ffn_down");
        if n_layers > 0 {
            if let Some(idx) = parse_layer_idx(name) {
                if is_down && is_positional_promote(idx, n_layers, ALTERNATING_STRIDE) {
                    return QuantLevel::Promote6;
                }
                // Edge layers: attn+FFN for MoE, FFN only for dense.
                if idx < 2 || idx >= n_layers.saturating_sub(2) {
                    if is_moe {
                        return QuantLevel::Promote6;
                    }
                    let is_ffn = name.contains("mlp.") || name.contains("ffn");
                    if is_ffn {
                        return QuantLevel::Promote6;
                    }
                }
            }
        }
        return QuantLevel::Base;
    }

    // Rule 5 (full mode 0): edge layers (first 2 + last 2).
    // Dense models: FFN only — attn promotion regresses PPL (+3.1% on 27B).
    // MoE models: attn+FFN — full promotion gives -19.8% PPL on 3.6-35B-A3B.
    // Bench: asym4 KV, ctx=8192, wikitext-2-test. See ppl_kmap_20260508.md.
    if n_layers > 0 {
        if let Some(idx) = parse_layer_idx(name) {
            if idx < 2 || idx >= n_layers.saturating_sub(2) {
                if is_moe {
                    // MoE: promote all tensors in edge layers (attn + FFN)
                    return QuantLevel::Promote6;
                }
                // Dense: promote FFN only — attn stays at Base
                let is_ffn = name.contains("mlp.") || name.contains("ffn");
                if is_ffn {
                    return QuantLevel::Promote6;
                }
            }
        }
    }

    // Rule 6: everything else
    QuantLevel::Base
}

/// Count the number of tensors whose `QuantLevel` differs between two plans.
///
/// Only tensors present in **both** maps are compared; tensors unique to one
/// plan are ignored (they cannot produce a meaningful diff).
pub fn promotion_diff(
    plan_a: &HashMap<String, QuantLevel>,
    plan_b: &HashMap<String, QuantLevel>,
) -> u32 {
    plan_a
        .iter()
        .filter(|(name, &level_a)| {
            plan_b.get(*name).map_or(false, |&level_b| level_a != level_b)
        })
        .count() as u32
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_layer_idx_safetensors_dense() {
        assert_eq!(parse_layer_idx("model.layers.0.self_attn.q_proj.weight"), Some(0));
        assert_eq!(parse_layer_idx("model.layers.63.mlp.gate_proj.weight"), Some(63));
    }

    #[test]
    fn parse_layer_idx_safetensors_moe() {
        assert_eq!(
            parse_layer_idx("model.language_model.layers.5.mlp.experts.0.gate_up_proj.weight"),
            Some(5)
        );
    }

    #[test]
    fn parse_layer_idx_gguf() {
        assert_eq!(parse_layer_idx("blk.0.attn_q.weight"), Some(0));
        assert_eq!(parse_layer_idx("blk.31.ffn_gate.weight"), Some(31));
    }

    #[test]
    fn parse_layer_idx_no_match() {
        assert_eq!(parse_layer_idx("token_embd.weight"), None);
        assert_eq!(parse_layer_idx("output.weight"), None);
    }

    #[test]
    fn kmap_norms_are_f16() {
        assert_eq!(kmap_resolve("model.layers.0.input_layernorm.weight", 64, false), QuantLevel::F16);
        assert_eq!(kmap_resolve("model.layers.30.post_attention_layernorm.weight", 64, false), QuantLevel::F16);
    }

    #[test]
    fn kmap_embeds_are_q8() {
        assert_eq!(kmap_resolve("model.embed_tokens.weight", 64, false), QuantLevel::Q8);
        assert_eq!(kmap_resolve("lm_head.weight", 64, false), QuantLevel::Q8);
        assert_eq!(kmap_resolve("output.weight", 64, false), QuantLevel::Q8);
    }

    #[test]
    fn kmap_moe_router_q8() {
        assert_eq!(
            kmap_resolve("model.language_model.layers.5.mlp.gate.weight", 64, true),
            QuantLevel::Q8
        );
        assert_eq!(
            kmap_resolve("model.language_model.layers.5.mlp.shared_expert_gate.weight", 64, true),
            QuantLevel::Q8
        );
    }

    #[test]
    fn kmap_moe_router_not_promoted_on_dense() {
        // On a dense model, mlp.gate.weight is not a router — falls to edge/base
        assert_ne!(
            kmap_resolve("model.layers.30.mlp.gate.weight", 64, false),
            QuantLevel::Q8
        );
    }

    #[test]
    fn kmap_moe_expert_ffn_promote6() {
        assert_eq!(
            kmap_resolve("model.language_model.layers.30.mlp.experts.5.gate_up_proj.weight", 64, true),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.language_model.layers.30.mlp.experts.5.down_proj.weight", 64, true),
            QuantLevel::Promote6
        );
    }

    #[test]
    fn kmap_edge_layers_dense_ffn_only() {
        // Dense: FFN in edge layers — promoted
        assert_eq!(kmap_resolve("model.layers.0.mlp.gate_proj.weight", 64, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.1.mlp.down_proj.weight", 64, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.62.mlp.up_proj.weight", 64, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.63.mlp.down_proj.weight", 64, false), QuantLevel::Promote6);
        // Dense: attn in edge layers — NOT promoted
        assert_eq!(kmap_resolve("model.layers.0.self_attn.q_proj.weight", 64, false), QuantLevel::Base);
        assert_eq!(kmap_resolve("model.layers.63.self_attn.v_proj.weight", 64, false), QuantLevel::Base);
        assert_eq!(kmap_resolve("model.layers.0.linear_attn.in_proj_qkv.weight", 64, false), QuantLevel::Base);
    }

    #[test]
    fn kmap_edge_layers_moe_attn_and_ffn() {
        // MoE: both attn and FFN in edge layers — promoted
        assert_eq!(kmap_resolve("model.language_model.layers.0.self_attn.q_proj.weight", 64, true), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.language_model.layers.0.mlp.gate_proj.weight", 64, true), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 64, true), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.language_model.layers.63.self_attn.v_proj.weight", 64, true), QuantLevel::Promote6);
    }

    #[test]
    fn kmap_middle_layers_base() {
        assert_eq!(kmap_resolve("model.layers.2.self_attn.q_proj.weight", 64, false), QuantLevel::Base);
        assert_eq!(kmap_resolve("model.layers.30.mlp.gate_proj.weight", 64, false), QuantLevel::Base);
        assert_eq!(kmap_resolve("model.layers.61.mlp.down_proj.weight", 64, false), QuantLevel::Base);
    }

    #[test]
    fn kmap_edge_layers_small_model_24_layers() {
        // 24 layers: edge = 0,1 and 22,23
        assert_eq!(kmap_resolve("model.layers.0.mlp.gate_proj.weight", 24, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.1.mlp.gate_proj.weight", 24, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.2.mlp.gate_proj.weight", 24, false), QuantLevel::Base);
        assert_eq!(kmap_resolve("model.layers.22.mlp.gate_proj.weight", 24, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.23.mlp.gate_proj.weight", 24, false), QuantLevel::Promote6);
    }

    #[test]
    fn kmap_n_layers_zero_disables_edge() {
        assert_eq!(kmap_resolve("model.layers.0.mlp.gate_proj.weight", 0, false), QuantLevel::Base);
    }

    #[test]
    fn kmap_edge_layers_tiny_model_3_layers() {
        // 3 layers: first-2 = {0,1}, last-2 = {1,2}. All layers promoted.
        assert_eq!(kmap_resolve("model.layers.0.mlp.gate_proj.weight", 3, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.1.mlp.gate_proj.weight", 3, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.2.mlp.gate_proj.weight", 3, false), QuantLevel::Promote6);
    }

    #[test]
    fn kmap_expert_not_promoted_on_dense() {
        // "mlp.experts." in name but is_moe=false — should NOT trigger rule 4
        assert_eq!(
            kmap_resolve("model.layers.30.mlp.experts.5.gate_up_proj.weight", 64, false),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_gguf_names() {
        // GGUF edge-layer FFN (dense) — promoted
        assert_eq!(kmap_resolve("blk.0.ffn_gate.weight", 64, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("blk.63.ffn_gate.weight", 64, false), QuantLevel::Promote6);
        // GGUF edge-layer attn (dense) — NOT promoted
        assert_eq!(kmap_resolve("blk.0.attn_q.weight", 64, false), QuantLevel::Base);
        // GGUF edge-layer attn (MoE) — promoted
        assert_eq!(kmap_resolve("blk.0.attn_q.weight", 64, true), QuantLevel::Promote6);
        // GGUF middle-layer — base
        assert_eq!(kmap_resolve("blk.30.ffn_gate.weight", 64, false), QuantLevel::Base);
    }

    // ── Alternating mode tests ───────────────────────────────────────────

    #[test]
    fn positional_promote_edges() {
        assert!(is_positional_promote(0, 40, 3));
        assert!(is_positional_promote(1, 40, 3));
        assert!(is_positional_promote(38, 40, 3));
        assert!(is_positional_promote(39, 40, 3));
    }

    #[test]
    fn positional_promote_stride3() {
        // Middle layers: every 3rd starting from idx 2
        assert!(is_positional_promote(2, 40, 3));  // edge
        assert!(!is_positional_promote(3, 40, 3));
        assert!(!is_positional_promote(4, 40, 3));
        assert!(is_positional_promote(5, 40, 3));
        assert!(!is_positional_promote(6, 40, 3));
        assert!(!is_positional_promote(7, 40, 3));
        assert!(is_positional_promote(8, 40, 3));
    }

    #[test]
    fn kmap_alternating_moe_experts() {
        // MoE experts: promoted in positional layers, base in others
        assert_eq!(
            kmap_resolve_mode("model.language_model.layers.0.mlp.experts.5.gate_up_proj.weight", 40, true, 1),
            QuantLevel::Promote6 // edge layer
        );
        assert_eq!(
            kmap_resolve_mode("model.language_model.layers.5.mlp.experts.5.gate_up_proj.weight", 40, true, 1),
            QuantLevel::Promote6 // stride hit (5-2=3, 3%3==0)
        );
        assert_eq!(
            kmap_resolve_mode("model.language_model.layers.3.mlp.experts.5.gate_up_proj.weight", 40, true, 1),
            QuantLevel::Base // not on stride
        );
    }

    #[test]
    fn kmap_alternating_ffn_down() {
        // ffn_down promoted in positional layers, base in others
        assert_eq!(
            kmap_resolve_mode("model.layers.0.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Promote6 // edge
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.5.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Promote6 // stride
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.3.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Base // not on stride
        );
        // gate_proj NOT promoted in middle layers
        assert_eq!(
            kmap_resolve_mode("model.layers.5.mlp.gate_proj.weight", 40, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_alternating_n_layers_zero() {
        // With n_layers=0, alternating mode should return Base for everything
        assert_eq!(
            kmap_resolve_mode("model.layers.0.mlp.down_proj.weight", 0, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_alternating_gguf_names() {
        // GGUF ffn_down in edge layer
        assert_eq!(
            kmap_resolve_mode("blk.0.ffn_down.weight", 40, false, 1),
            QuantLevel::Promote6
        );
        // GGUF ffn_down in middle non-stride layer
        assert_eq!(
            kmap_resolve_mode("blk.3.ffn_down.weight", 40, false, 1),
            QuantLevel::Base
        );
        // GGUF ffn_gate stays base in middle
        assert_eq!(
            kmap_resolve_mode("blk.5.ffn_gate.weight", 40, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_typed_promotes_down_and_v() {
        assert_eq!(
            kmap_resolve_mode("model.layers.15.mlp.down_proj.weight", 40, false, 2),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.15.self_attn.v_proj.weight", 40, false, 2),
            QuantLevel::Promote6
        );
        // gate_proj stays base
        assert_eq!(
            kmap_resolve_mode("model.layers.15.mlp.gate_proj.weight", 40, false, 2),
            QuantLevel::Base
        );
    }

    // ── MinMaxScale tests ───────────────────────────────────────────────

    #[test]
    fn minmax_scale_basic() {
        let s = MinMaxScale;
        let data = vec![0.0, 1.0, 2.0, 3.0];
        let (scale, zero) = s.solve_affine(&data, 16);
        assert!((scale - 0.2).abs() < 1e-6, "scale={scale}");
        assert!(zero.abs() < 1e-6, "zero={zero}");
    }

    #[test]
    fn minmax_scale_constant() {
        let s = MinMaxScale;
        let data = vec![5.0; 8];
        let (scale, zero) = s.solve_affine(&data, 16);
        assert_eq!(scale, 1.0);
        assert_eq!(zero, 5.0);
    }

    // ── NoPromotion tests ───────────────────────────────────────────────

    #[test]
    fn no_promotion_returns_empty() {
        let np = NoPromotion;
        let names = vec!["model.layers.0.mlp.gate_proj.weight"];
        let plan = np.plan(&names, 64, false);
        assert!(plan.is_empty());
    }

    // ── KmapPromotion tests ─────────────────────────────────────────────

    #[test]
    fn kmap_promotion_full_mode() {
        let kp = KmapPromotion { mode: 0 };
        let names = vec![
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.30.mlp.gate_proj.weight",
            "model.embed_tokens.weight",
        ];
        let plan = kp.plan(&names, 64, false);
        assert_eq!(plan.len(), 3);
        assert_eq!(plan["model.layers.0.mlp.gate_proj.weight"], QuantLevel::Promote6);
        assert_eq!(plan["model.layers.30.mlp.gate_proj.weight"], QuantLevel::Base);
        assert_eq!(plan["model.embed_tokens.weight"], QuantLevel::Q8);
    }

    // ── promotion_diff tests ────────────────────────────────────────────

    #[test]
    fn promotion_diff_identical() {
        let mut a = HashMap::new();
        a.insert("a".to_string(), QuantLevel::Base);
        a.insert("b".to_string(), QuantLevel::Q8);
        assert_eq!(promotion_diff(&a, &a), 0);
    }

    #[test]
    fn promotion_diff_one_changed() {
        let mut a = HashMap::new();
        a.insert("a".to_string(), QuantLevel::Base);
        a.insert("b".to_string(), QuantLevel::Q8);
        let mut b = a.clone();
        b.insert("a".to_string(), QuantLevel::Promote6);
        assert_eq!(promotion_diff(&a, &b), 1);
    }
}
