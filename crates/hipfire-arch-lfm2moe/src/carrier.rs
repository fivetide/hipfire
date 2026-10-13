// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

use crate::config::Lfm2MoeConfig;
use crate::lfm2moe::{Lfm2MoeState, Lfm2MoeWeights};
use crate::spec_impl::Lfm2MoeBundle;
use hipfire_runtime::loader_api::{LoadCtx, ModelSource};
use hipfire_runtime::model_source::ModelSource as ModelSourceTrait;

/// LFM EOS-class markers; the first one registered becomes `eos_tok`.
const EOS_MARKERS: [&str; 3] = ["<|im_end|>", "</s>", "<|endoftext|>"];

// Special ids come from the added-token table, never from `encode(s).len() == 1`:
// LFM tokenizers prepend BOS (`add_bos` defaults true), so `encode` never yields one id.
fn resolve_eos_tok(tokenizer: &hipfire_runtime::tokenizer::Tokenizer) -> u32 {
    EOS_MARKERS
        .iter()
        .find_map(|s| tokenizer.special_token_id(s))
        .unwrap_or(1)
}

/// Every EOS-class id the LFM decode loops stop on: `eos_tok` plus each
/// registered marker (`<|endoftext|>` ends documents, `<|im_end|>` ends turns).
pub fn stop_tokens(tokenizer: &hipfire_runtime::tokenizer::Tokenizer, eos_tok: u32) -> Vec<u32> {
    let mut v = vec![eos_tok];
    for id in EOS_MARKERS
        .iter()
        .filter_map(|s| tokenizer.special_token_id(s))
    {
        if !v.contains(&id) {
            v.push(id);
        }
    }
    v
}

fn tokenizer_from_dir(
    source: &hipfire_runtime::safetensors_source::SafetensorsSource,
) -> Result<hipfire_runtime::tokenizer::Tokenizer, String> {
    if let Some(tok_path) = source.tokenizer_json_path() {
        hipfire_runtime::tokenizer::Tokenizer::from_tokenizer_json(&tok_path)
            .map_err(|e| format!("failed to parse tokenizer at {}: {e}", tok_path.display()))?
            .ok_or_else(|| format!("failed to load tokenizer from {}", tok_path.display()))
    } else {
        Err("no tokenizer.json found in model directory".into())
    }
}

/// Build the LFM2.5-MoE GPU bundle from an HFQ or safetensors-directory source.
///
/// Verbatim relocation of the `Lfm2MoeCarrier::load` model work. Preserves every
/// early-return and error string byte-for-byte. The per-source pp refusals,
/// config/weights seam (`Lfm2MoeConfig::from_hfq` / `config_from_source` and
/// `Lfm2MoeWeights::load` / `load_weights_from_source`), `maybe_screen_mmq`,
/// `Lfm2MoeState::new_with_max_seq`, and eos candidate list `EOS_MARKERS`
/// (fallback `1`) are arch knowledge
/// and live here. `dir_diag`, `resolve_source_meta`, `build_speculator`, and
/// `LoadedModel::skeleton` stay in the loader (loader-private).
pub fn load_lfm2moe_bundle(src: ModelSource, ctx: &mut LoadCtx) -> Result<Lfm2MoeBundle, String> {
    if ctx.pp > 1 {
        return Err(match &src {
            ModelSource::Hfq(_) => "lfm2moe: pipeline-parallel (pp>1) unsupported",
            ModelSource::Dir(_) => "lfm2moe: safetensors + pp>1 unsupported",
        }
        .into());
    }
    match src {
        ModelSource::Hfq(mut hfq) => {
            let config = Lfm2MoeConfig::from_hfq(&hfq)?;
            let weights = Lfm2MoeWeights::load(&mut hfq, &config, ctx.gpu)?;
            hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
            let state = Lfm2MoeState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
                .map_err(|e| format!("lfm2moe: Lfm2MoeState::new_with_max_seq failed: {e}"))?;
            let tokenizer =
                hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
                    .map_err(|e| format!("tokenizer not found: {e}"))?;
            let eos_tok = resolve_eos_tok(&tokenizer);
            Ok(Lfm2MoeBundle {
                config,
                weights,
                state,
                eos_tok,
                lfm2_decode_batch: None,
                vision_config: None,
                vision_weights: None,
            })
        }
        ModelSource::Dir(source) => {
            let config = crate::config::config_from_source(&source)
                .ok_or_else(|| "lfm2moe: failed to parse config from safetensors".to_string())?;
            let weights = crate::lfm2moe::load_weights_from_source(&source, &config, ctx.gpu)?;
            hipfire_runtime::maybe_screen_mmq(&weights, ctx.gpu);
            let state = Lfm2MoeState::new_with_max_seq(ctx.gpu, &config, ctx.max_seq)
                .map_err(|e| format!("lfm2moe: Lfm2MoeState::new_with_max_seq failed: {e}"))?;
            let tokenizer = tokenizer_from_dir(&source)?;
            let eos_tok = resolve_eos_tok(&tokenizer);
            Ok(Lfm2MoeBundle {
                config,
                weights,
                state,
                eos_tok,
                lfm2_decode_batch: None,
                // Safetensors Dir loads stay text-only — the qwen35 carrier
                // makes the same choice; VL artifacts are HFQ-only here.
                vision_config: None,
                vision_weights: None,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::tokenizer::Tokenizer;
    use serde_json::json;

    /// LFM2.5-8B-A1B layout: BOS and the markers are high added ids while id 1
    /// is an ordinary content token (`ĊĊĊĊĊĊĊĊĊ`), and `add_bos` defaults true.
    fn lfm_tokenizer() -> Tokenizer {
        let tok = json!({
            "model": {
                "type": "BPE",
                "vocab": {"<unk>": 0, "\u{2581}\u{2581}": 1, "h": 2, "i": 3, "\u{2581}": 4},
                "merges": []
            },
            "added_tokens": [
                {"id": 10, "content": "<|startoftext|>", "special": true},
                {"id": 11, "content": "<|endoftext|>", "special": true},
                {"id": 12, "content": "<|im_start|>", "special": true},
                {"id": 13, "content": "<|im_end|>", "special": true}
            ]
        });
        let meta = json!({
            "architecture": "lfm2_moe",
            "tokenizer": tok.to_string(),
            "generation_config": {"bos_token_id": 10, "eos_token_id": 13},
            "tokenizer_config": {}
        });
        let t = Tokenizer::from_hfq_metadata(&meta.to_string()).expect("meta");
        assert!(t.add_bos, "fixture must exercise the LFM add_bos default");
        t
    }

    #[test]
    fn eos_and_stop_set_resolve_marker_ids_under_add_bos() {
        let t = lfm_tokenizer();
        let eos = resolve_eos_tok(&t);
        assert_eq!(eos, 13, "eos_tok must be <|im_end|>, never the fallback 1");
        let stops = stop_tokens(&t, eos);
        assert!(stops.contains(&13) && stops.contains(&11), "{stops:?}");
        assert!(
            !stops.contains(&1),
            "content token 1 must not stop: {stops:?}"
        );
    }
}
