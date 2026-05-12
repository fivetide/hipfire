//! hipfire-quantize: Quantize raw FP16/BF16/FP32 model weights to Q4_F16 format.
//!
//! Usage: hipfire-quantize --input <model_dir-or-gguf> --output <output.hfq> [--format mq4]
//!
//! Reads safetensors files from a HuggingFace model directory OR a single
//! `.gguf` file and produces a `.hfq` (HipFire Quantized) file with
//! RDNA-native quantized weights.

mod gguf_input;
mod imatrix;
pub mod strategy;
pub mod formats;
mod hfq_writer;

use strategy::{PromotionStrategy, NoPromotion, KmapPromotion, QuantLevel, kmap_resolve_mode, promotion_diff};
use formats::fwht::{cpu_fwht_256, gen_fwht_signs};
use formats::hfp4::{quantize_hfp4g32_2d, quantize_mfp4g32_2d};
use formats::hfq4::{quantize_hfq4g256, quantize_hfq4g128};
use formats::hfq6::quantize_hfq6g256;
use formats::hfq_sub4::{quantize_hfq3g256, quantize_hfq3g128, quantize_hfq2g256, quantize_hfq2g128};
use formats::mq4::quantize_mq4g256;
use formats::mq6::quantize_mq6g256;
use formats::mq8::quantize_mq8g256;
use formats::mq_sub4::{quantize_mq3g256, quantize_mq2g256, quantize_mq3g256_lloyd, quantize_mq2g256_lloyd};
use formats::q8::{quantize_q4f16_g64, quantize_q4k, quantize_q4_as_q8, quantize_q8f16, quantize_q8hfq};
use strategy::MinMaxScale;
use hfq_writer::*;

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

// ─── Safetensors Parser ─────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
struct SafetensorsMeta {
    #[serde(flatten)]
    tensors: HashMap<String, TensorMeta>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct SafetensorsFile {
    _file: File,
    mmap: Mmap,
    header_size: usize,
    tensors: HashMap<String, TensorMeta>,
}

impl SafetensorsFile {
    fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // First 8 bytes: u64 LE header size
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header_json = std::str::from_utf8(&mmap[8..8 + header_len]).unwrap();

        // Parse header, filtering out __metadata__ key
        let raw: serde_json::Value = serde_json::from_str(header_json).unwrap();
        let mut tensors = HashMap::new();
        if let serde_json::Value::Object(map) = raw {
            for (k, v) in map {
                if k == "__metadata__" {
                    continue;
                }
                let meta: TensorMeta = serde_json::from_value(v).unwrap();
                tensors.insert(k, meta);
            }
        }

        Ok(Self {
            _file: file,
            mmap,
            header_size: 8 + header_len,
            tensors,
        })
    }

    fn tensor_data(&self, name: &str) -> Option<(&TensorMeta, &[u8])> {
        let meta = self.tensors.get(name)?;
        let start = self.header_size + meta.data_offsets[0];
        let end = self.header_size + meta.data_offsets[1];
        Some((meta, &self.mmap[start..end]))
    }

    /// Advise the kernel to drop page cache for a tensor's data region.
    /// On UMA systems this is critical: 234 GB of mmap'd safetensors
    /// pages compete with hipMalloc for the same physical RAM.
    #[cfg(unix)]
    fn drop_tensor_pages(&self, name: &str) {
        if let Some(meta) = self.tensors.get(name) {
            let start = self.header_size + meta.data_offsets[0];
            let len = meta.data_offsets[1] - meta.data_offsets[0];
            use std::os::unix::io::AsRawFd;
            // POSIX_FADV_DONTNEED = 4
            unsafe {
                extern "C" { fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32; }
                posix_fadvise(self._file.as_raw_fd(), start as i64, len as i64, 4);
            }
        }
    }

    #[cfg(not(unix))]
    fn drop_tensor_pages(&self, _name: &str) {}

    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}


// ─── Model Discovery ────────────────────────────────────────────────────────

fn find_safetensors(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    files.sort();
    files
}

/// Determine which tensors to quantize (weight matrices) vs keep as F16 (norms, embeddings)
fn should_quantize(name: &str) -> bool {
    // Vision encoder weights stay FP16 (only 456M params, run once per image)
    if name.starts_with("model.visual.") || name.starts_with("visual.") {
        return false;
    }
    if name.contains("norm") || name.contains("bias") {
        return false;
    }
    // Quantize everything including embeddings (Q8 embedding saves ~2.3GB for 8B models)
    name.contains("weight")
}

/// For mixed quant: should this tensor be Q8 (fast) or Q4 (compressed)?
/// Q8: attention weights, embeddings, lm_head (need occupancy)
/// Q4: FFN weights (bulk of model, benefits from compression)
fn is_q8_tensor(name: &str) -> bool {
    name.contains("self_attn") || name.contains("attn_q") || name.contains("attn_k")
        || name.contains("attn_v") || name.contains("attn_output")
        || name.contains("q_proj") || name.contains("k_proj")
        || name.contains("v_proj") || name.contains("o_proj")
        || name.contains("embed") || name.contains("lm_head")
        // Qwen3.5 DeltaNet attention
        || name.contains("linear_attn")
        // Qwen3.5-MoE: the router (`mlp.gate.weight`, hidden_size × num_experts)
        // is small but precision-sensitive — flat-routing on a quantized router
        // shifts which experts a token sees. Same for the per-layer scalar
        // `mlp.shared_expert_gate.weight` that scales the shared expert. Keep
        // both at Q8 even in Q4-bulk modes.
        || name.ends_with("mlp.gate.weight")
        || name.ends_with("mlp.shared_expert_gate.weight")
}

// ─── Main ────────────────────────────────────────────────────────────────────

/// Resolve a model input to a local directory path.
/// Accepts: local path, HuggingFace model ID (org/name), or HF cache path.
/// If the input looks like a HF model ID and isn't a local path, tries to find it
/// in the HF cache or downloads it via huggingface-cli.
fn resolve_model_path(input: &str) -> String {
    let path = Path::new(input);

    // If it's already a valid local directory with config.json, use it directly
    if path.join("config.json").exists() {
        return input.to_string();
    }

    // Check if it looks like a HuggingFace model ID (contains exactly one /)
    if input.contains('/') && !input.contains(std::path::MAIN_SEPARATOR) || (cfg!(unix) && input.matches('/').count() == 1) {
        let parts: Vec<&str> = input.splitn(2, '/').collect();
        if parts.len() == 2 {
            let org = parts[0];
            let name = parts[1];

            // Check HF cache: ~/.cache/huggingface/hub/models--{org}--{name}/snapshots/*/
            let home = std::env::var("HOME").unwrap_or_default();
            let cache_dir = format!("{home}/.cache/huggingface/hub/models--{org}--{name}");
            let snapshots_dir = Path::new(&cache_dir).join("snapshots");

            if snapshots_dir.exists() {
                // Find the first snapshot directory
                if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                    for entry in entries.flatten() {
                        let snap_path = entry.path();
                        if snap_path.is_dir() && snap_path.join("config.json").exists() {
                            eprintln!("Resolved {input} -> {}", snap_path.display());
                            return snap_path.to_string_lossy().to_string();
                        }
                    }
                }
            }

            // Not in cache — try to download
            eprintln!("Model {input} not found locally. Downloading via huggingface-cli...");
            let status = std::process::Command::new("huggingface-cli")
                .args(["download", input])
                .status();

            match status {
                Ok(s) if s.success() => {
                    // Retry cache lookup after download
                    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                        for entry in entries.flatten() {
                            let snap_path = entry.path();
                            if snap_path.is_dir() && snap_path.join("config.json").exists() {
                                eprintln!("Downloaded {input} -> {}", snap_path.display());
                                return snap_path.to_string_lossy().to_string();
                            }
                        }
                    }
                }
                Ok(s) => eprintln!("huggingface-cli download failed with status {s}"),
                Err(e) => eprintln!("Failed to run huggingface-cli: {e}. Install with: pip install huggingface_hub"),
            }
        }
    }

    // Fall through: return as-is, will fail at config.json read with a helpful error
    input.to_string()
}

// ─── GGUF input pipeline ────────────────────────────────────────────────────

/// True if the path points to a `.gguf` file on disk.
fn is_gguf_input(p: &Path) -> bool {
    p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("gguf")
}

/// Translate llama.cpp GGUF tensor names to the HuggingFace safetensors
/// names that `hipfire_runtime::hfq::load_weights_hfq` expects. The mapping is
/// the canonical llama.cpp ↔ HF convention.
///
/// Returns None for tensors that don't have a known safetensors equivalent
/// (we then keep them under their GGUF name; the future loader can decide
/// what to do, or they're skipped).
pub(crate) fn gguf_to_safetensors_name(gguf_name: &str) -> Option<String> {
    // Top-level tensors.
    match gguf_name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".to_string()),
        "output.weight" => return Some("lm_head.weight".to_string()),
        "output_norm.weight" => return Some("model.norm.weight".to_string()),
        _ => {}
    }
    // Per-layer: blk.{N}.<slot>.weight  →  model.layers.{N}.<slot>.weight
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        // rest = "{N}.<slot>.weight"
        let dot = rest.find('.')?;
        let layer_idx = &rest[..dot];
        let slot_full = &rest[dot + 1..]; // "<slot>.weight"
        // Drop the trailing ".weight" so we can rewrite slots like "attn_q"→"self_attn.q_proj".
        let slot = slot_full.strip_suffix(".weight")?;
        let translated = match slot {
            "attn_norm" => "input_layernorm".to_string(),
            "ffn_norm" => "post_attention_layernorm".to_string(),
            "attn_q" => "self_attn.q_proj".to_string(),
            "attn_k" => "self_attn.k_proj".to_string(),
            "attn_v" => "self_attn.v_proj".to_string(),
            "attn_output" => "self_attn.o_proj".to_string(),
            "attn_q_norm" => "self_attn.q_norm".to_string(),
            "attn_k_norm" => "self_attn.k_norm".to_string(),
            "ffn_gate" => "mlp.gate_proj".to_string(),
            "ffn_up" => "mlp.up_proj".to_string(),
            "ffn_down" => "mlp.down_proj".to_string(),
            other => return Some(format!("model.layers.{layer_idx}.{other}.weight")),
        };
        return Some(format!("model.layers.{layer_idx}.{translated}.weight"));
    }
    None
}

/// True if the GGUF tensor's name is a 1D norm / RMSNorm scaling vector.
/// These stay F16 in the .hfq (no benefit from quantization, precision-sensitive).
fn gguf_is_norm_tensor(name: &str) -> bool {
    name.contains("_norm") || name.contains("norm.weight")
}

/// True if the tensor is the token embedding. We Q8 these (matches the
/// safetensors path's `is_embed` rule — Q4 is too lossy for embedding tables).
fn gguf_is_embed_tensor(name: &str) -> bool {
    name == "token_embd.weight"
}

/// Build the `config` JSON object that `hipfire_runtime::hfq::config_from_hfq`
/// reads. Mirrors the field names HuggingFace uses in `config.json` for
/// LlamaForCausalLM / Qwen3ForCausalLM, populated from the GGUF
/// `<arch>.*` metadata keys.
fn config_json_from_gguf(
    gguf: &gguf_input::GgufFile,
    arch_str: &str,
) -> serde_json::Value {
    // GGUF prefixes its model hyperparameters with the architecture name —
    // e.g. for `general.architecture=llama` the keys live under `llama.*`.
    let prefix = arch_str;

    let read_u = |k: &str| -> Option<u64> {
        gguf.metadata
            .get(k)
            .and_then(|v| match v {
                gguf_input::MetaValue::U8(x) => Some(*x as u64),
                gguf_input::MetaValue::I8(x) => Some(*x as u64),
                gguf_input::MetaValue::U16(x) => Some(*x as u64),
                gguf_input::MetaValue::I16(x) => Some(*x as u64),
                gguf_input::MetaValue::U32(x) => Some(*x as u64),
                gguf_input::MetaValue::I32(x) => Some(*x as u64),
                gguf_input::MetaValue::U64(x) => Some(*x),
                gguf_input::MetaValue::I64(x) => Some(*x as u64),
                _ => None,
            })
    };
    let read_f = |k: &str| -> Option<f64> {
        gguf.metadata
            .get(k)
            .and_then(|v| match v {
                gguf_input::MetaValue::F32(x) => Some(*x as f64),
                gguf_input::MetaValue::F64(x) => Some(*x),
                _ => None,
            })
    };

    let dim = read_u(&format!("{prefix}.embedding_length"));
    let n_layers = read_u(&format!("{prefix}.block_count"));
    let n_heads = read_u(&format!("{prefix}.attention.head_count"));
    let n_kv_heads = read_u(&format!("{prefix}.attention.head_count_kv"))
        .or(n_heads);
    let hidden_dim = read_u(&format!("{prefix}.feed_forward_length"));
    // vocab_size: prefer metadata, fall back to token_embd shape[1].
    let vocab_size = read_u(&format!("{prefix}.vocab_size")).or_else(|| {
        gguf.tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.shape.get(1).map(|&s| s as u64))
    });
    let max_seq_len = read_u(&format!("{prefix}.context_length"));
    let rope_theta = read_f(&format!("{prefix}.rope.freq_base"));
    let rms_eps = read_f(&format!("{prefix}.attention.layer_norm_rms_epsilon"));
    let head_dim = read_u(&format!("{prefix}.attention.key_length"))
        .or_else(|| {
            // Fall back: head_dim = dim / n_heads.
            dim.zip(n_heads)
                .map(|(d, h)| if h > 0 { d / h } else { d })
        });
    let bos = read_u("tokenizer.ggml.bos_token_id").unwrap_or(1);
    let eos = read_u("tokenizer.ggml.eos_token_id").unwrap_or(2);

    let mut cfg = serde_json::Map::new();
    cfg.insert(
        "model_type".to_string(),
        serde_json::Value::from(arch_str.to_string()),
    );
    if let Some(v) = dim {
        cfg.insert("hidden_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = n_layers {
        cfg.insert("num_hidden_layers".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = n_heads {
        cfg.insert(
            "num_attention_heads".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = n_kv_heads {
        cfg.insert(
            "num_key_value_heads".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = hidden_dim {
        cfg.insert(
            "intermediate_size".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = vocab_size {
        cfg.insert("vocab_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = max_seq_len {
        cfg.insert(
            "max_position_embeddings".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = rope_theta {
        cfg.insert("rope_theta".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = rms_eps {
        cfg.insert("rms_norm_eps".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = head_dim {
        cfg.insert("head_dim".to_string(), serde_json::Value::from(v));
    }
    cfg.insert("bos_token_id".to_string(), serde_json::Value::from(bos));
    cfg.insert("eos_token_id".to_string(), serde_json::Value::from(eos));
    serde_json::Value::Object(cfg)
}

/// Translate the GGUF metadata HashMap into a JSON object that ends up in
/// the `.hfq` header's metadata blob. A future engine-side `from_hfq` for
/// Llama-style models can read these fields the same way the existing
/// `from_gguf` reads them today.
fn gguf_meta_to_json(meta: &HashMap<String, gguf_input::MetaValue>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, v) in meta {
        let json_v = mv_to_json(v);
        map.insert(k.clone(), json_v);
    }
    serde_json::Value::Object(map)
}

fn mv_to_json(v: &gguf_input::MetaValue) -> serde_json::Value {
    use gguf_input::MetaValue as MV;
    match v {
        MV::U8(x) => serde_json::Value::from(*x),
        MV::I8(x) => serde_json::Value::from(*x),
        MV::U16(x) => serde_json::Value::from(*x),
        MV::I16(x) => serde_json::Value::from(*x),
        MV::U32(x) => serde_json::Value::from(*x),
        MV::I32(x) => serde_json::Value::from(*x),
        MV::F32(x) => serde_json::Value::from(*x),
        MV::Bool(x) => serde_json::Value::from(*x),
        MV::String(s) => serde_json::Value::from(s.clone()),
        MV::U64(x) => serde_json::Value::from(*x),
        MV::I64(x) => serde_json::Value::from(*x),
        MV::F64(x) => serde_json::Value::from(*x),
        // Tokenizer arrays (tokens, scores, merges, ...) can be huge —
        // serialize them as JSON arrays so the engine side can re-parse.
        MV::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(mv_to_json).collect())
        }
    }
}

/// 2D-weight quantization target chosen at the per-tensor level. The choice
/// per format flag:
///
/// | --format | 2D weights      | embedding | comment                          |
/// |----------|-----------------|-----------|----------------------------------|
/// | hfq4     | HFQ4G256        | Q8F16     | dense default — no FWHT, plain   |
/// | hfq6     | HFQ6G256        | Q8F16     | dense + higher quality           |
/// | mq4      | MQ4G256         | Q8F16     | Qwen3.5+ (DeltaNet) — FWHT-rot   |
/// | mq6      | MQ6G256         | Q8F16     | Qwen3.5+ (DeltaNet) + higher q   |
/// | mq3      | MQ3G256         | Q8F16     | Sub-4-bit FWHT (3.25 bpw)        |
/// | mq2      | MQ2G256         | Q8F16     | Sub-4-bit FWHT (2.25 bpw)        |
///
/// **MQ4/MQ6 for non-Qwen3.5 dense produces correct output on the Llama path
/// (the rotation cancels via `gemv_mq4g256_with_rotate`) but adds per-layer
/// `rotate_x_mq` overhead with no quality benefit — those rotations were
/// calibrated for Qwen3.5+ training.** Default is HFQ4 for dense GGUFs;
/// pass `--format mq4` only when the source is a Qwen3.5+ family model.
#[derive(Clone, Copy, Debug)]
enum GgufFormat {
    Hfq4,
    Hfq6,
    Mq4,
    Mq6,
    Mq3,
    Mq2,
    Mq2Lloyd,
    Mq3Lloyd,
    Hfp4,  // HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale)
    Mfp4,  // MFP4G32 — HFP4G32 + offline FWHT rotation (drop-in MQ4 replacement)
}

impl GgufFormat {
    fn from_flag(flag: &str) -> Option<Self> {
        match flag {
            "hfq4" | "hfq4g256" | "hf4" => Some(Self::Hfq4),
            "hfq6" | "hfq6g256" | "hf6" => Some(Self::Hfq6),
            "mq4" | "mq4g256" | "magnum" => Some(Self::Mq4),
            "mq6" | "mq6g256" => Some(Self::Mq6),
            "mq3" | "mq3g256" => Some(Self::Mq3),
            "mq2" | "mq2g256" => Some(Self::Mq2),
            "mq2-lloyd" | "mq2g256-lloyd" | "mq2lloyd" => Some(Self::Mq2Lloyd),
            "mq3-lloyd" | "mq3g256-lloyd" | "mq3lloyd" => Some(Self::Mq3Lloyd),
            "hfp4" | "hfp4g32" | "hf4p" | "fp4" => Some(Self::Hfp4),
            "mfp4" | "mfp4g32" | "mf4p" => Some(Self::Mfp4),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Hfq4 => "HFQ4G256",
            Self::Hfq6 => "HFQ6G256",
            Self::Mq4 => "MQ4G256",
            Self::Mq6 => "MQ6G256",
            Self::Mq3 => "MQ3G256",
            Self::Mq2 => "MQ2G256",
            Self::Mq2Lloyd => "MQ2G256Lloyd",
            Self::Mq3Lloyd => "MQ3G256Lloyd",
            Self::Hfp4 => "HFP4G32",
            Self::Mfp4 => "MFP4G32",
        }
    }
}

/// Convert a GGUF file to a hipfire `.hfq`. Per-format quantization target
/// applies to 2D weight matrices; the embedding table is always Q8F16
/// (Q4-grade is too lossy for embeddings) and 1D norms stay F16. Tensor
/// names are translated GGUF → safetensors style so the engine's existing
/// `load_weights_hfq` can consume the output.
fn run_gguf_pipeline(input: &Path, output: &Path, format: GgufFormat, no_kmap: bool, kmap_dense: bool, kmap_mode: u8, imatrix: Option<&imatrix::ImatrixData>) -> std::io::Result<()> {
    eprintln!("=== GGUF → {} conversion ===", format.label());
    eprintln!("Input:  {}", input.display());
    eprintln!("Output: {}", output.display());

    let gguf = gguf_input::GgufFile::open(input)?;
    eprintln!("GGUF version: {}", gguf.version);
    eprintln!("Tensors: {}", gguf.tensors.len());

    let arch_str = gguf
        .meta_str("general.architecture")
        .unwrap_or("llama")
        .to_string();
    let arch_id: u32 = match arch_str.as_str() {
        "llama" => 0,
        "qwen3" | "qwen2" => 1,
        "qwen3moe" => 6,
        other => {
            eprintln!("warning: unknown GGUF architecture '{other}', tagging as llama-compatible");
            0
        }
    };
    eprintln!("Architecture: {arch_str} (id={arch_id})");

    // Metadata JSON: must populate `config.*` so engine's `config_from_hfq`
    // can reconstruct LlamaConfig at load time. Also keep the raw GGUF
    // metadata tree under `gguf_meta` for any consumer that wants original
    // values (chat template, vocab, scores, merges, etc.).
    let config_json = config_json_from_gguf(&gguf, &arch_str);
    let mut metadata = serde_json::json!({
        "architecture": arch_str,
        "source": "gguf",
        "config": config_json,
        "gguf_meta": gguf_meta_to_json(&gguf.metadata),
    });
    if let Some(im) = imatrix {
        metadata["imatrix"] = serde_json::json!({
            "datasets": im.datasets,
            "chunk_count": im.chunk_count,
            "chunk_size": im.chunk_size,
            "tensor_count": im.tensors.len(),
        });
    }
    let metadata_json = serde_json::to_string(&metadata)?;

    // FWHT signs — only used when --format is mq4/mq6. Same seed pair as the
    // safetensors path so the engine's runtime FWHT inverse stays identical.
    let needs_signs = matches!(format,
        GgufFormat::Mq4 | GgufFormat::Mq6 | GgufFormat::Mq3 | GgufFormat::Mq2
        | GgufFormat::Mq2Lloyd | GgufFormat::Mq3Lloyd | GgufFormat::Mfp4);
    let signs1 = if needs_signs { gen_fwht_signs(42, 256) } else { Vec::new() };
    let signs2 = if needs_signs { gen_fwht_signs(1042, 256) } else { Vec::new() };

    // K-map setup for GGUF path
    let is_moe = arch_id == 6;
    let n_layers: usize = config_json
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // Build K-map / imatrix promotion plan using translated (safetensors-style)
    // names where available, falling back to raw GGUF names for untranslated tensors.
    //
    // K-map is gated to MoE models only. On dense models the author's own
    // bench shows a mixed picture (PPL +1.5% to +2.5% at 2K context on 4B
    // and 27B; PPL -4.8% on 27B at 8K context — crossover at ~3K). The
    // ship-default is the conservative shape per maintainer directive
    // (2026-05-08): never silently change dense quantization. Users who
    // want K-map on dense pass `--kmap-dense` (see flag parsing below).
    let all_out_names: Vec<String> = gguf.tensors.iter()
        .map(|info| gguf_to_safetensors_name(&info.name).unwrap_or_else(|| info.name.clone()))
        .collect();
    let all_refs: Vec<&str> = all_out_names.iter().map(|s| s.as_str()).collect();

    let promoter: Box<dyn PromotionStrategy> = if no_kmap {
        Box::new(NoPromotion)
    } else if let Some(im) = imatrix {
        im.check_coverage(&all_refs);
        Box::new(imatrix::ImatrixPromotion { imatrix: im, budget_mode: kmap_mode })
    } else if !is_moe && !kmap_dense {
        Box::new(NoPromotion)
    } else {
        Box::new(KmapPromotion { mode: kmap_mode })
    };
    let kmap: HashMap<String, QuantLevel> = promoter.plan(&all_refs, n_layers, is_moe);

    // ── Promotion report ──────────────────────────────────────────────────
    if !kmap.is_empty() {
        let mut counts = [0u32; 4];
        for level in kmap.values() {
            match level {
                QuantLevel::F16 => counts[0] += 1,
                QuantLevel::Q8 => counts[1] += 1,
                QuantLevel::Promote6 => counts[2] += 1,
                QuantLevel::Base => counts[3] += 1,
            }
        }
        if let Some(im) = imatrix {
            let matched = all_refs.iter().filter(|&&n| im.lookup(n).is_some()).count();
            let kmap_plan: HashMap<String, QuantLevel> = all_refs.iter()
                .map(|&n| (n.to_string(), kmap_resolve_mode(n, n_layers, is_moe, kmap_mode)))
                .collect();
            let diff = promotion_diff(&kmap, &kmap_plan);
            eprintln!("imatrix promotion plan ({} base, {n_layers} layers{}):",
                format.label(), if is_moe { ", MoE" } else { "" });
            eprintln!("  imatrix matched: {:>4} / {} tensors ({} datasets, {} chunks)",
                matched, all_refs.len(), im.datasets.len(), im.chunk_count);
            eprintln!("  F16:       {:>4} tensors", counts[0]);
            eprintln!("  Q8:        {:>4} tensors", counts[1]);
            eprintln!("  Promote6:  {:>4} tensors", counts[2]);
            eprintln!("  Base:      {:>4} tensors", counts[3]);
            eprintln!("  vs K-map:  {:>4} tensors differ", diff);
        } else {
            let mode_label = match kmap_mode { 0 => "full", 1 => "alternating", 2 => "typed", _ => "?" };
            eprintln!("K-map plan ({} base, {n_layers} layers{}, mode={mode_label}):",
                format.label(),
                if is_moe { ", MoE" } else { "" });
            eprintln!("  F16:       {:>4} tensors", counts[0]);
            eprintln!("  Q8:        {:>4} tensors", counts[1]);
            eprintln!("  Promote6:  {:>4} tensors", counts[2]);
            eprintln!("  Base:      {:>4} tensors", counts[3]);
        }
    }

    let mut hfq_tensors: Vec<HfqTensor> = Vec::with_capacity(gguf.tensors.len());
    let mut total_params: u64 = 0;
    let mut quant_params: u64 = 0;
    let mut total_bytes_in: u64 = 0;
    let mut total_bytes_out: u64 = 0;

    for info in &gguf.tensors {
        let raw = gguf.tensor_data(info);
        let n_elements = info.numel();
        total_params += n_elements as u64;
        total_bytes_in += raw.len() as u64;

        let shape: Vec<u32> = info.shape.iter().map(|&s| s as u32).collect();

        // Tensor classification (uses the original GGUF name).
        let is_norm = gguf_is_norm_tensor(&info.name);
        let is_embed = gguf_is_embed_tensor(&info.name);
        let is_2d = info.shape.len() == 2;
        let k_dim = if is_2d { info.shape[0] } else { n_elements };

        // Translate to the safetensors-style name `hipfire_runtime::hfq::load_weights_hfq`
        // expects. If we don't have a translation, keep the original name —
        // the future loader can ignore unknown tensors.
        let out_name = gguf_to_safetensors_name(&info.name)
            .unwrap_or_else(|| info.name.clone());

        let kmap_level = kmap.get(&out_name).copied().unwrap_or(QuantLevel::Base);

        // Imatrix slice for this tensor (None for embeds/lm_head and non-matched tensors).
        let im_slice: Option<&[f32]> = if is_embed || out_name.contains("lm_head") {
            None
        } else {
            imatrix
                .and_then(|im| im.lookup(&out_name))
                .and_then(|v| {
                    if v.len() == k_dim {
                        Some(v.as_slice())
                    } else {
                        eprintln!("  imatrix: {out_name} column mismatch ({} vs {k_dim}) — skipping", v.len());
                        None
                    }
                })
        };

        let (data, quant_type, group_size, label) = if is_norm || !is_2d {
            // Norms and 1D tensors always F16 (primary gate)
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            (f16_bytes, QuantType::F16, 0u32, "F16")
        } else if kmap_level == QuantLevel::Q8 || is_embed {
            // K-map Q8 or embedding
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let q = quantize_q8f16(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::Q8F16, 32u32, "Q8_F16")
        } else if kmap_level == QuantLevel::Promote6 && k_dim % 256 == 0 {
            // K-map promote to 6-bit
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Mq4 | GgufFormat::Mq3 | GgufFormat::Mq2
                | GgufFormat::Mq2Lloyd | GgufFormat::Mq3Lloyd | GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Hfq4 | GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Hfp4 => {
                    // No HFP6 variant in v1. Promote6 for HFP4 stays at HFP4G32 (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = if let Some(im) = im_slice {
                        let strat = imatrix::ImatrixFp4Scale { importance: im };
                        quantize_hfp4g32_2d(&f32_data, m, k, &strat)
                    } else {
                        quantize_hfp4g32_2d(&f32_data, m, k, &MinMaxScale)
                    };
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    // No MFP6 variant. Promote6 for MFP4 stays at MFP4G32 (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = if let Some(im) = im_slice {
                        let strat = imatrix::ImatrixFp4Scale { importance: im };
                        quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2, &strat)
                    } else {
                        quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2, &MinMaxScale)
                    };
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
            }
        } else if k_dim % 256 == 0 {
            // 256-aligned 2D weight — quantize per the chosen format (Base level).
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Hfq4 => {
                    let q = quantize_hfq4g256(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                }
                GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Mq4 => {
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                }
                GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Mq3 => {
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                }
                GgufFormat::Mq2 => {
                    let q = quantize_mq2g256(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                }
                GgufFormat::Mq2Lloyd => {
                    let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq3Lloyd => {
                    let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                }
                GgufFormat::Hfp4 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = if let Some(im) = im_slice {
                        let strat = imatrix::ImatrixFp4Scale { importance: im };
                        quantize_hfp4g32_2d(&f32_data, m, k, &strat)
                    } else {
                        quantize_hfp4g32_2d(&f32_data, m, k, &MinMaxScale)
                    };
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = if let Some(im) = im_slice {
                        let strat = imatrix::ImatrixFp4Scale { importance: im };
                        quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2, &strat)
                    } else {
                        quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2, &MinMaxScale)
                    };
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
            }
        } else {
            // K not divisible by 256 — fall back to HFQ4-G128 (no rotation).
            // This branch fires for the rare ragged dim; ignores --format
            // (no G128 variant of mq4/mq6 exists).
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let q = quantize_hfq4g128(&f32_data, &MinMaxScale);
            quant_params += n_elements as u64;
            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
        };

        total_bytes_out += data.len() as u64;
        eprintln!(
            "  {label:>9}: {} → {} {:?} ({} src={:?}, {:.1} KB → {:.1} KB)",
            info.name,
            out_name,
            info.shape,
            n_elements,
            info.dtype,
            raw.len() as f64 / 1024.0,
            data.len() as f64 / 1024.0,
        );

        hfq_tensors.push(HfqTensor {
            name: out_name,
            quant_type,
            shape,
            group_size,
            data,
            spilled_len: 0,
        });
    }

    eprintln!("\n=== GGUF → MQ4 Summary ===");
    eprintln!("  Tensors:        {}", hfq_tensors.len());
    eprintln!("  Total params:   {total_params}");
    eprintln!(
        "  Quant'd params: {quant_params} ({:.1}%)",
        100.0 * quant_params as f64 / total_params as f64
    );
    eprintln!("  Input size:     {:.1} MB", total_bytes_in as f64 / 1e6);
    eprintln!(
        "  Output size:    {:.1} MB ({:.1}% of input)",
        total_bytes_out as f64 / 1e6,
        100.0 * total_bytes_out as f64 / total_bytes_in as f64,
    );

    write_hfq(output, arch_id, &metadata_json, &hfq_tensors, None)?;
    eprintln!("\nWrote: {}", output.display());
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Bound rayon's pool to 80% of cores (default cap; override with --threads N
    // or HIPFIRE_QUANT_THREADS env). Quantization is CPU-bound and saturates
    // memory bandwidth, so leaving headroom for the rest of the system avoids
    // making the whole box unresponsive during a multi-hour quantize run.
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let default_threads = ((cores * 8) / 10).max(1);
    let threads = args.iter().position(|a| a == "--threads")
        .and_then(|i| args.get(i + 1).and_then(|s| s.parse::<usize>().ok()))
        .or_else(|| std::env::var("HIPFIRE_QUANT_THREADS").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(default_threads);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
    eprintln!("Rayon: {threads} worker threads ({cores} cores available, default 80% = {default_threads})");


    let input_dir = args.iter().position(|a| a == "--input")
        .map(|i| &args[i + 1])
        .unwrap_or_else(|| { eprintln!("Usage: hipfire-quantize --input <model_dir> --output <output.hfq>"); std::process::exit(1); });

    let output_path = args.iter().position(|a| a == "--output")
        .map(|i| &args[i + 1])
        .unwrap_or_else(|| { eprintln!("Usage: hipfire-quantize --input <model_dir> --output <output.hfq> [--format q8f16|q4f16]"); std::process::exit(1); });

    let format = args.iter().position(|a| a == "--format")
        .map(|i| args[i + 1].as_str())
        .unwrap_or("q8f16");
    // q8f16 = all weights Q8 (interleaved blocks)
    // q4f16 = all weights Q4_F16_G64
    // q8-mixed = Q8 attn + Q4_K FFN (best tok/s for VRAM-constrained)
    // q8-fast = Q8 attn + Q4-as-Q8 FFN (all Q8 occupancy, most VRAM)
    // q8hfq = all weights Q8_HFQ (split-metadata, 128B-aligned rows)
    let use_q8 = format == "q8f16" || format == "q8";
    let use_mixed = format == "q8-mixed" || format == "mixed";
    let use_fast = format == "q8-fast" || format == "fast";
    let use_q8hfq = format == "q8hfq";
    let use_q4k_all = format == "q4k";
    let use_q4k_q8embed = format == "q4k-q8embed";
    let use_mq8g256 = format == "mq8" || format == "mq8g256";
    let use_mq4g256 = format == "mq4" || format == "mq4g256" || format == "magnum";
    let use_hfq4g256 = format == "hfq4g256" || format == "hfq4" || format == "hf4";
    let use_hfq3g256 = format == "hfq3g256";
    let use_hfq3g128 = format == "hfq3g128" || format == "hfq3" || format == "hf3"; // default HF3 = G128
    let use_hfq2g256 = format == "hfq2g256";
    let use_hfq2g128 = format == "hfq2g128" || format == "hfq2" || format == "hf2";
    let use_hfq_mixed = format == "hfq-mixed";  // Q8 attn + HFQ4 FFN
    let use_mq6g256 = format == "mq6" || format == "mq6g256";
    // Mixed: MQ4 for attention/shared-expert + MQ6 for routed experts only.
    // Saves ~15 GB vs full MQ6 on 122B-A10B (75 GB vs 90 GB), fits in 125 GB UMA.
    let use_mq4_mq6exp = format == "mq4-mq6exp" || format == "mq4-mq6experts";
    if use_mq4_mq6exp {
        eprintln!(
            "warning: --format mq4-mq6exp is deprecated. Use --format mq4 instead — \
             K-map promotes expert FFNs (and edge layers) to MQ6 automatically. \
             Proceeding as --format mq4."
        );
    }
    let use_mq3g256 = format == "mq3" || format == "mq3g256";
    let use_mq2g256 = format == "mq2" || format == "mq2g256";
    let use_mq2g256_lloyd = format == "mq2-lloyd" || format == "mq2g256-lloyd" || format == "mq2lloyd";
    let use_mq3g256_lloyd = format == "mq3-lloyd" || format == "mq3g256-lloyd" || format == "mq3lloyd";
    let use_hfq6 = format == "hfq6" || format == "hfq6g256" || format == "hf6";
    // HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale). Spec at docs/quant-formats/hfp4.md.
    let use_hfp4 = format == "hfp4" || format == "hfp4g32" || format == "hf4p" || format == "fp4";
    // MFP4G32 — HFP4G32 + offline FWHT (drop-in MQ4 replacement). Same per-row layout
    // as HFP4G32 with format_flags bit 0 + bits 2-3 = 01 stamping the rotation kind.
    let use_mfp4 = format == "mfp4" || format == "mfp4g32" || format == "mf4p";
    let q8_router_flag = args.iter().any(|a| a == "--q8-router");
    let no_kmap = args.iter().any(|a| a == "--no-kmap" || a == "--uniform");
    // K-map gate: applies to MoE models by default. Dense models opt in
    // via --kmap-dense (the K-map dense PPL effect is mixed: regression at
    // short context, win at long context — see benchmarks/results/
    // ppl_kmap_20260508.md). Maintainer directive 2026-05-08: "intends to
    // help ONLY (never on dense)" by default.
    let kmap_dense = args.iter().any(|a| a == "--kmap-dense");
    // K-map mode: 0=full (all candidates promoted), 1=alternating (edge + every 3rd),
    // 2=typed (ffn_down+attn_v everywhere). Default: alternating — same PPL as full
    // at 17% less model size on MoE (22.9 vs 27.7 GB, PPL 8K: 19.96 vs 20.07).
    let kmap_mode: u8 = args.iter().position(|a| a == "--kmap-mode")
        .and_then(|i| args.get(i + 1))
        .map(|v| match v.as_str() {
            "full" | "0" => 0,
            "alternating" | "alt" | "1" => 1,
            "typed" | "2" => 2,
            _ => { eprintln!("warning: unknown --kmap-mode '{v}', using alternating"); 1 }
        })
        .unwrap_or(1);

    let imatrix_path: Option<String> = args.iter().position(|a| a == "--imatrix")
        .map(|i| args.get(i + 1).expect("--imatrix requires a path").clone());

    // ── Sub-4-bit guards (2026-04-30 sweep) ─────────────────────────────
    // MQ2 with the current uniform 4-level codebook collapses at every
    // model size validated locally (0.8B / 4B / 9B Qwen 3.5 → multilingual
    // mojibake on all 4 coherence-gate prompts). Refuse by default until
    // Path D Lloyd-Max non-uniform codebooks land (PRD §5.2).
    let allow_mq2 = args.iter().any(|a| a == "--allow-mq2")
        || std::env::var("HIPFIRE_ALLOW_MQ2").ok().as_deref() == Some("1");
    if use_mq2g256 && !allow_mq2 {
        eprintln!(
            "error: --format mq2 is reserved — empirical quality verdict is collapse on every model\n\
             size validated locally (0.8B / 4B / 9B Qwen 3.5 → mojibake / symbol soup on all 4\n\
             coherence-gate prompts). The current uniform 4-level codebook is fundamentally too\n\
             lossy; Path D Lloyd-Max non-uniform codebooks (per-block squared-error-minimising)\n\
             are the planned remediation per PRD §5.2.\n\
             \n\
             To opt in for research / ablation purposes anyway, pass --allow-mq2 or set\n\
             HIPFIRE_ALLOW_MQ2=1. Don't ship MQ2 artifacts to users until the codebook\n\
             improvement lands."
        );
        std::process::exit(1);
    }
    // MQ2-Lloyd: rescues uniform MQ2 by 41–55× (per benchmarks/results/
    // lloyd_max_findings_20260501.md) but still text-collapse — 9B ppl=2,163
    // vs 9B MQ4 ppl=10. Research-only: same opt-in gate so users don't
    // accidentally ship a 2-bpw model that won't produce coherent output.
    let allow_mq3_lloyd = args.iter().any(|a| a == "--allow-mq3-lloyd")
        || std::env::var("HIPFIRE_ALLOW_MQ3_LLOYD").ok().as_deref() == Some("1");
    if use_mq3g256_lloyd && !allow_mq3_lloyd {
        eprintln!(
            "note: --format mq3-lloyd is research — Lloyd-Max 8-entry codebook +\n\
             3-bit indices (112 B/group, +7.7% over uniform MQ3). Hypothesis is\n\
             non-uniform codebook lifts sub-9B MQ3 out of collapse (#114) and\n\
             tightens 9B MQ3's 4× ppl gap vs MQ4. Ppl evidence pending — DO NOT\n\
             ship MQ3-Lloyd artifacts to users until quality is validated against\n\
             baseline MQ3/MQ4 ppl.\n\
             \n\
             To proceed, pass --allow-mq3-lloyd or set HIPFIRE_ALLOW_MQ3_LLOYD=1."
        );
        std::process::exit(1);
    }
    let allow_mq2_lloyd = args.iter().any(|a| a == "--allow-mq2-lloyd")
        || std::env::var("HIPFIRE_ALLOW_MQ2_LLOYD").ok().as_deref() == Some("1");
    if use_mq2g256_lloyd && !allow_mq2_lloyd {
        eprintln!(
            "error: --format mq2-lloyd is research-only — Lloyd-Max codebook lifts\n\
             uniform MQ2 by 41–55× ppl but absolute quality is still collapse\n\
             (9B Qwen 3.5 wikitext2-test ppl=2,163 vs MQ4=10, MQ3=42; 0.8B ppl=19,651).\n\
             2 bpw is fundamentally too aggressive for usable text; the format\n\
             is plumbed for follow-on Lloyd-Max MQ3 (qt=20) experiments only.\n\
             \n\
             To opt in for research anyway, pass --allow-mq2-lloyd or set\n\
             HIPFIRE_ALLOW_MQ2_LLOYD=1. Don't ship MQ2-Lloyd artifacts to users."
        );
        std::process::exit(1);
    }
    // MQ3 quality threshold ≈ 9B from the same sweep — 27B + 9B fluent,
    // 4B partial-collapse (intent recognised, language drifts), 0.8B
    // gibberish. Print a soft advisory so users running --format mq3
    // against small models don't think the engine is broken.
    if use_mq3g256 {
        eprintln!(
            "note: MQ3 empirical quality threshold ≈ 9B params. 27B / 9B Qwen 3.5 produce\n\
             fluent output across the coherence-gate battery; 4B partially collapses\n\
             (intent recognised, language mixes / loops); 0.8B is incoherent. For models\n\
             below ~9B, prefer --format mq4 (same kernel family, ~30% larger but\n\
             reliably coherent).\n"
        );
    }

    // Load imatrix data (both safetensors and GGUF paths use it).
    // Must happen before the GGUF early-return branch.
    let imatrix_data: Option<imatrix::ImatrixData> = match &imatrix_path {
        Some(path) => {
            eprintln!("Loading imatrix from {path}...");
            match imatrix::load_imatrix(std::path::Path::new(path)) {
                Ok(data) => {
                    eprintln!("  {} tensors, {} datasets, {} chunks x {} tokens",
                        data.tensors.len(), data.datasets.len(), data.chunk_count, data.chunk_size);
                    Some(data)
                }
                Err(e) => {
                    eprintln!("error: failed to load imatrix: {e}");
                    std::process::exit(1);
                }
            }
        }
        None => None,
    };

    // GGUF input branch: if --input is a `.gguf` file, run the GGUF
    // pipeline and exit. Tensor names are translated GGUF → safetensors
    // style. The 2D quantization target follows --format:
    //   hfq4 (default for GGUF) | hfq6 | mq4 | mq6
    // Per CLAUDE.md guidance: dense (non-DeltaNet) models should use
    // hfq4/hfq6. mq4/mq6 are calibrated for Qwen3.5+ — using them on a
    // Llama-style model produces correct output (the FWHT cancels in
    // `gemv_mq4g256_with_rotate`) but adds runtime rotation overhead
    // with no quality benefit.
    {
        let raw_input = Path::new(input_dir.as_str());
        if is_gguf_input(raw_input) {
            let gguf_format = GgufFormat::from_flag(format).unwrap_or_else(|| {
                eprintln!(
                    "GGUF input: --format '{format}' not recognized. \
                     Supported: hfq4 (default), hfq6, mq4, mq6. \
                     Falling back to hfq4."
                );
                GgufFormat::Hfq4
            });
            let out = Path::new(output_path);
            if let Err(e) = run_gguf_pipeline(raw_input, out, gguf_format, no_kmap, kmap_dense, kmap_mode, imatrix_data.as_ref()) {
                eprintln!("GGUF pipeline failed: {e}");
                std::process::exit(2);
            }
            return;
        }
    }

    // Resolve input: local path or HuggingFace model ID (e.g. "Qwen/Qwen3-8B")
    let input_dir = resolve_model_path(input_dir);
    let input_dir = Path::new(&input_dir);
    let output_path = Path::new(output_path);

    // Read model config
    let config_path = input_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|_| panic!("Cannot read {}. If using a HuggingFace model ID, ensure it's downloaded: huggingface-cli download {}", config_path.display(), input_dir.display()));
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

    let arch_str = config.get("model_type").and_then(|v| v.as_str()).unwrap_or("llama");
    let arch_id = match arch_str {
        "llama" => 0u32,
        "qwen3" | "qwen2" => 1,
        "qwen3_5" | "qwen3_5_text" => 5,
        // Qwen3.5 MoE (Qwen3.5-35B-A3B and friends): hybrid LA+FA attention identical
        // to qwen3_5 dense, but every layer's FFN is MoE with stacked-3D expert
        // tensors (mlp.experts.gate_up_proj/down_proj are [num_experts, ...]).
        "qwen3_5_moe" | "qwen3_5_moe_text" => 6,
        other => { eprintln!("Warning: unknown architecture '{other}', treating as llama"); 0 }
    };
    eprintln!("Architecture: {arch_str} (id={arch_id})");
    let is_moe = arch_id == 6;
    // Q8 router: always on for MoE models. 4-bit router quantization destroys
    // routing precision on precision-sensitive models (Qwen3.6-A3B: 152/256
    // expert rows drop below 0.99 cosine similarity at HFQ4G256). Cost: ~0.05%
    // model size. See github.com/Kaden-Schutt/hipfire/issues/171.
    let q8_router = is_moe || q8_router_flag;
    if is_moe {
        eprintln!("  MoE detected — will split 3D expert tensors per-expert before quantization.");
    }

    // Extract layer count for K-map edge-layer promotion.
    // Qwen3.5+ nests config under "text_config"; try both paths.
    let n_layers: usize = config
        .get("num_hidden_layers")
        .or_else(|| config.get("text_config").and_then(|tc| tc.get("num_hidden_layers")))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if n_layers == 0 {
        eprintln!("  warning: num_hidden_layers not found in config.json — edge-layer promotion disabled");
    }

    // Read tokenizer if present
    let tokenizer_json = input_dir.join("tokenizer.json");
    let tokenizer_str = if tokenizer_json.exists() {
        std::fs::read_to_string(&tokenizer_json).ok()
    } else {
        None
    };

    // Read tokenizer_config.json (has chat_template)
    let tokenizer_config_path = input_dir.join("tokenizer_config.json");
    let tokenizer_config: Option<serde_json::Value> = if tokenizer_config_path.exists() {
        std::fs::read_to_string(&tokenizer_config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    } else {
        None
    };

    // Build metadata JSON for .hfq
    let mut metadata = serde_json::json!({
        "architecture": arch_str,
        "config": config,
        "tokenizer": tokenizer_str.as_deref().unwrap_or("{}"),
        "tokenizer_config": tokenizer_config,
    });
    if let Some(ref im) = imatrix_data {
        metadata["imatrix"] = serde_json::json!({
            "file": imatrix_path.as_deref().unwrap_or("unknown"),
            "datasets": im.datasets,
            "chunk_count": im.chunk_count,
            "chunk_size": im.chunk_size,
            "tensor_count": im.tensors.len(),
        });
    }
    let metadata_json = serde_json::to_string(&metadata).unwrap();

    // Load all safetensors files
    let st_files: Vec<SafetensorsFile> = find_safetensors(input_dir)
        .iter()
        .map(|p| {
            eprintln!("Loading: {}", p.display());
            SafetensorsFile::open(p).unwrap()
        })
        .collect();

    // Collect all tensor names
    let mut all_tensors: Vec<(&str, usize)> = Vec::new();
    for (fi, st) in st_files.iter().enumerate() {
        for name in st.tensor_names() {
            all_tensors.push((name, fi));
        }
    }
    all_tensors.sort_by_key(|(name, _)| name.to_string());
    eprintln!("Found {} tensors", all_tensors.len());

    if let Some(ref im) = imatrix_data {
        let all_refs: Vec<&str> = all_tensors.iter().map(|(n, _)| *n).collect();
        im.check_coverage(&all_refs);
    }

    // ── K-map / imatrix pre-pass ─────────────────────────────────────────────
    // Build per-tensor quant level map via PromotionStrategy dispatch.
    // Gated to MoE models by default (maintainer directive 2026-05-08):
    // K-map's dense PPL effect is mixed (+1.5% to +2.5% at 2K, -4.8% at
    // 8K — crossover at ~3K context). Dense models opt out by default and
    // require `--kmap-dense` to enable.
    //
    // When --imatrix is provided, ImatrixPromotion replaces the positional
    // K-map heuristic while keeping the same promotion budget.
    let promoter: Box<dyn PromotionStrategy> = if no_kmap {
        Box::new(NoPromotion)
    } else if let Some(ref im) = imatrix_data {
        Box::new(imatrix::ImatrixPromotion { imatrix: im, budget_mode: kmap_mode })
    } else if !is_moe && !kmap_dense {
        Box::new(NoPromotion)
    } else {
        Box::new(KmapPromotion { mode: kmap_mode })
    };
    let all_names: Vec<&str> = all_tensors.iter().map(|(n, _)| *n).collect();
    let kmap: HashMap<String, QuantLevel> = promoter.plan(&all_names, n_layers, is_moe);

    // ── Promotion report ──────────────────────────────────────────────────
    if !kmap.is_empty() {
        let mut counts = [0u32; 4];
        for level in kmap.values() {
            match level {
                QuantLevel::F16 => counts[0] += 1,
                QuantLevel::Q8 => counts[1] += 1,
                QuantLevel::Promote6 => counts[2] += 1,
                QuantLevel::Base => counts[3] += 1,
            }
        }
        if imatrix_data.is_some() {
            let im = imatrix_data.as_ref().unwrap();
            let matched = all_names.iter()
                .filter(|&&n| im.lookup(n).is_some())
                .count();
            let kmap_plan: HashMap<String, QuantLevel> = all_names.iter()
                .map(|&n| (n.to_string(), kmap_resolve_mode(n, n_layers, is_moe, kmap_mode)))
                .collect();
            let diff = promotion_diff(&kmap, &kmap_plan);
            eprintln!("imatrix promotion plan ({format} base, {n_layers} layers{}):",
                if is_moe { ", MoE" } else { "" });
            eprintln!("  imatrix matched: {:>4} / {} tensors ({} datasets, {} chunks)",
                matched, all_names.len(), im.datasets.len(), im.chunk_count);
            eprintln!("  F16:       {:>4} tensors (norms, biases)", counts[0]);
            eprintln!("  Q8:        {:>4} tensors (embed, lm_head, routers)", counts[1]);
            eprintln!("  Promote6:  {:>4} tensors", counts[2]);
            eprintln!("  Base:      {:>4} tensors (remaining)", counts[3]);
            eprintln!("  vs K-map:  {:>4} tensors differ", diff);
        } else {
            let mode_label = match kmap_mode { 0 => "full", 1 => "alternating", 2 => "typed", _ => "?" };
            eprintln!("K-map plan ({format} base, {n_layers} layers{}, mode={mode_label}):",
                if is_moe { ", MoE" } else { "" });
            eprintln!("  F16:       {:>4} tensors (norms, biases)", counts[0]);
            eprintln!("  Q8:        {:>4} tensors (embed, lm_head, routers)", counts[1]);
            eprintln!("  Promote6:  {:>4} tensors", counts[2]);
            eprintln!("  Base:      {:>4} tensors (remaining)", counts[3]);
        }
    }

    // Quantize
    let mut hfq_tensors = Vec::new();
    let mut total_params = 0u64;
    let mut quantized_params = 0u64;
    // Spill file for large models — keeps peak RSS bounded by flushing
    // completed tensor data to disk when accumulated memory exceeds 32 GB.
    let spill_dir = output_path.parent().unwrap_or(Path::new("."));
    let mut spill = TensorSpill::new(spill_dir).ok();
    let mut total_quant_error = 0.0f64;
    let mut max_quant_error = 0.0f32;
    let mut _n_quant_groups = 0u64;

    let include_vision = std::env::args().any(|a| a == "--include-vision");
    let vision_quant = std::env::args().position(|a| a == "--vision-quant")
        .and_then(|i| std::env::args().nth(i + 1))
        .unwrap_or_default();
    let mut skipped_params = 0u64;
    for (name, file_idx) in &all_tensors {
        // Skip MTP head; optionally include vision encoder for VL inference
        let is_vision = name.starts_with("model.visual.") || name.starts_with("visual.");
        if is_vision && !include_vision {
            let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
            let n: usize = meta.shape.iter().product();
            skipped_params += n as u64;
            continue;
        }
        if name.starts_with("mtp.") {
            let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
            let n: usize = meta.shape.iter().product();
            skipped_params += n as u64;
            continue;
        }

        let (meta, raw_data) = st_files[*file_idx].tensor_data(name).unwrap();
        let n_elements: usize = meta.shape.iter().product();
        total_params += n_elements as u64;

        // ── MoE 3D-stacked expert tensor split ─────────────────────────────────
        // Qwen3.5-MoE stores routed experts as 3D tensors:
        //   model.language_model.layers.{N}.mlp.experts.gate_up_proj
        //     shape: [num_experts, 2 * moe_intermediate, hidden_size]
        //   model.language_model.layers.{N}.mlp.experts.down_proj
        //     shape: [num_experts, hidden_size, moe_intermediate]
        // Note: no `.weight` suffix on these, so should_quantize() returns false
        // and the standard path would store them as F16 — defeating the purpose.
        // We split into per-expert 2D MQ4G256 quantized tensors named
        //   model.language_model.layers.{N}.mlp.experts.{X}.{base}.weight
        // so the engine loader can fish them out by expert index.
        if is_moe
            && name.contains("mlp.experts.")
            && (name.ends_with("gate_up_proj") || name.ends_with("down_proj"))
            && meta.shape.len() == 3
        {
            let n_experts = meta.shape[0];
            let inner_n: usize = meta.shape[1..].iter().product();
            let elem_size = match meta.dtype.as_str() {
                "F32" => 4, "F16" | "BF16" => 2,
                other => panic!("unsupported expert tensor dtype: {other}"),
            };
            let inner_bytes = inner_n * elem_size;
            let inner_shape: Vec<u32> = meta.shape[1..].iter().map(|&s| s as u32).collect();
            let base_name = if name.ends_with("gate_up_proj") { "gate_up_proj" } else { "down_proj" };
            // Strip the trailing base; what remains is the parent path with `experts.` already on the end
            let parent = &name[..name.len() - base_name.len()];

            // Inner quantization for experts — respects --format flag.
            // MQ6 reduces quantization error that compounds across 48 MoE
            // layers × 9 expert contributions per layer at the cost of ~50%
            // more VRAM per expert. MQ4 is the default for VRAM efficiency.
            let signs1 = gen_fwht_signs(42, 256);
            let signs2 = gen_fwht_signs(1042, 256);
            let inner_k = inner_shape[1] as usize;
            let supports_g256 = inner_k % 256 == 0;
            // K-map: check the parent tensor name directly. The parent
            // (e.g. "...mlp.experts.gate_up_proj") contains "mlp.experts."
            // so kmap_resolve rule 4 matches it. The kmap HashMap was built
            // from all_tensors which has these parent names as keys.
            let kmap_promote = kmap.get(*name) == Some(&QuantLevel::Promote6);
            let expert_mq6 = (use_mq6g256 || use_mq4_mq6exp || (kmap_promote && use_mq4g256)) && supports_g256;
            let expert_hfq6 = (use_hfq6 || (kmap_promote && use_hfq4g256)) && supports_g256;
            let expert_hfq4 = use_hfq4g256 && !kmap_promote && supports_g256;

            // Parallelize across the 256 expert slices via rayon. Each slice
            // dequant→FWHT→quant→pack is a CPU-bound, self-contained job.
            // The outer Rayon pool size is set in main() before this runs.
            use rayon::prelude::*;
            let dtype = meta.dtype.clone();
            let parent_owned = parent.to_string();
            let inner_shape_clone = inner_shape.clone();
            let base_owned = base_name.to_string();
            // Borrow imatrix_data before the parallel closure to avoid move.
            let imatrix_ref = imatrix_data.as_ref();
            let mut new_tensors: Vec<HfqTensor> = (0..n_experts).into_par_iter().map(|x| {
                let slice_off = x * inner_bytes;
                let slice = &raw_data[slice_off..slice_off + inner_bytes];
                let f32_slice = to_f32(slice, &dtype);
                let expert_name = format!("{parent_owned}{x}.{base_owned}.weight");
                let im_slice: Option<&[f32]> = imatrix_ref
                    .and_then(|im| im.lookup(&expert_name))
                    .map(|v| v.as_slice());
                let (quantized, qt, gs) = if expert_mq6 {
                    let q = quantize_mq6g256(&f32_slice, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ6G256, 256u32)
                } else if expert_hfq6 {
                    let q = quantize_hfq6g256(&f32_slice, &MinMaxScale);
                    (q, QuantType::HFQ6G256, 256u32)
                } else if expert_hfq4 {
                    let q = quantize_hfq4g256(&f32_slice, &MinMaxScale);
                    (q, QuantType::HFQ4G256, 256u32)
                } else if supports_g256 {
                    let q = quantize_mq4g256(&f32_slice, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ4G256, 256u32)
                } else {
                    let q = quantize_hfq4g128(&f32_slice, &MinMaxScale);
                    (q, QuantType::HFQ4G128, 128u32)
                };
                HfqTensor {
                    name: expert_name,
                    quant_type: qt,
                    shape: inner_shape_clone.clone(),
                    group_size: gs,
                    data: quantized,
                    spilled_len: 0,
                }
            }).collect();
            quantized_params += inner_n as u64 * n_experts as u64;
            // Single eprintln to summarize the whole expert sweep.
            let label = if expert_mq6 { "MQ6G256" } else if expert_hfq6 { "HFQ6G256" } else if expert_hfq4 { "HFQ4G256" } else if supports_g256 { "MQ4G256" } else { "HFQ4G128" };
            let bytes_per = new_tensors.first().map(|t| t.data.len()).unwrap_or(0);
            eprintln!("  {label:>8}: {parent_owned}{{0..{n_experts}}}.{base_owned}.weight {:?} (×{n_experts} experts || {:.1} KB/expert, parallel)",
                inner_shape, bytes_per as f64 / 1024.0);
            hfq_tensors.append(&mut new_tensors);
            // Drop source pages and spill quantized data after each expert batch.
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut s) = spill {
                maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024); // 2 GB threshold
            }
            continue;
        }

        if should_quantize(name) && n_elements >= 32 {
            let f32_data = to_f32(raw_data, &meta.dtype);
            quantized_params += n_elements as u64;

            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();

            // Q8HFQ path: split-metadata per-row layout (needs M and K)
            // Exclude embeddings — they use a lookup kernel, not GEMV
            if use_q8hfq && meta.shape.len() == 2 && !name.contains("embed_tokens") {
                let m = meta.shape[0];
                let k = meta.shape[1];
                let (quantized, row_stride) = quantize_q8hfq(&f32_data, m, k);

                // Compute quantization error for Q8HFQ
                let n_groups = k / 32;
                let scales_bytes = n_groups * 2;
                for row in 0..m {
                    let row_off = row * row_stride;
                    for g in 0..n_groups {
                        let scale = f16_to_f32(u16::from_le_bytes([
                            quantized[row_off + g * 2],
                            quantized[row_off + g * 2 + 1],
                        ]));
                        for i in 0..32 {
                            let qval = quantized[row_off + scales_bytes + g * 32 + i] as i8;
                            let dequant = scale * qval as f32;
                            let orig_idx = row * k + g * 32 + i;
                            let err = (dequant - f32_data[orig_idx]).abs();
                            total_quant_error += err as f64;
                            max_quant_error = max_quant_error.max(err);
                        }
                        _n_quant_groups += 1;
                    }
                }

                eprintln!("  {:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB, stride={})",
                    "Q8_HFQ", name, meta.shape, n_elements,
                    raw_data.len() as f64 / 1024.0,
                    quantized.len() as f64 / 1024.0,
                    row_stride);

                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::Q8HFQ,
                    shape,
                    group_size: 32,
                    data: quantized,
                    spilled_len: 0,
                });
            } else {

            // ── imatrix slice lookup ─────────────────────────────────────────
            // Skip imatrix for embeddings and lm_head (Q8 already, imatrix
            // hurts lookup tables where every row is equally important).
            let im_slice: Option<&[f32]> = if name.contains("embed_tokens") || name.contains("lm_head") {
                None
            } else {
                imatrix_data.as_ref()
                    .and_then(|im| im.lookup(*name))
                    .and_then(|v| {
                        let k_dim = *meta.shape.last().unwrap_or(&0);
                        if v.len() == k_dim {
                            Some(v.as_slice())
                        } else {
                            eprintln!("  imatrix: {name} column mismatch ({} vs {k_dim}) — skipping", v.len());
                            None
                        }
                    })
            };

            // ── K-map override ──────────────────────────────────────────────
            let kmap_level = kmap.get(&**name).copied().unwrap_or(QuantLevel::Base);

            let (quantized, qt, gs, label) = if kmap_level == QuantLevel::Q8 {
                // K-map says Q8 (embed, lm_head, router)
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if kmap_level == QuantLevel::F16 {
                // K-map says F16 (should not normally reach here — should_quantize filters first)
                let f16_bytes: Vec<u8> = f32_data
                    .iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect();
                (f16_bytes, QuantType::F16, 0u32, "F16")
            } else if kmap_level == QuantLevel::Promote6 {
                // K-map says promote to 6-bit
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if (use_mq4g256 || use_mq4_mq6exp || use_mq3g256 || use_mq2g256
                    || use_mq2g256_lloyd || use_mq3g256_lloyd) && k_dim % 256 == 0
                {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                } else if (use_hfq4g256 || use_hfq3g256 || use_hfq3g128
                    || use_hfq2g256 || use_hfq2g128) && k_dim % 256 == 0
                {
                    let q = quantize_hfq6g256(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                } else if use_mq6g256 && k_dim % 256 == 0 {
                    // Already 6-bit MQ — no-op promotion
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                } else if use_hfq6 && k_dim % 256 == 0 {
                    // Already 6-bit HFQ — no-op promotion
                    let q = quantize_hfq6g256(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                } else {
                    // Non-256-aligned fallback: Q8
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                }
            } else {
            // QuantLevel::Base — existing format-specific logic below

            // Choose quant format per tensor
            let this_q8 = if use_q4k_all {
                false // everything Q4_K
            } else if use_q4k_q8embed {
                name.contains("embed") || name.contains("lm_head") // only embed/output Q8
            } else if use_mixed || use_fast {
                is_q8_tensor(name)
            } else {
                use_q8 || use_q8hfq // 1D Q8HFQ tensors fall back to Q8F16
            };
            let this_q4as8 = use_fast && !this_q8; // FFN tensors in q8-fast mode
            let this_q4k = use_q4k_all || use_q4k_q8embed || use_mixed;

            // Embeddings stored as Q8 in HFQ4 mode — Q4 is too lossy for
            // large-dim models (9B: dim=4096, values ~0.016, Q4 step ~0.007)
            let is_embed = name.contains("embed_tokens");

            if use_hfq_mixed {
                // hfq-mixed: Q8 for attention, HFQ4 for FFN (fits 9B in 8GB VRAM)
                let is_ffn = name.contains("mlp.") || name.contains("ffn");
                if !is_ffn {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else {
                    let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                    if k_dim % 256 == 0 {
                        let q = quantize_hfq4g256(&f32_data, &MinMaxScale);
                        (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                    } else {
                        let q = quantize_hfq4g128(&f32_data, &MinMaxScale);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                }
            } else if use_hfq6 {
                // HFQ6-G256: all weights 6-bit, embeddings Q8
                if is_embed {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else {
                    let q = quantize_hfq6g256(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
            } else if (use_hfq2g256 || use_hfq2g128) && is_embed {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_hfq2g128 {
                let q = quantize_hfq2g128(&f32_data, &MinMaxScale);
                (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
            } else if use_hfq2g256 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let q = quantize_hfq2g256(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ2G256, 256u32, "HFQ2G256")
                } else {
                    // Fallback to HFQ4 for non-256-aligned
                    let q = quantize_hfq4g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                }
            } else if use_mq8g256 && is_embed {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_mq8g256 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq8g256(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ8G256, 256u32, "MQ8G256")
                } else {
                    // Fallback to Q8 for non-256-aligned
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                }
            } else if q8_router && is_q8_tensor(name) {
                // Q8 router for MoE: keep mlp.gate.weight and
                // shared_expert_gate.weight at Q8 regardless of --format.
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if (use_mq4g256 || use_mq4_mq6exp) && is_embed {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_mq4g256 || use_mq4_mq6exp {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                } else {
                    // Fallback to standard HFQ4-G128 for non-256-aligned
                    let q = quantize_hfq4g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                }
            } else if use_hfp4 && is_embed {
                // HFP4 embeddings stay Q8F16 (matches MQ4 / HFQ4 pattern — embedding lookup is
                // accuracy-sensitive, FP4 codes too lossy for vocab-sized tables).
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_hfp4 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 32 == 0 && meta.shape.len() == 2 {
                    let m = meta.shape[0];
                    let q = if let Some(im) = im_slice {
                        let strat = imatrix::ImatrixFp4Scale { importance: im };
                        quantize_hfp4g32_2d(&f32_data, m, k_dim, &strat)
                    } else {
                        quantize_hfp4g32_2d(&f32_data, m, k_dim, &MinMaxScale)
                    };
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                } else {
                    // Fallback to HFQ4-G128 for non-32-aligned ragged dims (rare).
                    let q = quantize_hfq4g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                }
            } else if use_mfp4 && is_embed {
                // MFP4 embeddings stay Q8F16 (same rationale as HFP4 / MQ4).
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_mfp4 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 && meta.shape.len() == 2 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let m = meta.shape[0];
                    let q = if let Some(im) = im_slice {
                        let strat = imatrix::ImatrixFp4Scale { importance: im };
                        quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2, &strat)
                    } else {
                        quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2, &MinMaxScale)
                    };
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                } else {
                    // Fallback to HFQ4-G128 for non-256-aligned ragged dims (rotation
                    // requires 256-element segments). Matches MQ4's ragged fallback.
                    let q = quantize_hfq4g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                }
            } else if use_mq6g256 && is_embed {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_mq6g256 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                } else {
                    // Fallback to HFQ6-G256 for non-256-aligned (no rotation)
                    let q = quantize_hfq6g256(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
            } else if (use_mq3g256 || use_mq2g256 || use_mq2g256_lloyd || use_mq3g256_lloyd) && is_embed {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_mq3g256_lloyd {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                } else {
                    let q = quantize_hfq3g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                }
            } else if use_mq2g256_lloyd {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                } else {
                    // Fallback to HFQ2-G128 for non-256-aligned (no rotation)
                    let q = quantize_hfq2g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                }
            } else if use_mq3g256 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                } else {
                    // Fallback to HFQ3-G128 for non-256-aligned (no rotation)
                    let q = quantize_hfq3g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                }
            } else if use_mq2g256 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq2g256(&f32_data, &signs1, &signs2, &MinMaxScale);
                    (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                } else {
                    // Fallback to HFQ2-G128 for non-256-aligned (no rotation)
                    let q = quantize_hfq2g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                }
            } else if (use_hfq3g256 || use_hfq3g128) && is_embed {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_hfq3g128 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 128 == 0 {
                    let q = quantize_hfq3g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                } else {
                    let q = quantize_hfq3g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                }
            } else if use_hfq3g256 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let q = quantize_hfq3g256(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ3G256, 256u32, "HFQ3G256")
                } else {
                    let q = quantize_hfq3g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                }
            } else if use_hfq4g256 && is_embed {
                // HFQ4 embeddings: half the size of Q8, same 18-VGPR lookup kernel
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let q = quantize_hfq4g256(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                } else {
                    let q = quantize_hfq4g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                }
            } else if use_hfq4g256 {
                // Auto-select G128 vs G256 based on K dimension
                // G256 preferred: better coalescing, fewer scale/zero overheads
                // G128 only as fallback when K isn't divisible by 256
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let q = quantize_hfq4g256(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                } else if k_dim % 128 == 0 {
                    let q = quantize_hfq4g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                } else {
                    // Pad to 128-element boundary
                    let q = quantize_hfq4g128(&f32_data, &MinMaxScale);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                }
            } else if this_q8 {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_FP16")
            } else if this_q4as8 {
                let q = quantize_q4_as_q8(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q4asQ8")
            } else if this_q4k {
                let q = quantize_q4k(&f32_data);
                (q, QuantType::Q4K, 256u32, "Q4_K")
            } else {
                let q = quantize_q4f16_g64(&f32_data);
                (q, QuantType::Q4F16G64, 64u32, "Q4_F16")
            }
            }; // end K-map outer if-else

            // Compute quantization error (skip for Q8 embeddings — always negligible)
            let block_size = gs as usize;
            let is_hfq4 = label == "HFQ4G256" || label == "HFQ4G128";
            // Only compute detailed error for HFQ4 tensors — Q8/HFQ6 error is negligible
            let skip_error = !is_hfq4;
            let n_blocks = if !skip_error { (n_elements + block_size - 1) / block_size } else { 0 };
            for b in 0..n_blocks {
                let start = b * block_size;
                let end = (start + block_size).min(n_elements);
                if is_hfq4 {
                    // Both G128 (72B) and G256 (136B): [f32 scale][f32 zero][nibbles]
                    let block_bytes = if block_size == 256 { 136 } else { 72 };
                    let off = b * block_bytes;
                    let scale = f32::from_le_bytes([quantized[off], quantized[off+1], quantized[off+2], quantized[off+3]]);
                    let zero = f32::from_le_bytes([quantized[off+4], quantized[off+5], quantized[off+6], quantized[off+7]]);
                    for i in 0..(end - start) {
                        let byte_idx = i / 2;
                        let nibble = if i % 2 == 0 { quantized[off + 8 + byte_idx] & 0xF } else { quantized[off + 8 + byte_idx] >> 4 };
                        let dequant = scale * nibble as f32 + zero;
                        let err = (dequant - f32_data[start + i]).abs();
                        total_quant_error += err as f64;
                        max_quant_error = max_quant_error.max(err);
                    }
                } else if label == "Q8_FP16" || label == "Q4asQ8" || label == "Q8_F16" {
                    // NB: string match because this_q8/this_q4as8 are scoped inside Base block.
                    let off = b * 34;
                    let scale = f16_to_f32(u16::from_le_bytes([quantized[off], quantized[off + 1]]));
                    for i in 0..(end - start) {
                        let qval = quantized[off + 2 + i] as i8;
                        let dequant = scale * qval as f32;
                        let err = (dequant - f32_data[start + i]).abs();
                        total_quant_error += err as f64;
                        max_quant_error = max_quant_error.max(err);
                    }
                } else {
                    let off = b * 36;
                    let scale = f16_to_f32(u16::from_le_bytes([quantized[off], quantized[off + 1]]));
                    let min_val = f16_to_f32(u16::from_le_bytes([quantized[off + 2], quantized[off + 3]]));
                    for i in 0..(end - start) {
                        let byte_idx = if i < 32 { i } else { i - 32 };
                        let nibble = if i < 32 {
                            quantized[off + 4 + byte_idx] & 0xF
                        } else {
                            quantized[off + 4 + byte_idx] >> 4
                        };
                        let dequant = nibble as f32 * scale + min_val;
                        let err = (dequant - f32_data[start + i]).abs();
                        total_quant_error += err as f64;
                        max_quant_error = max_quant_error.max(err);
                    }
                }
                _n_quant_groups += 1;
            }

            eprintln!("  {label:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB)",
                name, meta.shape, n_elements,
                raw_data.len() as f64 / 1024.0,
                quantized.len() as f64 / 1024.0);

            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: qt,
                shape,
                group_size: gs,
                data: quantized,
                spilled_len: 0,
            });
            } // end else (non-Q8HFQ path)
        } else if is_vision && vision_quant == "hfq4" && n_elements >= 32 {
            // Quantize vision weights to HFQ4G256 (for speed-critical VL workloads)
            let f32_data = to_f32(raw_data, &meta.dtype);
            quantized_params += n_elements as u64;
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let k_dim = if shape.len() == 2 { shape[1] as usize } else { n_elements };
            let (quantized, gs) = if k_dim % 256 == 0 {
                (quantize_hfq4g256(&f32_data, &MinMaxScale), 256u32)
            } else {
                (quantize_hfq4g128(&f32_data, &MinMaxScale), 128u32)
            };
            let qt = if gs == 256 { QuantType::HFQ4G256 } else { QuantType::HFQ4G128 };
            let label = if gs == 256 { "HFQ4G256" } else { "HFQ4G128" };
            eprintln!("  {label:>8}: {} {:?} ({} elements, {:.1} KB -> {:.1} KB) [vision]",
                name, meta.shape, n_elements,
                raw_data.len() as f64 / 1024.0, quantized.len() as f64 / 1024.0);
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: qt,
                shape,
                group_size: gs,
                data: quantized,
                spilled_len: 0,
            });
        } else if is_vision && vision_quant == "bf16" && meta.dtype == "BF16" {
            // Store vision weights as original BF16 (zero precision loss)
            quantized_params += n_elements as u64;
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            eprintln!("  BF16:       {} {:?} ({} elements, {:.1} KB) [vision, lossless]",
                name, meta.shape, n_elements, raw_data.len() as f64 / 1024.0);
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::BF16,
                shape,
                group_size: 0,
                data: raw_data.to_vec(),
                spilled_len: 0,
            });
        } else if is_vision && vision_quant == "bf16" {
            // Non-BF16 source (F16/F32) — store as F16
            let data = if meta.dtype == "F16" { raw_data.to_vec() } else {
                let f32_vals = to_f32(raw_data, &meta.dtype);
                f32_vals.iter().flat_map(|&v| f32_to_f16(v).to_le_bytes()).collect()
            };
            quantized_params += n_elements as u64;
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            eprintln!("  F16:        {} {:?} ({:.1} KB) [vision, bf16 fallback]",
                name, meta.shape, data.len() as f64 / 1024.0);
            hfq_tensors.push(HfqTensor {
                name: name.to_string(), quant_type: QuantType::F16,
                shape, group_size: 0, data, spilled_len: 0,
            });
        } else {
            // Keep as F16 (convert BF16 -> F16 if needed)
            let f16_data = match meta.dtype.as_str() {
                "F16" => raw_data.to_vec(),
                "BF16" => {
                    // BF16 → F32 → F16
                    let f32_vals = to_f32(raw_data, "BF16");
                    f32_vals.iter()
                        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                        .collect()
                }
                "F32" => {
                    let f32_vals = to_f32(raw_data, "F32");
                    f32_vals.iter()
                        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                        .collect()
                }
                other => panic!("unsupported dtype for norm/embd: {other}"),
            };

            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            eprintln!("  F16:        {} {:?} ({} elements, {:.1} KB)",
                name, meta.shape, n_elements, f16_data.len() as f64 / 1024.0);

            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F16,
                shape,
                group_size: 0,
                data: f16_data,
                spilled_len: 0,
            });
        }
        // Release source file page cache after each tensor to prevent
        // mmap'd pages from starving GPU allocations on UMA systems.
        st_files[*file_idx].drop_tensor_pages(name);
    }

    // Summary
    let total_bytes: usize = hfq_tensors.iter().map(|t| if t.spilled_len > 0 { t.spilled_len as usize } else { t.data.len() }).sum();
    let mean_quant_error = if quantized_params > 0 {
        total_quant_error / quantized_params as f64
    } else { 0.0 };

    eprintln!("\n=== Quantization Summary ===");
    if skipped_params > 0 {
        eprintln!("  Skipped params:   {skipped_params} (mtp/visual — use --include-vision for VL)");
    }
    eprintln!("  Total params:     {total_params}");
    eprintln!("  Quantized params: {quantized_params} ({:.1}%)", 100.0 * quantized_params as f64 / total_params as f64);
    eprintln!("  Mean quant error: {mean_quant_error:.8}");
    eprintln!("  Max quant error:  {max_quant_error:.8}");
    eprintln!("  Output size:      {:.1} MB", total_bytes as f64 / 1e6);

    // Write .hfq file
    eprintln!("\nWriting: {}", output_path.display());
    // Final spill before writing
    if let Some(ref mut s) = spill {
        maybe_spill(&mut hfq_tensors, s, 0); // spill everything remaining
    }
    write_hfq(output_path, arch_id, &metadata_json, &hfq_tensors, spill.as_mut()).unwrap();
    if let Some(s) = spill { s.cleanup(); }

    let file_size = std::fs::metadata(output_path).unwrap().len();
    eprintln!("Done: {:.1} MB written", file_size as f64 / 1e6);
}

// K-map and strategy tests live in strategy.rs
