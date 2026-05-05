# Quality Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a quality gate that measures inference quality of hipfire's quantized models against golden reference logits from llama.cpp.

**Architecture:** Shell script orchestrates llama-perplexity (golden generation + GGUF KL) and a new hipfire `logit_dump` binary (teacher-forced logit capture in llama.cpp format). A Python comparison script computes token agreement, top-k overlap, KL divergence, and perplexity delta, outputting JSON with pass/fail against configurable thresholds.

**Tech Stack:** Rust (logit_dump binary), Python 3 + numpy (metrics), Bash (orchestration), llama-perplexity (golden + GGUF path)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/hipfire-runtime/examples/logit_dump.rs` | **Replace existing.** Teacher-forced eval: tokenize corpus, run forward pass per token, write full logit vectors in llama.cpp binary format |
| `scripts/quality_metrics.py` | **Create.** Load two binary logit files, compute all metrics, output JSON |
| `scripts/quality-gate.sh` | **Create.** Orchestrate golden generation, GGUF KL, hipfire logit dump, Python metrics |
| `tests/quality-baselines/qwen35-9b-mq4.json` | **Create.** Example threshold file |
| `benchmarks/quality-goldens/.gitignore` | **Create.** Ignore golden .bin files |
| `benchmarks/quality-goldens/MANIFEST.md` | **Create.** Document golden file provenance |

---

### Task 1: Replace `logit_dump.rs` with Teacher-Forced Evaluation

**Files:**
- Replace: `crates/hipfire-runtime/examples/logit_dump.rs`

This is the core binary. It reads a corpus text file, tokenizes it, runs `forward_scratch` per token (teacher-forced — feeds ground-truth tokens, not model predictions), and writes the full logit vector at each position to the output file in llama.cpp's binary format.

- [ ] **Step 1: Write the new `logit_dump.rs`**

Replace `crates/hipfire-runtime/examples/logit_dump.rs` entirely:

```rust
//! Teacher-forced logit dump for quality-gate comparison against llama.cpp.
//!
//! Tokenizes a corpus, runs forward passes feeding ground-truth tokens,
//! and writes per-position logit vectors in llama.cpp's binary format:
//!   [n_vocab: u32][n_tokens: u32][logits: f32 × n_vocab × n_tokens]
//!
//! Usage: logit_dump <model.hfq> <output.bin> -f <corpus.txt> [--ctx-size 512] [--max-tokens N]

#[cfg(not(feature = "deltanet"))]
fn main() { eprintln!("build with --features deltanet"); }

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::llama::KvCache;
    use std::io::Write;
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();

    // Parse args
    let mut model_path: Option<&str> = None;
    let mut out_path: Option<&str> = None;
    let mut corpus_path: Option<&str> = None;
    let mut ctx_size: usize = 512;
    let mut max_tokens: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-f" => { i += 1; corpus_path = Some(&args[i]); }
            "--ctx-size" => { i += 1; ctx_size = args[i].parse().expect("--ctx-size must be a number"); }
            "--max-tokens" => { i += 1; max_tokens = Some(args[i].parse().expect("--max-tokens must be a number")); }
            _ if model_path.is_none() => { model_path = Some(&args[i]); }
            _ if out_path.is_none() => { out_path = Some(&args[i]); }
            other => { eprintln!("unknown arg: {other}"); std::process::exit(2); }
        }
        i += 1;
    }

    let model_path = model_path.expect("Usage: logit_dump <model.hfq> <output.bin> -f <corpus.txt>");
    let out_path = out_path.expect("Usage: logit_dump <model.hfq> <output.bin> -f <corpus.txt>");
    let corpus_path = corpus_path.expect("Usage: logit_dump <model.hfq> <output.bin> -f <corpus.txt>");

    // Read corpus
    let corpus_text = std::fs::read_to_string(corpus_path).expect("read corpus");
    eprintln!("logit_dump: corpus={corpus_path} ({} bytes)", corpus_text.len());

    // Load model
    let hfq = HfqFile::open(Path::new(model_path)).expect("open model");
    let config = qwen35::config_from_hfq(&hfq).expect("read config");
    let tokenizer = hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer");

    // Tokenize corpus
    let all_tokens = tokenizer.encode(&corpus_text);
    let n_eval = match max_tokens {
        Some(max) => max.min(all_tokens.len().saturating_sub(1)),
        None => all_tokens.len().saturating_sub(1),
    };
    eprintln!("logit_dump: {} corpus tokens, evaluating {} positions", all_tokens.len(), n_eval);
    eprintln!("logit_dump: vocab_size={}, ctx_size={ctx_size}", config.vocab_size);

    // Init GPU + model
    let mut gpu = rdna_compute::Gpu::init().expect("gpu init");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load weights");
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu, config.n_layers, config.n_kv_heads, config.head_dim, ctx_size,
    ).unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    // Open output and write header
    let mut out = std::io::BufWriter::new(std::fs::File::create(out_path).expect("create output"));
    let n_vocab = config.vocab_size as u32;
    let n_tokens_out = n_eval as u32;
    out.write_all(&n_vocab.to_le_bytes()).unwrap();
    out.write_all(&n_tokens_out.to_le_bytes()).unwrap();

    // Teacher-forced evaluation: process in chunks of ctx_size
    let mut tokens_written: usize = 0;
    let mut chunk_start: usize = 0;

    while tokens_written < n_eval {
        let chunk_end = (chunk_start + ctx_size).min(all_tokens.len());
        let chunk_tokens = &all_tokens[chunk_start..chunk_end];

        // Reset KV cache and DeltaNet state for each chunk
        kv_cache.reset();
        dn_state.reset(&mut gpu).unwrap();

        // Forward each token in the chunk, collect logits for positions we need
        for (pos_in_chunk, &token) in chunk_tokens.iter().enumerate() {
            qwen35::forward_scratch(
                &mut gpu, &weights, &config, token, pos_in_chunk,
                &mut kv_cache, &mut dn_state, &scratch,
            ).expect("forward failed");

            // The logits after feeding token[i] predict token[i+1].
            // We want logits for positions chunk_start..chunk_end-1 (predicting chunk_start+1..chunk_end).
            // Skip the last token in the chunk (it predicts beyond the chunk).
            // But we need to track absolute position for the output limit.
            let abs_pos = chunk_start + pos_in_chunk;
            if abs_pos < n_eval {
                let logits = gpu.download_f32(&scratch.logits).unwrap();
                // Write raw f32 logits (little-endian, which is native on x86/amd64)
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        logits.as_ptr() as *const u8,
                        logits.len() * 4,
                    )
                };
                out.write_all(bytes).unwrap();
                tokens_written += 1;
            }

            if tokens_written >= n_eval { break; }
        }

        chunk_start += ctx_size;

        if tokens_written % 500 == 0 || tokens_written == n_eval {
            eprintln!("  {tokens_written}/{n_eval} positions");
        }
    }

    out.flush().unwrap();
    let file_size = 8 + (n_eval * config.vocab_size * 4);
    eprintln!("logit_dump: wrote {out_path} ({} MB, {n_eval} positions × {} vocab)",
        file_size / (1024 * 1024), config.vocab_size);
}
```

- [ ] **Step 2: Verify it compiles**

Run:
```bash
cargo build --release --features deltanet -p hipfire-runtime --example logit_dump
```
Expected: compiles without errors.

- [ ] **Step 3: Smoke test with a small corpus**

Create a tiny test corpus and run:
```bash
echo "The quick brown fox jumps over the lazy dog." > /tmp/test_corpus.txt
./target/release/examples/logit_dump ~/.hipfire/models/qwen3.5-0.8b.mq4 /tmp/test_logits.bin \
    -f /tmp/test_corpus.txt --max-tokens 5
```
Expected: creates `/tmp/test_logits.bin`, stderr shows progress, file size = 8 + 5 × vocab_size × 4.

Verify header:
```bash
python3 -c "
import struct
with open('/tmp/test_logits.bin', 'rb') as f:
    n_vocab, n_tokens = struct.unpack('<II', f.read(8))
    print(f'n_vocab={n_vocab}, n_tokens={n_tokens}')
    import os
    expected = 8 + n_vocab * n_tokens * 4
    actual = os.path.getsize('/tmp/test_logits.bin')
    print(f'expected_size={expected}, actual_size={actual}')
    assert expected == actual, 'SIZE MISMATCH'
    print('OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add crates/hipfire-runtime/examples/logit_dump.rs
git commit -m "feat(quality-gate): replace logit_dump with teacher-forced eval binary

Writes full logit vectors per position in llama.cpp's binary format
for quality-gate comparison. Processes corpus in ctx-size chunks with
per-token forward passes (teacher-forced, not autoregressive)."
```

---

### Task 2: Create `quality_metrics.py`

**Files:**
- Create: `scripts/quality_metrics.py`

- [ ] **Step 1: Write `scripts/quality_metrics.py`**

```python
#!/usr/bin/env nix-shell
#!nix-shell -i python3 -p python3Packages.numpy
"""
Quality metrics: compare two llama.cpp-format logit binary files.

Computes: token agreement, top-k overlap, KL divergence, perplexity delta.
Outputs JSON to stdout or --output file.

Usage:
    python3 scripts/quality_metrics.py <golden.bin> <test.bin> [--top-k 10] [--thresholds FILE] [--output FILE]
"""
import argparse
import json
import sys
import struct
import numpy as np
from pathlib import Path


def load_logits(path: str) -> tuple[int, int, np.memmap]:
    """Load a llama.cpp binary logits file as a memory-mapped numpy array."""
    with open(path, 'rb') as f:
        n_vocab, n_tokens = struct.unpack('<II', f.read(8))
    # Memory-map the logits portion (offset 8 bytes)
    logits = np.memmap(path, dtype=np.float32, mode='r', offset=8,
                       shape=(n_tokens, n_vocab))
    return n_vocab, n_tokens, logits


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def compute_metrics(golden_path: str, test_path: str, top_k: int) -> dict:
    n_vocab_g, n_tokens_g, golden = load_logits(golden_path)
    n_vocab_t, n_tokens_t, test = load_logits(test_path)

    if n_vocab_g != n_vocab_t:
        raise ValueError(f"Vocab mismatch: golden={n_vocab_g}, test={n_vocab_t}")
    if n_tokens_g != n_tokens_t:
        raise ValueError(f"Token count mismatch: golden={n_tokens_g}, test={n_tokens_t}")

    n_vocab = n_vocab_g
    n_tokens = n_tokens_g

    # Process in batches to avoid loading everything into RAM at once
    batch_size = 256
    n_batches = (n_tokens + batch_size - 1) // batch_size

    agree_count = 0
    jaccard_sum = 0.0
    rank_delta_sum = 0.0
    kl_values = []
    golden_log_probs = []
    test_log_probs = []

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n_tokens)
        g_batch = np.array(golden[start:end])  # force load from mmap
        t_batch = np.array(test[start:end])

        # Token agreement
        g_argmax = g_batch.argmax(axis=-1)
        t_argmax = t_batch.argmax(axis=-1)
        agree_count += int((g_argmax == t_argmax).sum())

        # Top-k overlap (Jaccard)
        g_topk = np.argpartition(g_batch, -top_k, axis=-1)[:, -top_k:]
        t_topk = np.argpartition(t_batch, -top_k, axis=-1)[:, -top_k:]
        for i in range(end - start):
            g_set = set(g_topk[i])
            t_set = set(t_topk[i])
            jaccard_sum += len(g_set & t_set) / len(g_set | t_set)

        # Rank delta: where does golden's argmax land in test's ranking?
        for i in range(end - start):
            g_choice = g_argmax[i]
            # Rank = how many test logits are >= the test logit at golden's choice
            rank = int((t_batch[i] >= t_batch[i, g_choice]).sum()) - 1  # 0-indexed
            rank_delta_sum += rank

        # KL divergence: KL(golden || test) per position
        g_probs = softmax(g_batch)
        t_probs = softmax(t_batch)
        # Clip to avoid log(0)
        g_probs = np.clip(g_probs, 1e-10, None)
        t_probs = np.clip(t_probs, 1e-10, None)
        kl_per_pos = (g_probs * (np.log(g_probs) - np.log(t_probs))).sum(axis=-1)
        kl_values.extend(kl_per_pos.tolist())

        # Perplexity: log prob of the "next token" (golden argmax) under each distribution
        # For perplexity we use log(softmax(logit[argmax]))
        g_log_probs = np.log(g_probs)
        t_log_probs = np.log(t_probs)
        for i in range(end - start):
            golden_log_probs.append(float(g_log_probs[i, g_argmax[i]]))
            test_log_probs.append(float(t_log_probs[i, g_argmax[i]]))

    # Aggregate
    kl_arr = np.array(kl_values)
    golden_ppl = float(np.exp(-np.mean(golden_log_probs)))
    test_ppl = float(np.exp(-np.mean(test_log_probs)))

    return {
        "n_vocab": n_vocab,
        "n_tokens": n_tokens,
        "token_agreement": agree_count / n_tokens,
        "top_k_overlap": {"k": top_k, "mean_jaccard": jaccard_sum / n_tokens},
        "mean_rank_delta": rank_delta_sum / n_tokens,
        "kl_divergence": {
            "mean": float(kl_arr.mean()),
            "max": float(kl_arr.max()),
            "p95": float(np.percentile(kl_arr, 95)),
        },
        "perplexity": {
            "golden": golden_ppl,
            "test": test_ppl,
            "delta": test_ppl - golden_ppl,
        },
    }


def check_thresholds(metrics: dict, thresholds_file: str) -> bool:
    """Check metrics against thresholds. Returns True if all pass."""
    with open(thresholds_file) as f:
        config = json.load(f)
    thresholds = config["thresholds"]
    passed = True
    if metrics["token_agreement"] < thresholds.get("token_agreement_min", 0):
        passed = False
    if metrics["top_k_overlap"]["mean_jaccard"] < thresholds.get("top_k_overlap_min", 0):
        passed = False
    if metrics["kl_divergence"]["mean"] > thresholds.get("mean_kl_max", float("inf")):
        passed = False
    if metrics["kl_divergence"]["max"] > thresholds.get("max_kl_max", float("inf")):
        passed = False
    if metrics["perplexity"]["delta"] > thresholds.get("perplexity_delta_max", float("inf")):
        passed = False
    return passed


def main():
    parser = argparse.ArgumentParser(description="Compare two logit binary files")
    parser.add_argument("golden", help="Path to golden logits .bin")
    parser.add_argument("test", help="Path to test logits .bin")
    parser.add_argument("--top-k", type=int, default=10, help="K for top-k overlap (default: 10)")
    parser.add_argument("--thresholds", help="Path to thresholds JSON file")
    parser.add_argument("--output", "-o", help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    metrics = compute_metrics(args.golden, args.test, args.top_k)

    if args.thresholds:
        metrics["pass"] = check_thresholds(metrics, args.thresholds)
        metrics["thresholds_file"] = args.thresholds
    else:
        metrics["pass"] = None

    output = json.dumps(metrics, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(output)

    # Exit code reflects pass/fail
    if metrics["pass"] is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make executable and test with synthetic data**

```bash
chmod +x scripts/quality_metrics.py
```

Create two identical synthetic logit files and verify metrics show perfect agreement:

```bash
nix-shell -p python3Packages.numpy --run "python3 -c \"
import numpy as np, struct
n_vocab, n_tokens = 100, 10
logits = np.random.randn(n_tokens, n_vocab).astype(np.float32)
with open('/tmp/golden_test.bin', 'wb') as f:
    f.write(struct.pack('<II', n_vocab, n_tokens))
    f.write(logits.tobytes())
with open('/tmp/test_test.bin', 'wb') as f:
    f.write(struct.pack('<II', n_vocab, n_tokens))
    f.write(logits.tobytes())
print('wrote identical test files')
\""
```

Run comparison:
```bash
nix-shell -p python3Packages.numpy --run "python3 scripts/quality_metrics.py /tmp/golden_test.bin /tmp/test_test.bin"
```
Expected: `token_agreement: 1.0`, `mean_jaccard: 1.0`, `mean_rank_delta: 0.0`, `kl_divergence.mean: 0.0`.

- [ ] **Step 3: Test with divergent data**

```bash
nix-shell -p python3Packages.numpy --run "python3 -c \"
import numpy as np, struct
n_vocab, n_tokens = 100, 10
golden = np.random.randn(n_tokens, n_vocab).astype(np.float32)
# Add noise to create divergence
test = golden + np.random.randn(n_tokens, n_vocab).astype(np.float32) * 0.5
with open('/tmp/golden_div.bin', 'wb') as f:
    f.write(struct.pack('<II', n_vocab, n_tokens))
    f.write(golden.tobytes())
with open('/tmp/test_div.bin', 'wb') as f:
    f.write(struct.pack('<II', n_vocab, n_tokens))
    f.write(test.tobytes())
print('wrote divergent test files')
\""
nix-shell -p python3Packages.numpy --run "python3 scripts/quality_metrics.py /tmp/golden_div.bin /tmp/test_div.bin"
```
Expected: `token_agreement < 1.0`, `kl_divergence.mean > 0`, `mean_rank_delta > 0`.

- [ ] **Step 4: Test threshold checking**

```bash
cat > /tmp/test_thresholds.json << 'EOF'
{
  "model": "test",
  "format": "test",
  "thresholds": {
    "token_agreement_min": 0.99,
    "mean_kl_max": 0.001
  }
}
EOF
nix-shell -p python3Packages.numpy --run "python3 scripts/quality_metrics.py /tmp/golden_div.bin /tmp/test_div.bin --thresholds /tmp/test_thresholds.json -o /tmp/results.json"
echo "exit code: $?"
cat /tmp/results.json | python3 -c "import sys,json; print('pass:', json.load(sys.stdin)['pass'])"
```
Expected: exit code 1, `pass: False` (divergent data fails strict thresholds).

- [ ] **Step 5: Commit**

```bash
git add scripts/quality_metrics.py
git commit -m "feat(quality-gate): add quality_metrics.py comparison tool

Computes token agreement, top-k Jaccard overlap, KL divergence (mean/max/p95),
rank delta, and perplexity delta between two llama.cpp-format logit files.
Outputs JSON with optional pass/fail against threshold baselines."
```

---

### Task 3: Create `quality-gate.sh` Orchestration Script

**Files:**
- Create: `scripts/quality-gate.sh`

- [ ] **Step 1: Write `scripts/quality-gate.sh`**

```bash
#!/usr/bin/env bash
# Quality gate: measure inference quality of quantized models against golden reference.
#
# Orchestrates:
#   Phase 1: Golden generation via llama-perplexity (if --generate-golden)
#   Phase 2: GGUF KL via llama-perplexity (if --gguf-quantized provided)
#   Phase 3: Hipfire logit dump
#   Phase 4: Full metric comparison via quality_metrics.py
#
# Exit codes:
#   0  all metrics within thresholds (PASS)
#   1  one or more metrics exceeded thresholds (FAIL)
#   2  build or environment error
#
# Usage:
#   ./scripts/quality-gate.sh --model qwen3.5-9b --generate-golden
#   ./scripts/quality-gate.sh --model qwen3.5-9b
#   ./scripts/quality-gate.sh --help

set -u
cd "$(dirname "$0")/.."

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="qwen3.5-9b"
CORPUS="benchmarks/calib/calib-1m.txt"
GOLDEN=""
GENERATE_GOLDEN=0
GGUF_QUANTIZED=""
GGUF_FP16=""
CTX_SIZE=512
MAX_TOKENS=10000
THRESHOLDS=""
HFQ_MODEL=""
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"

# ── Parse args ────────────────────────────────────────────────────────────────
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift ;;
        --corpus) CORPUS="$2"; shift ;;
        --golden) GOLDEN="$2"; shift ;;
        --generate-golden) GENERATE_GOLDEN=1 ;;
        --gguf-quantized) GGUF_QUANTIZED="$2"; shift ;;
        --gguf-fp16) GGUF_FP16="$2"; shift ;;
        --hfq) HFQ_MODEL="$2"; shift ;;
        --ctx-size) CTX_SIZE="$2"; shift ;;
        --max-tokens) MAX_TOKENS="$2"; shift ;;
        --thresholds) THRESHOLDS="$2"; shift ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

# ── Resolve paths ─────────────────────────────────────────────────────────────
GOLDEN_DIR="benchmarks/quality-goldens"
mkdir -p "$GOLDEN_DIR"

if [ -z "$GOLDEN" ]; then
    GOLDEN="$GOLDEN_DIR/${MODEL}-fp16.bin"
fi
if [ -z "$HFQ_MODEL" ]; then
    # Try common patterns
    for pattern in "$MODELS_DIR/${MODEL}.mq4" "$MODELS_DIR/${MODEL//-/.}.mq4"; do
        if [ -f "$pattern" ]; then HFQ_MODEL="$pattern"; break; fi
    done
fi
if [ -z "$THRESHOLDS" ]; then
    THRESHOLDS="tests/quality-baselines/${MODEL}-mq4.json"
fi

WORKDIR="${HIPFIRE_QUALITY_OUT:-/tmp/quality-gate-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$WORKDIR"
echo "quality-gate: workdir=$WORKDIR"
echo "quality-gate: model=$MODEL corpus=$CORPUS ctx=$CTX_SIZE max_tokens=$MAX_TOKENS"

# ── Phase 1: Golden generation ────────────────────────────────────────────────
if [ "$GENERATE_GOLDEN" -eq 1 ]; then
    if [ -z "$GGUF_FP16" ]; then
        echo "ERROR: --generate-golden requires --gguf-fp16 <path>" >&2
        exit 2
    fi
    if ! command -v llama-perplexity &>/dev/null; then
        echo "ERROR: llama-perplexity not found in PATH" >&2
        exit 2
    fi
    echo "quality-gate: Phase 1 — generating golden from $GGUF_FP16"
    llama-perplexity -m "$GGUF_FP16" -f "$CORPUS" \
        --save-all-logits "$GOLDEN" -c "$CTX_SIZE" -n "$MAX_TOKENS" 2>&1 | \
        tail -5
    if [ ! -f "$GOLDEN" ]; then
        echo "ERROR: golden generation failed — $GOLDEN not created" >&2
        exit 2
    fi
    echo "quality-gate: golden written to $GOLDEN"
fi

# ── Phase 2: GGUF KL (optional) ──────────────────────────────────────────────
if [ -n "$GGUF_QUANTIZED" ]; then
    if [ ! -f "$GOLDEN" ]; then
        echo "ERROR: golden file not found: $GOLDEN (run with --generate-golden first)" >&2
        exit 2
    fi
    echo "quality-gate: Phase 2 — GGUF KL divergence"
    llama-perplexity -m "$GGUF_QUANTIZED" -f "$CORPUS" \
        --kl-divergence --kl-divergence-base "$GOLDEN" \
        -c "$CTX_SIZE" -n "$MAX_TOKENS" 2>&1 | tee "$WORKDIR/gguf-kl.txt" | tail -10
fi

# ── Phase 3: Hipfire logit dump ───────────────────────────────────────────────
EXE="./target/release/examples/logit_dump"

# Rebuild if needed
if [ ! -x "$EXE" ] || [ "crates/hipfire-runtime/examples/logit_dump.rs" -nt "$EXE" ]; then
    echo "quality-gate: rebuilding logit_dump..."
    if ! cargo build --release --features deltanet -p hipfire-runtime --example logit_dump >&2; then
        echo "ERROR: build failed" >&2
        exit 2
    fi
fi

if [ -z "$HFQ_MODEL" ]; then
    echo "ERROR: could not find HFQ model for $MODEL — use --hfq <path>" >&2
    exit 2
fi

echo "quality-gate: Phase 3 — hipfire logit dump ($HFQ_MODEL)"
"$EXE" "$HFQ_MODEL" "$WORKDIR/hipfire.bin" -f "$CORPUS" \
    --ctx-size "$CTX_SIZE" --max-tokens "$MAX_TOKENS"

if [ ! -f "$WORKDIR/hipfire.bin" ]; then
    echo "ERROR: logit_dump did not produce output" >&2
    exit 2
fi

# ── Phase 4: Metrics comparison ───────────────────────────────────────────────
if [ ! -f "$GOLDEN" ]; then
    echo "ERROR: golden file not found: $GOLDEN" >&2
    echo "Run with --generate-golden --gguf-fp16 <path> first." >&2
    exit 2
fi

echo "quality-gate: Phase 4 — computing metrics"
METRICS_CMD="python3 scripts/quality_metrics.py \"$GOLDEN\" \"$WORKDIR/hipfire.bin\" -o \"$WORKDIR/results.json\""
if [ -n "$THRESHOLDS" ] && [ -f "$THRESHOLDS" ]; then
    METRICS_CMD="$METRICS_CMD --thresholds \"$THRESHOLDS\""
fi

if nix-shell -p python3Packages.numpy --run "$METRICS_CMD"; then
    echo "quality-gate: PASS"
    echo "results: $WORKDIR/results.json"
    cat "$WORKDIR/results.json"
    exit 0
else
    EC=$?
    if [ $EC -eq 1 ]; then
        echo "quality-gate: FAIL (threshold exceeded)"
        cat "$WORKDIR/results.json"
        exit 1
    else
        echo "quality-gate: ERROR in metrics computation" >&2
        exit 2
    fi
fi
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/quality-gate.sh
```

- [ ] **Step 3: Test --help**

```bash
./scripts/quality-gate.sh --help
```
Expected: prints usage text.

- [ ] **Step 4: Commit**

```bash
git add scripts/quality-gate.sh
git commit -m "feat(quality-gate): add orchestration script

Orchestrates golden generation (llama-perplexity), optional GGUF KL path,
hipfire logit dump, and Python metrics comparison. Outputs JSON results
with pass/fail exit codes."
```

---

### Task 4: Create Threshold Baseline and Supporting Files

**Files:**
- Create: `tests/quality-baselines/qwen35-9b-mq4.json`
- Create: `benchmarks/quality-goldens/.gitignore`
- Create: `benchmarks/quality-goldens/MANIFEST.md`

- [ ] **Step 1: Create threshold baseline**

```bash
mkdir -p tests/quality-baselines
```

Write `tests/quality-baselines/qwen35-9b-mq4.json`:

```json
{
  "model": "qwen3.5-9b",
  "format": "mq4",
  "corpus": "calib-1m.txt",
  "ctx_size": 512,
  "max_tokens": 10000,
  "thresholds": {
    "token_agreement_min": 0.90,
    "top_k_overlap_min": 0.80,
    "mean_kl_max": 0.05,
    "max_kl_max": 2.0,
    "perplexity_delta_max": 0.5
  }
}
```

- [ ] **Step 2: Create golden directory with gitignore**

```bash
mkdir -p benchmarks/quality-goldens
```

Write `benchmarks/quality-goldens/.gitignore`:

```
# Golden logit files are large (100MB+) — generated on demand
*.bin
```

- [ ] **Step 3: Create MANIFEST.md**

Write `benchmarks/quality-goldens/MANIFEST.md`:

```markdown
# Golden Logit Files

Generated by `llama-perplexity --save-all-logits` from FP16 GGUF models.
These files are gitignored (too large to commit). Regenerate with:

    ./scripts/quality-gate.sh --generate-golden --gguf-fp16 <path-to-fp16.gguf> --model <tag>

## Format

Binary: `[n_vocab: u32][n_tokens: u32][logits: f32 × n_vocab × n_tokens]`
(llama.cpp native format, little-endian)

## Registry

| File | Source GGUF | Corpus | ctx | max_tokens | md5 |
|------|------------|--------|-----|-----------|-----|
| (generate and record here) | | | | | |
```

- [ ] **Step 4: Commit**

```bash
git add tests/quality-baselines/qwen35-9b-mq4.json \
        benchmarks/quality-goldens/.gitignore \
        benchmarks/quality-goldens/MANIFEST.md
git commit -m "feat(quality-gate): add threshold baselines and golden file infrastructure

Adds example threshold file for qwen3.5-9b MQ4, gitignore for large golden
logit binaries, and manifest for tracking golden file provenance."
```

---

### Task 5: Integration Test — End-to-End Dry Run

**Files:** None new — validates the whole pipeline works together.

- [ ] **Step 1: Generate a small golden file using hipfire itself as reference**

For a quick integration test (no FP16 GGUF needed), generate "golden" from hipfire, then compare against itself (should get perfect scores):

```bash
./target/release/examples/logit_dump ~/.hipfire/models/qwen3.5-0.8b.mq4 /tmp/self-golden.bin \
    -f benchmarks/calib/calib-1m.txt --ctx-size 512 --max-tokens 100
cp /tmp/self-golden.bin /tmp/self-test.bin
```

- [ ] **Step 2: Run quality_metrics.py on identical files**

```bash
nix-shell -p python3Packages.numpy --run \
    "python3 scripts/quality_metrics.py /tmp/self-golden.bin /tmp/self-test.bin"
```
Expected: `token_agreement: 1.0`, `kl_divergence.mean: 0.0`, `pass: null`.

- [ ] **Step 3: Run quality-gate.sh with pre-generated golden**

```bash
./scripts/quality-gate.sh \
    --golden /tmp/self-golden.bin \
    --hfq ~/.hipfire/models/qwen3.5-0.8b.mq4 \
    --corpus benchmarks/calib/calib-1m.txt \
    --max-tokens 100 \
    --ctx-size 512
```
Expected: Phase 3 runs logit_dump, Phase 4 computes metrics, exit 0 (PASS — comparing model to itself).

- [ ] **Step 4: Verify file sizes are consistent**

```bash
python3 -c "
import struct, os
for f in ['/tmp/self-golden.bin']:
    with open(f, 'rb') as fh:
        nv, nt = struct.unpack('<II', fh.read(8))
    actual = os.path.getsize(f)
    expected = 8 + nv * nt * 4
    status = 'OK' if actual == expected else 'MISMATCH'
    print(f'{f}: n_vocab={nv} n_tokens={nt} size={actual} expected={expected} [{status}]')
"
```
Expected: all OK.

- [ ] **Step 5: Final commit — update PRD with implementation status**

No code changes. This step confirms the pipeline is operational.

```bash
git status  # should be clean after Task 4 commit
```

---

## Summary of Deliverables

| Task | Produces | Validates |
|------|----------|-----------|
| 1 | `logit_dump.rs` (teacher-forced) | Compiles, produces correct binary format |
| 2 | `quality_metrics.py` | Correct metrics on synthetic data |
| 3 | `quality-gate.sh` | Parses args, orchestrates phases |
| 4 | Threshold + golden infrastructure | Files in correct locations |
| 5 | Integration test | End-to-end pipeline works |
