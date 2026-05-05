//! Teacher-forced evaluation for quality gating.
//!
//! Feeds a reference token sequence through the model and measures how well
//! the model predicts each next token. Outputs per-position JSONL with
//! cross-entropy, top-1/top-5 accuracy, and logit statistics.
//!
//! Usage: teacher_eval <model.hfq> <reference.txt> [output.jsonl]
//!
//! The reference text is tokenized and run through the model position by
//! position. At each position i, the model sees token[i] and we check
//! whether its logits would predict token[i+1].
//!
//! Output JSONL format (one line per position):
//!   {"pos":0,"ref_token":1234,"top1":1234,"top1_correct":true,
//!    "top5_correct":true,"cross_entropy":0.123,"top1_logit":5.6,
//!    "ref_logit":5.6,"ref_rank":1}

#[cfg(not(feature = "deltanet"))]
fn main() {
    eprintln!("build with --features deltanet");
    std::process::exit(1);
}

#[cfg(feature = "deltanet")]
fn main() {
    use hipfire_arch_qwen35::qwen35::{self, DeltaNetState, Qwen35Scratch};
    use hipfire_runtime::hfq::HfqFile;
    use hipfire_runtime::llama::{self, KvCache};
    use std::io::Write;
    use std::path::Path;

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: teacher_eval <model.hfq> <reference.txt> [output.jsonl]");
        std::process::exit(1);
    }
    let model_path = &args[1];
    let reference_path = &args[2];
    let output_path = args.get(3).map(|s| s.as_str());

    let reference_text = std::fs::read_to_string(reference_path)
        .unwrap_or_else(|e| {
            eprintln!("Failed to read reference: {e}");
            std::process::exit(1);
        });

    let hfq = HfqFile::open(Path::new(model_path)).expect("failed to open model");
    let config = qwen35::config_from_hfq(&hfq).expect("failed to read config");
    let tokenizer =
        hipfire_runtime::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
            .expect("need tokenizer");

    let tokens = tokenizer.encode(&reference_text);
    let n_tokens = tokens.len();
    if n_tokens < 2 {
        eprintln!("Reference text too short ({n_tokens} tokens, need at least 2)");
        std::process::exit(1);
    }

    eprintln!("teacher_eval: model={model_path}");
    eprintln!("  reference={reference_path} ({n_tokens} tokens)");
    eprintln!("  config: dim={}, layers={}, heads={}", config.dim, config.n_layers, config.n_heads);

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    eprintln!("  gpu: {}", gpu.arch);

    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("failed to load weights");
    let max_seq = n_tokens.min(4096);
    let mut kv_cache = KvCache::new_gpu_q8(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        max_seq,
    )
    .unwrap();
    let mut dn_state = DeltaNetState::new(&mut gpu, &config).unwrap();
    let scratch = Qwen35Scratch::new(&mut gpu, &config, 128).unwrap();

    let mut output: Box<dyn Write> = match output_path {
        Some(path) => Box::new(
            std::fs::File::create(path)
                .unwrap_or_else(|e| panic!("Failed to create output file {path}: {e}")),
        ),
        None => Box::new(std::io::stdout().lock()),
    };

    let eval_len = (n_tokens - 1).min(max_seq - 1);
    eprintln!("  evaluating {eval_len} positions...");

    let mut total_ce = 0.0f64;
    let mut top1_hits = 0u64;
    let mut top5_hits = 0u64;
    let mut count = 0u64;

    for pos in 0..eval_len {
        let input_token = tokens[pos];
        let target_token = tokens[pos + 1];

        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            input_token,
            pos,
            &mut kv_cache,
            &mut dn_state,
            &scratch,
        )
        .unwrap_or_else(|e| {
            eprintln!("forward failed at pos {pos}: {e:?}");
            std::process::exit(1);
        });

        let logits = gpu.download_f32(&scratch.logits).unwrap();
        let vocab_size = logits.len();

        // Compute softmax cross-entropy for the target token
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f64 = logits.iter().map(|&l| ((l - max_logit) as f64).exp()).sum();
        let log_sum_exp = max_logit as f64 + sum_exp.ln();
        let target_logit = if (target_token as usize) < vocab_size {
            logits[target_token as usize]
        } else {
            f32::NEG_INFINITY
        };
        let ce = -(target_logit as f64 - log_sum_exp);

        // Top-1
        let top1 = llama::argmax(&logits);
        let top1_correct = top1 == target_token;

        // Top-5 + rank of target
        let mut indexed: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let ref_rank = indexed
            .iter()
            .position(|(id, _)| *id == target_token)
            .map(|r| r + 1)
            .unwrap_or(vocab_size);
        let top5_correct = ref_rank <= 5;

        total_ce += ce;
        if top1_correct {
            top1_hits += 1;
        }
        if top5_correct {
            top5_hits += 1;
        }
        count += 1;

        writeln!(
            output,
            "{{\"pos\":{pos},\"ref_token\":{target_token},\"top1\":{top1},\
             \"top1_correct\":{top1_correct},\"top5_correct\":{top5_correct},\
             \"cross_entropy\":{ce:.6},\"top1_logit\":{:.6},\"ref_logit\":{:.6},\
             \"ref_rank\":{ref_rank}}}",
            indexed[0].1, target_logit
        )
        .ok();

        if pos % 100 == 0 || pos == eval_len - 1 {
            let running_ce = total_ce / count as f64;
            let running_ppl = running_ce.exp();
            eprintln!(
                "  pos {:4}/{}: ce={:.4} ppl={:.2} top1={:.1}% top5={:.1}%",
                pos + 1,
                eval_len,
                running_ce,
                running_ppl,
                top1_hits as f64 / count as f64 * 100.0,
                top5_hits as f64 / count as f64 * 100.0,
            );
        }
    }

    output.flush().ok();

    // Print summary to stderr
    let mean_ce = total_ce / count as f64;
    let ppl = mean_ce.exp();
    let top1_acc = top1_hits as f64 / count as f64;
    let top5_acc = top5_hits as f64 / count as f64;

    eprintln!("\n=== Teacher-Forced Eval Summary ===");
    eprintln!("  tokens evaluated: {count}");
    eprintln!("  mean cross-entropy: {mean_ce:.6}");
    eprintln!("  perplexity: {ppl:.4}");
    eprintln!("  top-1 accuracy: {:.2}%", top1_acc * 100.0);
    eprintln!("  top-5 accuracy: {:.2}%", top5_acc * 100.0);

    // Also emit summary as final JSONL line with type=summary
    writeln!(
        std::io::stderr(),
        "QUALITY_SUMMARY: ce={mean_ce:.6} ppl={ppl:.4} top1={:.4} top5={:.4} n={count}",
        top1_acc, top5_acc
    )
    .ok();
}
