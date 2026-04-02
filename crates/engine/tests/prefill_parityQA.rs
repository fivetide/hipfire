//! QA mirror for prefill parity.
//! This branch does not expose a compileable `qwen35::prefill_forward`, so the QA
//! variant validates full-prompt parity between `forward` and `forward_scratch`.
//! Explicitly ignored by default because it needs a real model and ROCm GPU.

#[cfg(feature = "deltanet")]
#[test]
#[ignore = "requires QWEN35_TEST_MODEL and ROCm GPU"]
fn sequential_matches_batched_prefill_qa() {
    use engine::hfq::HfqFile;
    use engine::qwen35;
    use std::path::Path;

    let model_path = std::env::var("QWEN35_TEST_MODEL")
        .expect("QWEN35_TEST_MODEL must point to a Qwen3.5 HFQ file");
    let hfq = HfqFile::open(Path::new(&model_path)).expect("failed to open HFQ");
    let config = qwen35::config_from_hfq(&hfq).expect("failed to parse config");
    let tokenizer = engine::tokenizer::Tokenizer::from_hfq_metadata(&hfq.metadata_json)
        .expect("tokenizer metadata missing");

    let mut gpu = rdna_compute::Gpu::init().expect("GPU init failed");
    let weights = qwen35::load_weights(&hfq, &config, &mut gpu).expect("load weights failed");

    let prompt = "The quick brown fox jumps over the lazy dog and then";
    let tokens = tokenizer.encode(prompt);
    assert!(tokens.len() >= 4, "prompt too short: {} tokens", tokens.len());

    let kv_seq_len = 2048usize;

    let mut kv_seq = engine::llama::KvCache::new_gpu(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_seq_len,
    ).expect("failed to allocate sequential KV cache");
    let mut dn_seq = qwen35::DeltaNetState::new_with_quant(
        &mut gpu,
        &config,
        qwen35::StateQuant::FP32,
    ).expect("failed to allocate sequential DeltaNet state");

    let mut logits_seq = Vec::new();
    for (pos, &tok) in tokens.iter().enumerate() {
        logits_seq = qwen35::forward(&mut gpu, &weights, &config, tok, pos, &mut kv_seq, &mut dn_seq)
            .expect("sequential forward failed");
    }

    let mut kv_scratch = engine::llama::KvCache::new_gpu(
        &mut gpu,
        config.n_layers,
        config.n_kv_heads,
        config.head_dim,
        kv_seq_len,
    ).expect("failed to allocate scratch KV cache");
    let mut dn_scratch = qwen35::DeltaNetState::new_with_quant(
        &mut gpu,
        &config,
        qwen35::StateQuant::FP32,
    ).expect("failed to allocate scratch DeltaNet state");
    let scratch = qwen35::Qwen35Scratch::new(&mut gpu, &config, 64)
        .expect("failed to allocate qwen35 scratch buffers");

    for (pos, &tok) in tokens.iter().enumerate() {
        qwen35::forward_scratch(
            &mut gpu,
            &weights,
            &config,
            tok,
            pos,
            &mut kv_scratch,
            &mut dn_scratch,
            &scratch,
        ).expect("forward_scratch prefill failed");
    }
    let logits_scratch = gpu.download_f32(&scratch.logits)
        .expect("failed to download scratch logits");

    assert_eq!(logits_seq.len(), logits_scratch.len(), "logit length mismatch");

    let max_abs_err = logits_seq.iter().zip(&logits_scratch)
        .map(|(a, b)| (*a - *b).abs())
        .fold(0.0f32, f32::max);

    assert!(max_abs_err < 1e-4,
        "prefill parity QA failed: max logit error {max_abs_err:.2e} exceeds 1e-4");
}
