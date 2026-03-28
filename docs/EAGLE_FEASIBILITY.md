# EAGLE-3 Speculative Decode — Feasibility Assessment

## Hardware: RX 5700 XT, Qwen3-8B HFQ4-G256

## Key Measurement

Sequential 7-token generation: 117ms (16.7ms each)
This means naive "generate k draft + verify sequentially" is SLOWER:
  117ms verify + 1ms draft = 118ms for ~3.5 accepted tokens
  = 29.7 tok/s (WORSE than 59.8 baseline)

## Why Batched Verification is Critical

The speedup comes from reading weights ONCE for all 7 verification tokens.
Weight GEMV is 93% of forward time. Batching saves 6/7 of weight reads.

Batched verification estimate:
  Batched GEMV (weights read once): 15.2ms  
  Sequential attention × 7: ~7ms (attention is only 7% of forward)
  Draft head: ~1ms
  Total cycle: ~23ms for ~3.5 accepted tokens
  Effective: **152 tok/s** (2.5× speedup)

## Infrastructure Status

| Component | Status | Notes |
|-----------|--------|-------|
| Batched GEMM kernel | ✅ EXISTS | gemm_hfq4g256, BATCH_TILE=8, 32 VGPRs |
| Batched RoPE | ✅ EXISTS | rope_batched_f32 |
| Batched KV write (Q8) | ✅ EXISTS | kv_cache_write_q8_0_batched |
| Batched causal attention | ⚠️ PARTIAL | Works for prefill (fresh KV), needs modification for EAGLE (existing KV + new tokens) |
| EAGLE draft head weights | ❓ UNKNOWN | Searching HuggingFace |
| forward_verify() function | ❌ MISSING | Needs: batched GEMM + per-token attention against full cache |
| Speculative loop | ❌ MISSING | Draft → verify → accept/reject |

## What Needs to Be Built

### 1. forward_verify(tokens: &[u32], start_pos: usize) → Vec<Vec<f32>>
- Use weight_gemm for all projection layers (batched, weights read once)
- Use per-token attention against existing KV cache (sequential, small cost)
- Write new K/V to cache at positions start_pos..start_pos+k
- Return logits for each verification position

### 2. EAGLE draft head
- One transformer block (~0.5 GB at FP16)
- Takes: base model hidden state from last layer
- Produces: draft token predictions
- Needs pretrained weights (not trainable in hipfire)

### 3. Speculative loop
- Standard EAGLE-3 algorithm
- Tree-structured drafting is optional (linear is simpler, ~80% of benefit)

## Blocker: EAGLE Head Weights
Without pretrained EAGLE weights for Qwen3-8B, we can't test the full system.
Alternative: Lookahead decoding (no draft model needed, ~1.7× speedup).

## VGPR Verification
- Batched GEMM (gemm_hfq4g256): 32 VGPRs, 0 spills ✅
- Single-token attention (attention_q8_0_kv): 39 VGPRs, 0 spills ✅
- Neither is a blocker for EAGLE feasibility
