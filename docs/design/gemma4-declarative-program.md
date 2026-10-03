# Gemma 4 declarative layer program (#666 G6)

Status: in progress on `feat/gemma-declarative` (cut from #795
`feat/qwen35-declarative` at `d101a15d0`).

## Goal

Gemma 4 declares each decoder layer as one typed `Step` list, as Qwen3.5
(#795) and Qwen4 do. `hipfire-dispatch` owns every executor body and every
arch-, dtype- or tier-gated route choice. One program serves decode
(`rows == 1`), batched prefill and EAGLE verify (`rows > 1`), and the arch-22
EAGLE/MTP drafter.

The architecture crate keeps:

- weight and state ownership (one config, one weight set, KV allocation);
- the per-layer program shape (`src/program.rs`: which ops, in which order,
  bound to which tensors);
- lifecycle: hipGraph capture, the spec-decode driver, Redline shadow, the
  chunk loop.

The #397 super-op facade (`lowered.rs` `lower_variant` / `Gemma4Bindings`)
and the hand-written decode arms in `forward.rs` and `lowered.rs` are deleted.
Gemma 4 has no EP/TP route, so nothing keeps the super-op program alive for it.

## Today

Two parallel stacks, each with its own config, weights and loader:

| Stack | Fixtures | KV | Notes |
|---|---|---|---|
| eager (`config.rs`, `gemma4.rs`, `forward.rs`) | 12B dense (default), E2B/E4B, EAGLE target | Q8 both tiers, `physical_cap = max_seq` | fused qk-norm+RoPE and fused post-norm+residual; hipGraph decode; `forward_batch{,_spec}` B≤64 |
| lowered (`lowered.rs`) | 26B-A4B MoE (always); 12B with `HIPFIRE_{BATCHED,WMMA}_PREFILL=1` | sliding Q8 ring (`window`), full asym3 hd512 | super-op facade; per-token serve prefill; `forward_prefill_batch` only for calibration tools |

## Vocabulary

New model-neutral ops live in `hipfire-dispatch/src/pipeline/sandwich.rs`
(not `deltanet`-gated). Each executor takes `rows`.

| Step | Semantics | Route choice owned by dispatch |
|---|---|---|
| `SandwichAttention` | `x = residual + post_norm(o_proj(attend(rope(norm_h(q)), rope(norm_h(k)), norm(v))))` on `input_norm(x)`. Optional V projection (`None` = K=V from pre-norm K); optional weightless V norm; Q prescale; **explicit `Rope { kind: RotateHalf \| PartialHalved { rot_pairs }, theta }`**; KV `write: Option` (`None` = read-only shared slot, used by E-series KV sharing and the drafter); `window` | fused norm+FWHT-rotate + prerotated GEMVs (MQ4), fused Q8 q+k, fused qk-norm+RoPE, fused post-norm+residual, KV tier via `KvTierPlan` |
| `SandwichMlp` | `x = residual + post_norm(down(act(gate(n)) * up(n)))`, `n = pre_norm(x)`, `act ∈ {GeluTanh}` | fused norm+rotate + fused gate/up (MQ4), fused post-norm+residual |
| `ParallelMoeMlp` | Gemma 4 MoE block: dense GeGLU MLP ∥ routed experts (router input `rms(x)·router_scale/√dim`, softmax top-k renorm, per-expert scale, GeGLU), `post_ffn_norm(norm1(mlp) + norm2(moe))`, residual add | expert dtype routes (MQ4 / HFQ4 / Q8 indexed) |
| `PerLayerInput` | E-series PLE branch: `x += post_norm(proj(gelu(gate(x)) * ple[layer]))` | batched strided PLE kernel |
| `Scale` | `x *= s` (layer scalar), replay-safe recorded scale | — |
| `Softcap` | `logits = tanh(logits / c) · c` | — |

Projections, the final norm and the LM head reuse `Gemv` and
`RmsnormAutomatic`.

## Slices

1. **Eager stack** (dense, E-series, EAGLE target). `SandwichAttention`,
   `SandwichMlp`, `PerLayerInput`, `Scale`, `Softcap` with `rows == 1` and
   `rows > 1` executors; `decode_step_body` and `forward_batch_spec` run the
   same program. The token-level PLE staging and embedding stay in the arch
   crate, outside the captured body.
2. **MoE.** `ParallelMoeMlp` (`rows == 1` and `rows > 1`); 26B-A4B runs the
   shared program on its ring/asym3 KV.
3. **EAGLE/MTP drafter.** The drafter block is `[SandwichAttention(q-only,
   write: None, target KV), SandwichMlp, Scale]`; pre/post projection and head
   are `Gemv` steps.
4. **Deletion.** `lowered.rs` super-op facade and hand arms; calibration
   tools move to the shared batched program.

## Calibration taps

Calibration records each projection's unrotated input through
`Gpu::maybe_capture_activation`. The sandwich projection helpers (`gemv`,
`gemm_rows`) and the fused-QKV family fire it, so tools need no
per-architecture tap. While a collector is armed, the `rows == 1` fusions that
never materialize the unrotated input (fused norm+FWHT feeding prerotated
GEMVs) step aside; the other fusions tap their shared input. `gemm_rows`
stages BF16 teacher inputs for the MFMA GEMM and runs formats without a
batched kernel one GEMV per row.

## Gates per slice

- 12B dense, E2B, E4B AR: greedy token stream and the per-step top-16 logit
  trace (`HIPFIRE_GEMMA4_LOGIT_TRACE_DIR`) byte-identical to the pre-port
  daemon (`8a6c4710`).
- 26B-A4B MoE: the shared ops take the eager route's certified fusions
  (qk-norm+RoPE, post-norm+residual) that the lowered path never used, so its
  logits move in the last bits. Gate: coherent decoded text and recorded
  divergence point against the pre-port daemon, same relaxation #795 applied
  to its MTP and decode-arm slices.
- EAGLE drafter: tau parity and coherent text; greedy EAGLE output equals AR
  output where it did before.
- Redline: the 26B shadow oracle (`redline_shadow_gemma4`) stays exact.
  Residual and K=V copies are `copy_f32_buffer` launches, not D2D memcpys,
  so the retained recorder sees them; a memcpy there made PM4 replay inexact.
- Measured (gfx1151, pre-port daemon `8a6c4710`): every 12B/E2B/E4B row
  bitwise; 26B with the fused qk-norm+RoPE and post-norm routes off
  byte-identical (measured before those switches were removed), with them on
  2 of 5 prompts identical and 3 diverging late, coherent; EAGLE per-prompt tau
  3.368/3.459/2.667/3.447/3.514 before and after; 26B
  `redline_daemon_harness.py --pm4 --skip-prefill` shadow exact (1083
  launches, was 1263; tape hash `249f24e5f39d425d`, was `0909f3792962c37d`).
- `serve_harness.py --thinking off --sampling greedy --compare-transcript`
  against the pre-port daemon: 12B battery and chain, E4B battery
  `transcript_byte_identical=true`; 26B battery coherent (no runaway, empty or
  attractor turns), turn 2 diverges.
- Calibration tools on the program (gfx1151, `gemma4-12b.mq4`):
  `prefill_parity_gemma4` (725-token prompt) batched vs per-token same argmax
  and identical 24-token continuation on the 12B (8x faster prefill) and the
  26B-A4B q8-experts MoE (2.4x); `calib_sweep` coverage 328/328, Hessian
  consistency 0, q/k/(v) and gate/up identity PASS; `eval_hipfire` prefill vs
  per-token scoring within 0.3% KLD. After the port the serve gates above
  still hold: 12B/E2B/E4B tokens and top-16 logits bitwise, EAGLE tau
  unchanged, 26B text identical to the first port, Redline shadow exact with
  the same tape hash `249f24e5f39d425d`.

## Progress

| Piece | State | Commit |
|---|---|---|
| Eager decode + batched prefill/verify + E-series PLE as `[SandwichAttention, SandwichMlp, PerLayerInput?, Scale?]` (`program.rs`, dispatch `sandwich.rs`); hand arms deleted | landed; bitwise | `30b55f753` |
| MoE decode (`ParallelMoeMlp`), lowered decode on the shared program; super-op facade (`lower_variant`, `Gemma4Bindings`) and lowered hand arms deleted; unsupported expert formats refuse at load | landed; 26B byte-identical with the two non-bitwise fusions off, coherent with them on | `287e67d5b` |
| EAGLE draft head as one step list (`Gemv` pre-projection, query-only `[SandwichAttention, SandwichMlp, Scale?]` blocks over the target's last slot, norm, `lm_head`, post-projection) | landed; EAGLE text and per-prompt tau identical | `287e67d5b` |
| Calibration tools (`calib_sweep`, `eval_hipfire`, `prefill_parity_gemma4`) on the batched program; dispatch-owned calibration taps; batched `ParallelMoeMlp`; old batched prefill, `HIPFIRE_BATCHED_PREFILL`/`HIPFIRE_WMMA_PREFILL` deleted | landed; tool parity above | (this commit) |

## Remaining

- Two weight stacks remain (`gemma4.rs` eager for dense/E-series/EAGLE,
  `lowered.rs` for MoE) with duplicate config and loaders; both bind to the
  same program.
