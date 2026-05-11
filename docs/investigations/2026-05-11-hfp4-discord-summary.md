# HFP4 Quality Investigation — Results

**Full report:** `docs/investigations/2026-05-11-hfp4-quality-analysis.md` (branch `docs/hfp4-quality-investigation`)

---

## TL;DR

We benchmarked MFP4G32 (our FWHT-rotated E2M1 FP4 format) against MQ4G256 (FWHT-rotated INT4) on Qwen3.5 0.8B/4B/9B wikitext2 perplexity.

**MFP4 loses by +25–94% PPL.** E2M1 also loses unrotated at 4B (+16%).

## Why

FWHT + E2M1 is an **anti-synergy**. We measured post-FWHT per-block distributions at kurtosis ~2.82 (sub-Gaussian, stable across 0.8B/27B/35B-MoE). This is the zone where uniform quantization (INT4) is near-optimal. E2M1's non-uniform codebook concentrates codes near zero — exactly where the post-FWHT distribution no longer needs them.

Lloyd-Max analysis on real data confirms: the optimal 16-code codebook for post-FWHT weights is **nearly uniform**. E2M1 has 58.8% more MSE than optimal; INT4 has 33.7% more.

Interesting twist: **E2M1 beats INT4 at g=256** (wider blocks, more dynamic range). HFP4's g=32 block size works *against* E2M1 by normalizing away the variance it's designed to exploit.

## Decomposition (9B, +25% PPL gap)

- 60% — E2M1 codebook mismatch for sub-Gaussian data
- 25% — UE8M0 power-of-2 scale (26.6% avg overshoot). FP16 block scale gives 8.76% NRMSE gain; FP32 adds zero more.
- 10% — No zero-point. INT4's per-group affine adaptation is its key advantage.
- 5% — FP16 vs FP32 scale precision

## NRMSE Paradox

E2M1 + FP16 block scale **beats MQ4 on per-tensor NRMSE** (0.101 vs 0.109) but still loses on PPL. Per-tensor reconstruction quality is not the right optimization target — error propagation pattern through the transformer stack matters more.

## What This Means for RDNA4

E2M1 is hardware-locked on RDNA4 (`v_cvt_pk_fp8_e2m1`). To make it competitive:

1. **Don't pair with FWHT** — use E2M1 unrotated on native heavy-tailed weights
2. **FP16 block scales** — UE8M0's power-of-2 constraint wastes 8.76% NRMSE for nothing (+0.25 bpw)
3. **Add per-block zero-point** — INT4's real advantage is affine adaptation, not uniform spacing
4. **Pre-RDNA4: Lloyd-Max LUT** — 37% less MSE than E2M1, it's just a software table swap

MQ4 remains quality king for FWHT-rotated inference. MFP4 (FWHT + E2M1) should not receive further investment.

Full methodology, PPL tables, codebook analysis, and scale precision data in the report.
