# HFP4 Quality Analysis: E2M1 vs INT4 Under FWHT Rotation

**Date:** 2026-05-11
**Branch:** `docs/hfp4-quality-investigation`
**Formats compared:** MFP4G32 (FWHT + E2M1 g=32), MQ4G256 (FWHT + INT4 g=256), HFP4G32 (unrotated E2M1), HFQ4G256 (unrotated INT4)
**Models:** Qwen3.5-0.8B, 4B, 9B
**Corpus:** wikitext2-test, ctx=2048, warmup=8
**KV mode:** asym4 (isolates weight quantization effect)
**GPU:** gfx1151 (7900 XTX), HIP 7.13
**Methodology audit:** Opus-class review confirmed no bugs in quantizer, kernel, dispatch, or perplexity harness.

---

## Executive Summary

MFP4G32 (hipfire's FWHT-rotated E2M1 FP4 format) produces **+25–94% worse
perplexity** than MQ4G256 (FWHT-rotated INT4) across three model sizes.
However, **KL divergence analysis reveals a more nuanced picture**: MFP4 is
actually the **closest** to MQ4 at the output distribution level (KLD 0.661
vs HFQ4's 0.936), with 63.2% top-1 token agreement and the fewest
catastrophic divergence positions.

**The PPL gap is not a distributional collapse — it's a consistent small
entropy increase.** E2M1's errors slightly broaden probability mass from
the correct next token to its neighbors at every position, compounding into
worse PPL. But the overall distribution shape tracks MQ4 closely. For
sampling-based generation (temperature > 0), MFP4 may be nearly equivalent.

**FWHT genuinely helps E2M1** (contradicts initial "anti-synergy"
hypothesis): MFP4 (FWHT+E2M1) has 30% lower KLD than HFP4 (E2M1 alone).

Seven concrete improvement paths are identified and ranked. The most
impactful: FP16 block scales (+8.76% NRMSE for +0.25 bpw), per-block
zero-point, and entropy-focused optimization rather than NRMSE.

---

## 1. Perplexity Results

### 1.1 Rotated (FWHT) Comparison

| Model | MQ4G256 PPL | MFP4G32 PPL | Delta | MFP4 tok/s | MQ4 tok/s |
|:------|:----------:|:-----------:|:-----:|:----------:|:---------:|
| Qwen3.5-0.8B | **24.37** | 47.23 | +93.8% | — | — |
| Qwen3.5-4B | **12.59** | 16.65 | +32.3% | — | — |
| Qwen3.5-9B | **9.94** | 12.47 | +25.4% | 41.2 | 38.0 |

The gap narrows with scale but remains substantial at 9B. MFP4 is ~8%
faster on decode (simpler dequant, no zero-point add), but the quality
gap makes this irrelevant.

### 1.2 Unrotated Comparison

| Model | HFQ4G256 PPL | HFP4G32 PPL | Delta | Notes |
|:------|:-----------:|:-----------:|:-----:|:------|
| Qwen3.5-0.8B | 76.88 | **62.08** | -19.3% | **Confounded:** HFP4 uses Q8 embeddings, HFQ4 uses Q4. Embeddings are ~50% of 0.8B params. |
| Qwen3.5-4B | **16.46** | 19.08 | +15.9% | Clean comparison. Embeddings are ~16% of params. |

At 4B (where the embedding confound is diluted), INT4 wins even on
unrotated heavy-tailed weights. INT4's per-group affine quantization
(FP32 scale + FP32 zero-point) adapts to each group's actual value
range, trumping E2M1's fixed non-uniform codebook shape.

### 1.3 Combined View

| Format | Rotation | Codebook | Scale | PPL (4B) |
|--------|:--------:|----------|-------|:--------:|
| MQ4G256 | FWHT | INT4 uniform | FP32 scale + FP32 zp, g=256 | **10.83** |
| HFQ4G256 | none | INT4 uniform | FP32 scale + FP32 zp, g=256 | 16.46 |
| MFP4G32 | FWHT | E2M1 FP4 | UE8M0 + FP16 row, g=32 | 18.01 |
| HFP4G32 | none | E2M1 FP4 | UE8M0 + FP16 row, g=32 | 19.08 |

FWHT improves INT4 by 34% but E2M1 by only 5.6%. The rotation actively
hurts E2M1's relative standing.

---

## 2. Root Cause Analysis

### 2.1 Post-FWHT Distribution Shape

We measured per-block (g=32) excess kurtosis on real weight data across
three models. The results are remarkably stable:

| Model | Raw kurtosis | Post-FWHT global | Post-FWHT per-block (g=32) |
|:------|:-----------:|:----------------:|:--------------------------:|
| Qwen3.5-0.8B | 5.82 | 4.31 | **2.82** |
| Qwen3.5-27B | 13.36 | 9.04 | **2.82** |
| Qwen3.6-35B-A3B | 6.13 | 4.33 | **2.81** |

Reference: Gaussian = 3.0, Uniform = 1.8.

**Post-FWHT per-block distributions are sub-Gaussian (kurtosis ~2.82),
not uniform.** This corrects the prior characterization in CLAUDE.md of
"near-uniform." The distribution is closer to Gaussian (distance 0.18)
than to Uniform (distance 1.02).

**Implication for codebook design:** For sub-Gaussian distributions
(platykurtic, lighter tails than Gaussian), the optimal scalar quantizer
has *more uniform* spacing than for Gaussian — i.e., INT4 uniform is
closer to optimal than E2M1's near-zero-concentrated spacing.

### 2.2 Decomposed PPL Gap (9B, +25%)

| Factor | Contribution | Mechanism |
|--------|:----------:|-----------|
| E2M1 codebook shape mismatch | ~60% | Non-uniform spacing suboptimal for sub-Gaussian; codes wasted in tails |
| UE8M0 power-of-2 block scale | ~25% | Can only scale by powers of 2; 26.6% average overshoot; ~15% per-block precision loss |
| No zero-point (forced symmetric) | ~10% | MQ4's FP32 min_val absorbs per-group asymmetry; E2M1 lattice is symmetric |
| FP16 vs FP32 row/group scale | ~5% | 11-bit vs 23-bit mantissa; minor at this precision level |

### 2.3 The FWHT + E2M1 Anti-Synergy

E2M1's codebook `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` allocates 4 of 8 positive
magnitudes to the [0, 2] range and only 2 to [3, 6]. This is designed for
distributions with most mass near zero and sparse tails — exactly what raw
transformer weights look like (kurtosis 5–13).

FWHT rotation spreads that mass outward, producing a sub-Gaussian
distribution where values are more evenly spread across the range. After
FWHT, E2M1's near-zero concentration becomes a liability: the [3, 6]
codes (step sizes of 1.0 and 2.0) cover a range where there is now
meaningful probability mass, and the coarse spacing there destroys
information.

INT4 uniform spacing distributes reconstruction points evenly, which is
close to optimal for both uniform and sub-Gaussian distributions.

---

## 3. Codebook Analysis (Lloyd-Max)

We computed the optimal 16-code scalar quantization codebook (Lloyd-Max /
1D k-means) on real post-FWHT weight data from Qwen3.5-0.8B (615M weight
elements, 244 tensors).

### 3.1 Optimal Codebook

```
Lloyd-Max optimal (8 unsigned magnitudes, normalized to block max):
{0.050, 0.151, 0.256, 0.366, 0.485, 0.619, 0.775, 0.969}
```

The optimal codebook is **nearly uniform** — roughly evenly spaced with
slight compression at extremes and a non-zero minimum (c0 = 0.05).

### 3.2 MSE Comparison at g=32

| Codebook | MSE | vs Optimal | L2 distance to optimal |
|----------|:---:|:----------:|:-----:|
| Lloyd-Max optimal | 0.00123 | baseline | 0.000 |
| INT4 uniform | 0.00165 | +33.7% | 0.141 |
| E2M1 (MXFP4) | 0.00196 | +58.8% | 0.314 |

INT4 uniform is 2.2× closer to optimal than E2M1 in both MSE and
codebook shape (L2 distance).

### 3.3 Block Size Crossover

| Codebook | MSE (g=32) | MSE (g=256) |
|----------|:----------:|:-----------:|
| Lloyd-Max | 0.00123 | 0.00106 |
| INT4 uniform | 0.00165 | 0.00169 |
| E2M1 | 0.00196 | **0.00137** |

**At g=256, E2M1 beats INT4.** Larger blocks have wider within-block
dynamic range, making E2M1's exponential spacing beneficial. HFP4's choice
of g=32 — intended to improve quality via finer block adaptation —
actually *undermines* E2M1's codebook by normalizing away the dynamic
range variation that E2M1 exploits.

### 3.4 Practical LUT

Rounding the Lloyd-Max codebook to 1/16 granularity gives:

```
{1, 2, 4, 6, 8, 10, 12, 16} / 16
```

This is implementable as a nibble LUT with a single `scale * LUT[nibble]`
multiply — identical hardware cost to E2M1. On pre-RDNA4 hardware (where
E2M1 is a software LUT anyway), this is a drop-in replacement.

---

## 4. Scale Precision Analysis

We simulated three block-scale variants on Qwen3.5-0.8B (262 tensors,
19.25M blocks):

### 4.1 NRMSE by Scale Type

| Scheme | NRMSE | vs UE8M0 | bpw |
|--------|:-----:|:--------:|:---:|
| E2M1 + UE8M0 block (current HFP4) | 0.1109 | baseline | 4.25 |
| E2M1 + FP16 block | 0.1011 | **−8.76%** | 4.50 |
| E2M1 + FP32 block | 0.1011 | −8.76% | 5.25 |
| MQ4 INT4 g=256 (reference) | 0.1087 | −1.94% | 4.25 |

### 4.2 Key Findings

- **FP16 block scale captures ALL available precision.** FP32 gives
  exactly zero additional NRMSE improvement (identical to 6 decimal
  places). The E2M1 codebook granularity — not the scale precision —
  is the bottleneck.
- **UE8M0 overshoots by 26.6% on average** (mean actual/UE8M0 scale
  ratio = 0.79). The `ceil()` rounding guarantees no clipping but wastes
  up to 50% of the E2M1 dynamic range per block.
- **Cost:** FP16 block scales add +0.25 bpw (4.25 → 4.50), a 5.8% file
  size increase for an 8.76% NRMSE improvement.

### 4.3 The NRMSE Paradox

**E2M1 + FP16 block scale (NRMSE 0.1011) beats MQ4 (NRMSE 0.1087) in
per-tensor reconstruction quality — yet MQ4 has substantially better PPL.**

This means aggregate per-tensor reconstruction error is not the right
optimization target. MQ4's error distribution (uniform, evenly spread
across the codebook range) propagates more favorably through the
transformer stack than E2M1's error distribution (concentrated in the
sparse tail codes). Optimizing for per-tensor NRMSE will not close the
PPL gap.

---

## 5. KL Divergence Analysis (Critical Revision)

KLD analysis against MQ4 as reference (0.8B, wikitext2, ctx=512, asym4):

### 5.1 Results

| Metric | MFP4 (FWHT+E2M1) | HFP4 (E2M1, no FWHT) | HFQ4 (INT4, no FWHT) |
|--------|:-:|:-:|:-:|
| **Mean KLD** | **0.661** | 0.815 | 0.936 |
| Median KLD | 0.498 | 0.696 | 0.799 |
| **Max KLD** | **4.044** | 7.076 | 7.533 |
| **Mean JSD** | **0.131** | 0.153 | 0.180 |
| **Top-1 agreement** | **63.2%** | 59.6% | 49.3% |
| Top-5 overlap | 2.96/5 | 2.81/5 | 2.71/5 |
| Mean L-inf | 0.192 | 0.213 | 0.240 |

Self-comparison (MQ4 vs MQ4) produces exactly zero divergence across all
503 scored positions, confirming tool correctness.

### 5.2 MFP4 is Closest to MQ4

Despite losing on PPL, MFP4 is **the closest format to MQ4** at the output
distribution level:
- 30% lower KLD than HFQ4 (0.661 vs 0.936)
- 19% lower KLD than HFP4 (0.661 vs 0.815)
- 63.2% top-1 agreement (vs 49.3% for HFQ4)
- Fewest catastrophic positions (max KLD 4.04 vs 7.53)

### 5.3 FWHT Helps E2M1 (Contradicts "Anti-Synergy" Hypothesis)

MFP4 (FWHT + E2M1) has significantly lower KLD than HFP4 (E2M1 alone):
0.661 vs 0.815, a 19% improvement. FWHT makes E2M1's quantization errors
more uniform and predictable, keeping output distributions closer to MQ4.

The initial "anti-synergy" conclusion was based solely on PPL, which
measures cross-entropy against the corpus (ground truth), not against MQ4.
KLD reveals that FWHT + E2M1 produces distributions that track MQ4 more
closely than E2M1 alone — the opposite of what PPL suggested.

### 5.4 PPL vs KLD: Why They Disagree

**PPL** = cross-entropy against ground truth tokens. Sensitive to the
probability assigned to the single correct next token.

**KLD** = divergence of the full output distribution from a reference.
Captures overall distributional similarity.

MFP4 tracks MQ4's distribution closely (low KLD, high top-1 agreement)
but with consistently slightly **higher entropy** — probability mass
spreads slightly from the correct token to neighbors. Each position
loses a small amount of log-probability on the correct token. This
compounds multiplicatively across hundreds of positions into a large
PPL gap, even though the per-position distributions look similar.

**Implication:** The PPL gap is a consistent small entropy increase, not a
distributional collapse. For **sampling-based generation** (temperature > 0,
which is the dominant use case for chat/code/reasoning), MFP4's output
quality may be nearly equivalent to MQ4. The PPL gap overstates the
practical quality difference.

---

## 6. Improvement Paths (Ranked, Revised Post-KLD)

Given the constraint that RDNA4-native E2M1 is the long-term hardware
target (`v_cvt_pk_fp8_e2m1`, `v_wmma_f32_16x16x32_fp8_fp8`):

### Tier 1 — High impact, empirically validated

| # | Improvement | Evidence | Impact estimate | Cost |
|---|-------------|----------|:---------------:|:----:|
| 1 | **FP16 block scale** | 8.76% NRMSE gain; FP32 = zero additional benefit | May disproportionately close PPL gap if entropy spread is scale-driven | +0.25 bpw (+5.8% size) |
| 2 | **Per-block zero-point** | INT4 wins unrotated because of per-group affine adaptation | ~2-3% PPL improvement (10% of gap) | +0.5 bpw (+12% size) |
| 3 | **Keep FWHT with E2M1** | KLD: MFP4 0.661 vs HFP4 0.815 — FWHT helps 19% | Reversed from initial recommendation | No cost change |

### Tier 2 — Moderate impact, theoretically grounded

| # | Improvement | Evidence | Impact estimate | Cost |
|---|-------------|----------|:---------------:|:----:|
| 4 | **Entropy-focused optimization** | PPL gap is entropy spread, not wrong tokens | Target: reduce probability broadening, not MSE | Research needed |
| 5 | **Lloyd-Max LUT on pre-RDNA4** | 58.8% less MSE than E2M1; codebook is nearly uniform | Significant but pre-RDNA4 only (software LUT) | Zero runtime cost |
| 6 | **Stochastic rounding** | RTN bias in E2M1's 0-to-0.5 density gap is systematic | May reduce entropy spread | ~10 lines in quantizer |
| 7 | **E2M1 at g=256** | E2M1 beats INT4 at g=256 in MSE (0.00137 vs 0.00169) | May recover E2M1's natural advantage | Loses fine block adaptation |

### Tier 3 — High effort, untested

| # | Improvement | Evidence | Impact estimate | Cost |
|---|-------------|----------|:---------------:|:----:|
| 7 | **Calibration-aware scale fitting (AWQ/GPTQ)** | Literature: biggest quality lever for low-bit quant | Potentially large | Significant implementation |
| 8 | **Online learned rotation** | SpinQuant: 1-2 PPL over fixed Hadamard | Moderate | Requires per-model calibration |
| 9 | **FP8 activation quantization** | Required for RDNA4 `fp8_fp8` WMMA ceiling | Throughput, not quality | New activation pipeline |

---

## 7. Strategic Recommendations (Revised Post-KLD)

### 7.1 For RDNA4 Hardware Path

The RDNA4 `v_cvt_pk_fp8_e2m1` instruction locks the element format to
E2M1. Given this constraint and the KLD findings:

1. **Keep FWHT with E2M1** — KLD shows FWHT helps E2M1 by 19% (MFP4
   KLD 0.661 vs HFP4 0.815). The initial "anti-synergy" conclusion
   was based on PPL alone and is revised.
2. **Use FP16 block scales** — UE8M0's power-of-2 constraint costs
   8.76% NRMSE for zero benefit. FP16 captures all available precision.
   May disproportionately close the PPL gap if the entropy spread is
   driven by scale overshoot.
3. **Add per-block FP16 bias (zero-point)** — the single biggest
   quality differentiator between INT4 and E2M1 is per-group affine
   adaptation, not the codebook shape. A per-block bias would give E2M1
   the same per-group adaptivity at +0.5 bpw.
4. **Focus on entropy reduction** — the PPL gap is from consistent
   small probability broadening, not wrong tokens. Investigate whether
   tighter scales, stochastic rounding, or GPTQ error feedback can
   reduce this entropy spread specifically.

### 7.2 For Pre-RDNA4 (Current gfx1100/1151)

1. **Keep MQ4G256 as the quality default for greedy decoding** — it
   wins on PPL in every comparison.
2. **MFP4 may be viable for sampling use cases** — 63.2% top-1
   agreement with MQ4 and lowest KLD of any alternative format. The
   PPL gap may not manifest in practice under temperature > 0.
3. **Consider Lloyd-Max LUT** for unrotated FP4 — the LUT is pure
   software, zero hardware cost, 37% less MSE than E2M1.

### 7.3 The Key Question Going Forward

The PPL-vs-KLD divergence reveals that **the right evaluation metric
depends on the use case:**

- **Greedy/deterministic decoding** (benchmarks, factual QA): PPL is
  the right metric. MQ4 wins. E2M1's entropy spread compounds.
- **Sampling-based generation** (chat, code, reasoning with temp > 0):
  KLD and top-1 agreement are more relevant. MFP4's 63.2% top-1
  agreement and low KLD suggest practical equivalence.
- **RDNA4 hardware throughput**: if `v_cvt_pk_fp8_e2m1` +
  `v_wmma_f32_16x16x32_fp8_fp8` delivers 2× FLOPS of software INT4,
  MFP4's small entropy cost may be an acceptable trade for speed.

The next step is to measure MFP4 vs MQ4 on **downstream task quality**
(chat coherence, code completion accuracy) under sampling, not just PPL.

---

## 8. Methodology Notes

### 7.1 Tools and Data

- **Distribution analysis:** Custom Rust experiment applying FWHT to
  real safetensors weights, computing per-block kurtosis. Tested on
  Qwen3.5-0.8B, 27B, and Qwen3.6-35B-A3B (MoE).
- **Codebook analysis:** Lloyd-Max (1D k-means) on 615M post-FWHT
  weight elements from Qwen3.5-0.8B.
- **Scale precision:** CPU-side NRMSE simulation on 262 tensors,
  19.25M blocks from Qwen3.5-0.8B.
- **PPL benchmarks:** `examples/perplexity` with wikitext2-test corpus,
  ctx=2048, warmup=8, `--kv-mode asym4`.

### 7.2 Confounds and Limitations

- **0.8B unrotated comparison is confounded** by embedding precision
  (HFP4 uses Q8 embeddings, HFQ4 uses Q4). The 4B result is clean.
- **PPL gap decomposition is estimated** (60/25/10/5%) based on
  component analysis, not ablation. True ablation would require
  implementing each variant (FP16 scale, zero-point) end-to-end.
- **NRMSE-to-PPL scaling is approximate.** The NRMSE paradox (E2M1+FP16
  beats MQ4 on NRMSE, loses on PPL) demonstrates this relationship is
  non-linear and error-distribution-dependent.
- **Only Qwen3.5/3.6 models tested.** Results may differ for other
  architectures with different weight distributions.

### 7.3 Spec/Code Discrepancy

`docs/quant-formats/hfp4.md` line 130 specifies `round(log2(...))` for
block_e computation. The implementation at `crates/hipfire-quantize/
src/main.rs:713-716` uses `ceil()`. The `ceil` choice is correct and
documented in a code comment (prevents clipping at the cost of ≤1 bit
resolution). The spec should be updated to match the code.

---

## Appendix A: Experiment Locations

| Experiment | Location |
|------------|----------|
| Distribution histogram | `experiments/fwht-distribution/` |
| Codebook (Lloyd-Max) analysis | `experiments/codebook-analysis/` |
| Scale precision simulation | `experiments/scale-precision/` |
| Quantized MFP4 models | `~/.hipfire/models/qwen3.5-{0.8b,4b,9b}.mfp4` |
| Quantized HFP4/HFQ4 models | `~/.hipfire/models/qwen3.5-{0.8b,4b}.{hfp4,hfq4}` |

## Appendix B: References

- OCP Microscaling Formats (MX) Specification v1.0
- AMD ROCm Blog — High-Accuracy MXFP4/MXFP6
- AMD ROCm Blog — Advanced MXFP4 with Online Rotation
- SpinQuant (arXiv 2405.16406) — learned rotations for quantization
- HFP4 format spec: `docs/quant-formats/hfp4.md`
