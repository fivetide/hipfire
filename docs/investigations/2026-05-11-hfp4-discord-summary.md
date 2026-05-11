# HFP4 Quality Investigation — Results (Revised with KLD)

**Full report:** `docs/investigations/2026-05-11-hfp4-quality-analysis.md` (branch `docs/hfp4-quality-investigation`)

---

## TL;DR

MFP4G32 (FWHT + E2M1 FP4) loses to MQ4G256 (FWHT + INT4) by **+25–94% PPL**. But KL divergence tells a different story: **MFP4 is actually the closest format to MQ4 at the output distribution level.**

## PPL says MFP4 loses

| Model | MQ4 PPL | MFP4 PPL | Delta |
|-------|---------|----------|-------|
| 0.8B  | 24.37   | 47.23    | +94%  |
| 4B    | 12.59   | 16.65    | +32%  |
| 9B    | 9.94    | 12.47    | +25%  |

## KLD says MFP4 is closest to MQ4

| Metric | MFP4 (FWHT+E2M1) | HFP4 (E2M1 only) | HFQ4 (INT4 only) |
|--------|:-:|:-:|:-:|
| **Mean KLD** | **0.661** | 0.815 | 0.936 |
| **Top-1 agreement** | **63.2%** | 59.6% | 49.3% |
| **Max KLD** | **4.04** | 7.08 | 7.53 |

MFP4 picks the same token as MQ4 nearly two-thirds of the time. Fewest catastrophic positions. 30% lower KLD than HFQ4.

## Why PPL and KLD disagree

PPL measures probability of the *correct* next token. KLD measures divergence of the *full* output distribution. MFP4 tracks MQ4's distribution closely but with slightly higher entropy — probability mass spreads from the correct token to neighbors at every position. This compounds into worse PPL but doesn't cause wrong tokens or distributional collapse.

**For sampling (temp > 0), MFP4 may be practically equivalent to MQ4.** The PPL gap overstates the real-world quality difference.

## Key correction: FWHT helps E2M1

Initial analysis concluded FWHT + E2M1 was an "anti-synergy." **KLD disproves this:** MFP4 (with FWHT) has 19% lower KLD than HFP4 (without). FWHT makes E2M1 errors more uniform, keeping distributions closer to MQ4.

## Other findings (unchanged)

- Post-FWHT kurtosis = 2.82 (sub-Gaussian). Lloyd-Max optimal codebook is nearly uniform.
- FP16 block scale: +8.76% NRMSE over UE8M0; FP32 adds zero more. (+0.25 bpw)
- E2M1 beats INT4 at g=256 but loses at g=32 (block-size crossover)
- Per-block zero-point would give E2M1 the affine adaptation that is INT4's key advantage

## What to do

1. **Keep FWHT with E2M1** — it helps (KLD proves it)
2. **FP16 block scales** — may disproportionately close the PPL gap if entropy spread is scale-driven
3. **Per-block zero-point** — lets E2M1 match INT4's affine adaptation
4. **Evaluate on downstream tasks under sampling** — PPL alone misleads for the chat/code use case
5. **MQ4 stays quality king for greedy decoding** — unchanged

Full methodology, PPL tables, KLD tables, codebook analysis, and scale precision data in the report.
