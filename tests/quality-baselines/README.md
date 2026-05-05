# Quality Gate Baselines

Per-arch quality baselines for teacher-forced evaluation.

## Structure

```
tests/quality-baselines/
  <arch>/
    <size>_<reference>.json
```

Example: `gfx1100/0.8b_wikipedia_combustion.json`

## Generating Baselines

```bash
./scripts/quality-gate.sh --update
```

## Format

Each baseline JSON contains:
- `tolerance`: allowed regression fraction (default 0.05 = 5%)
- `mean_cross_entropy`: baseline CE loss
- `perplexity`: baseline PPL
- `top1_accuracy`: baseline top-1 token prediction accuracy
- `top5_accuracy`: baseline top-5 token prediction accuracy

The gate fails if any metric regresses beyond `baseline * (1 + tolerance)`
for CE/PPL, or drops below `baseline * (1 - tolerance)` for accuracies.
