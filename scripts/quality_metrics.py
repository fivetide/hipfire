#!/usr/bin/env python3
"""Quality metrics computation from teacher-forced eval JSONL output.

Reads per-position JSONL from teacher_eval and computes aggregate metrics.
Compares against baseline thresholds if provided.

Usage:
    python3 quality_metrics.py <eval.jsonl> [--baseline <baseline.json>] [--output <result.json>]

Exit codes:
    0  metrics within thresholds (or no baseline provided)
    1  regression detected (metric worse than baseline threshold)
    2  input/usage error
"""

import json
import math
import sys
from pathlib import Path


def compute_metrics(jsonl_path: str) -> dict:
    """Compute aggregate quality metrics from teacher_eval JSONL output."""
    total_ce = 0.0
    top1_hits = 0
    top5_hits = 0
    count = 0
    ce_values = []

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            ce = rec.get("cross_entropy")
            if ce is None:
                continue

            total_ce += ce
            ce_values.append(ce)
            if rec.get("top1_correct"):
                top1_hits += 1
            if rec.get("top5_correct"):
                top5_hits += 1
            count += 1

    if count == 0:
        print("ERROR: no valid records in input", file=sys.stderr)
        sys.exit(2)

    mean_ce = total_ce / count
    ppl = math.exp(min(mean_ce, 100.0))  # cap to avoid overflow
    top1_acc = top1_hits / count
    top5_acc = top5_hits / count

    # Compute stddev of cross-entropy for stability assessment
    variance = sum((ce - mean_ce) ** 2 for ce in ce_values) / count
    ce_stddev = math.sqrt(variance)

    # Median cross-entropy
    ce_sorted = sorted(ce_values)
    median_ce = ce_sorted[count // 2] if count % 2 == 1 else (
        (ce_sorted[count // 2 - 1] + ce_sorted[count // 2]) / 2
    )

    return {
        "n_tokens": count,
        "mean_cross_entropy": round(mean_ce, 6),
        "median_cross_entropy": round(median_ce, 6),
        "cross_entropy_stddev": round(ce_stddev, 6),
        "perplexity": round(ppl, 4),
        "top1_accuracy": round(top1_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
    }


def check_thresholds(metrics: dict, baseline: dict) -> list:
    """Compare metrics against baseline thresholds. Returns list of failures."""
    failures = []

    # Cross-entropy: fail if increased beyond tolerance
    tolerance = baseline.get("tolerance", 0.05)

    ce_base = baseline.get("mean_cross_entropy")
    if ce_base is not None:
        ce_max = ce_base * (1 + tolerance)
        if metrics["mean_cross_entropy"] > ce_max:
            failures.append(
                f"mean_cross_entropy: {metrics['mean_cross_entropy']:.6f} > "
                f"threshold {ce_max:.6f} (baseline {ce_base:.6f} + {tolerance*100:.0f}%)"
            )

    # Top-1 accuracy: fail if decreased beyond tolerance
    t1_base = baseline.get("top1_accuracy")
    if t1_base is not None:
        t1_min = t1_base * (1 - tolerance)
        if metrics["top1_accuracy"] < t1_min:
            failures.append(
                f"top1_accuracy: {metrics['top1_accuracy']:.4f} < "
                f"threshold {t1_min:.4f} (baseline {t1_base:.4f} - {tolerance*100:.0f}%)"
            )

    # Top-5 accuracy: fail if decreased beyond tolerance
    t5_base = baseline.get("top5_accuracy")
    if t5_base is not None:
        t5_min = t5_base * (1 - tolerance)
        if metrics["top5_accuracy"] < t5_min:
            failures.append(
                f"top5_accuracy: {metrics['top5_accuracy']:.4f} < "
                f"threshold {t5_min:.4f} (baseline {t5_base:.4f} - {tolerance*100:.0f}%)"
            )

    # Perplexity: fail if increased beyond tolerance
    ppl_base = baseline.get("perplexity")
    if ppl_base is not None:
        ppl_max = ppl_base * (1 + tolerance)
        if metrics["perplexity"] > ppl_max:
            failures.append(
                f"perplexity: {metrics['perplexity']:.4f} > "
                f"threshold {ppl_max:.4f} (baseline {ppl_base:.4f} + {tolerance*100:.0f}%)"
            )

    return failures


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compute quality metrics from teacher_eval output")
    parser.add_argument("jsonl", help="Path to teacher_eval JSONL output")
    parser.add_argument("--baseline", help="Path to baseline JSON for threshold comparison")
    parser.add_argument("--output", "-o", help="Write result JSON to this path")
    parser.add_argument("--tolerance", type=float, help="Override tolerance (default from baseline or 0.05)")
    args = parser.parse_args()

    if not Path(args.jsonl).exists():
        print(f"ERROR: input file not found: {args.jsonl}", file=sys.stderr)
        sys.exit(2)

    metrics = compute_metrics(args.jsonl)

    print(f"=== Quality Metrics ===")
    print(f"  tokens evaluated:    {metrics['n_tokens']}")
    print(f"  mean cross-entropy:  {metrics['mean_cross_entropy']:.6f}")
    print(f"  median cross-entropy:{metrics['median_cross_entropy']:.6f}")
    print(f"  CE stddev:           {metrics['cross_entropy_stddev']:.6f}")
    print(f"  perplexity:          {metrics['perplexity']:.4f}")
    print(f"  top-1 accuracy:      {metrics['top1_accuracy']*100:.2f}%")
    print(f"  top-5 accuracy:      {metrics['top5_accuracy']*100:.2f}%")

    result = {"metrics": metrics, "status": "PASS", "failures": []}

    if args.baseline:
        if not Path(args.baseline).exists():
            print(f"WARNING: baseline file not found: {args.baseline}", file=sys.stderr)
        else:
            with open(args.baseline) as f:
                baseline = json.load(f)
            if args.tolerance is not None:
                baseline["tolerance"] = args.tolerance
            failures = check_thresholds(metrics, baseline)
            result["failures"] = failures
            if failures:
                result["status"] = "FAIL"
                print(f"\n=== QUALITY REGRESSION DETECTED ===")
                for fail in failures:
                    print(f"  FAIL: {fail}")
            else:
                print(f"\n=== All metrics within thresholds ===")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")

    sys.exit(1 if result["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
