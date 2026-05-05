#!/usr/bin/env python3
"""Quality metrics computation from teacher-forced eval output.

Supports two modes:

1. JSONL mode (original): Reads per-position JSONL from teacher_eval and computes
   aggregate metrics. Compares against baseline thresholds if provided.

2. Binary comparison mode: Loads two llama.cpp-format binary logit files and computes
   the full metric set: KL divergence (mean/max/p95), top-k Jaccard overlap,
   mean rank delta, token agreement, and perplexity delta.

Usage:
    # JSONL mode
    python3 quality_metrics.py jsonl <eval.jsonl> [--baseline <baseline.json>] [--output <result.json>]

    # Binary comparison mode
    python3 quality_metrics.py compare <baseline.bin> <candidate.bin> [--top-k 10] [--output <result.json>]

Exit codes:
    0  metrics within thresholds (or no baseline provided)
    1  regression detected (metric worse than baseline threshold)
    2  input/usage error
"""

import json
import math
import struct
import sys
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


def load_binary_logits(path: str):
    """Load a llama.cpp-format binary logit file.

    Format: [n_vocab:u32 LE][n_tokens:u32 LE][f32 x n_vocab x n_tokens]

    Returns (n_vocab, n_tokens, logits) where logits is either a numpy array
    of shape (n_tokens, n_vocab) or a list of lists if numpy is unavailable.
    """
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) < 8:
            print(f"ERROR: file too short for header: {path}", file=sys.stderr)
            sys.exit(2)
        n_vocab, n_tokens = struct.unpack("<II", header)
        expected_bytes = n_vocab * n_tokens * 4
        data = f.read(expected_bytes)
        if len(data) < expected_bytes:
            print(
                f"ERROR: file truncated: expected {expected_bytes} bytes of logit data, "
                f"got {len(data)} (n_vocab={n_vocab}, n_tokens={n_tokens})",
                file=sys.stderr,
            )
            sys.exit(2)

    if HAS_NUMPY:
        logits = np.frombuffer(data, dtype=np.float32).reshape(n_tokens, n_vocab)
    else:
        flat = struct.unpack(f"<{n_vocab * n_tokens}f", data)
        logits = [list(flat[i * n_vocab:(i + 1) * n_vocab]) for i in range(n_tokens)]

    return n_vocab, n_tokens, logits


def _softmax_row(row):
    """Numerically stable softmax for a single row (list of floats)."""
    max_val = max(row)
    exp_vals = [math.exp(x - max_val) for x in row]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def _topk_indices(row, k):
    """Return set of indices of top-k values in a row."""
    indexed = sorted(range(len(row)), key=lambda i: row[i], reverse=True)
    return set(indexed[:k])


def _argmax(row):
    """Return index of maximum value."""
    best_idx = 0
    best_val = row[0]
    for i in range(1, len(row)):
        if row[i] > best_val:
            best_val = row[i]
            best_idx = i
    return best_idx


def _percentile(sorted_vals, pct):
    """Return the value at the given percentile from a sorted list."""
    idx = int(len(sorted_vals) * pct)
    if idx >= len(sorted_vals):
        idx = len(sorted_vals) - 1
    return sorted_vals[idx]


def compute_kl_divergence_pure(baseline_logits, candidate_logits, n_tokens):
    """Compute per-position KL(baseline || candidate) — pure Python."""
    eps = 1e-10
    kl_values = []

    for i in range(n_tokens):
        p = _softmax_row(baseline_logits[i])
        q = _softmax_row(candidate_logits[i])

        kl = 0.0
        for j in range(len(p)):
            pj = max(p[j], eps)
            qj = max(q[j], eps)
            kl += pj * (math.log(pj) - math.log(qj))
        kl_values.append(max(kl, 0.0))

    kl_sorted = sorted(kl_values)
    return {
        "mean": sum(kl_values) / len(kl_values),
        "max": max(kl_values),
        "p95": _percentile(kl_sorted, 0.95),
        "median": _percentile(kl_sorted, 0.50),
    }


def compute_kl_divergence_np(baseline_logits, candidate_logits):
    """Compute per-position KL(baseline || candidate) — numpy."""
    def softmax(logits):
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=-1, keepdims=True)

    p = softmax(baseline_logits)
    q = softmax(candidate_logits)

    eps = 1e-10
    q_clamped = np.maximum(q, eps)
    p_clamped = np.maximum(p, eps)

    kl_per_pos = np.sum(p_clamped * (np.log(p_clamped) - np.log(q_clamped)), axis=-1)
    kl_per_pos = np.maximum(kl_per_pos, 0.0)

    kl_sorted = np.sort(kl_per_pos)
    p95_idx = int(len(kl_sorted) * 0.95)

    return {
        "mean": float(np.mean(kl_per_pos)),
        "max": float(np.max(kl_per_pos)),
        "p95": float(kl_sorted[min(p95_idx, len(kl_sorted) - 1)]),
        "median": float(np.median(kl_per_pos)),
    }


def compute_topk_jaccard_pure(baseline_logits, candidate_logits, n_tokens, k):
    """Compute top-k Jaccard overlap — pure Python."""
    jaccard_values = []

    for i in range(n_tokens):
        b_set = _topk_indices(baseline_logits[i], k)
        c_set = _topk_indices(candidate_logits[i], k)
        intersection = len(b_set & c_set)
        union = len(b_set | c_set)
        jaccard_values.append(intersection / union if union > 0 else 1.0)

    jaccard_sorted = sorted(jaccard_values)
    return {
        "mean": sum(jaccard_values) / len(jaccard_values),
        "min": min(jaccard_values),
        "p5": _percentile(jaccard_sorted, 0.05),
        "k": k,
    }


def compute_topk_jaccard_np(baseline_logits, candidate_logits, k):
    """Compute top-k Jaccard overlap — numpy."""
    n_tokens = baseline_logits.shape[0]
    baseline_topk = np.argpartition(baseline_logits, -k, axis=-1)[:, -k:]
    candidate_topk = np.argpartition(candidate_logits, -k, axis=-1)[:, -k:]

    jaccard_values = []
    for i in range(n_tokens):
        b_set = set(baseline_topk[i])
        c_set = set(candidate_topk[i])
        intersection = len(b_set & c_set)
        union = len(b_set | c_set)
        jaccard_values.append(intersection / union if union > 0 else 1.0)

    jaccard_sorted = sorted(jaccard_values)
    return {
        "mean": sum(jaccard_values) / len(jaccard_values),
        "min": min(jaccard_values),
        "p5": _percentile(jaccard_sorted, 0.05),
        "k": k,
    }


def compute_rank_delta_pure(baseline_logits, candidate_logits, n_tokens):
    """Compute rank delta — pure Python."""
    rank_deltas = []

    for i in range(n_tokens):
        target = _argmax(baseline_logits[i])
        target_val = candidate_logits[i][target]
        rank = sum(1 for v in candidate_logits[i] if v > target_val)
        rank_deltas.append(rank)

    rank_sorted = sorted(rank_deltas)
    return {
        "mean": sum(rank_deltas) / len(rank_deltas),
        "max": max(rank_deltas),
        "median": _percentile(rank_sorted, 0.50),
        "p95": _percentile(rank_sorted, 0.95),
    }


def compute_rank_delta_np(baseline_logits, candidate_logits):
    """Compute rank delta — numpy."""
    n_tokens = baseline_logits.shape[0]
    baseline_argmax = np.argmax(baseline_logits, axis=-1)

    rank_deltas = np.empty(n_tokens, dtype=np.float64)
    for i in range(n_tokens):
        target = baseline_argmax[i]
        target_val = candidate_logits[i, target]
        rank_deltas[i] = int(np.sum(candidate_logits[i] > target_val))

    rank_sorted = np.sort(rank_deltas)
    return {
        "mean": float(np.mean(rank_deltas)),
        "max": float(np.max(rank_deltas)),
        "median": float(np.median(rank_deltas)),
        "p95": float(rank_sorted[int(len(rank_sorted) * 0.95)]),
    }


def compute_token_agreement_pure(baseline_logits, candidate_logits, n_tokens):
    """Compute top-1 agreement — pure Python."""
    agree = sum(
        1 for i in range(n_tokens)
        if _argmax(baseline_logits[i]) == _argmax(candidate_logits[i])
    )
    return {"top1_agreement": agree / n_tokens}


def compute_token_agreement_np(baseline_logits, candidate_logits):
    """Compute top-1 agreement — numpy."""
    baseline_argmax = np.argmax(baseline_logits, axis=-1)
    candidate_argmax = np.argmax(candidate_logits, axis=-1)
    return {"top1_agreement": float(np.mean(baseline_argmax == candidate_argmax))}


def compute_perplexity_delta_pure(baseline_logits, candidate_logits, n_tokens):
    """Compute perplexity delta — pure Python."""
    baseline_argmax = [_argmax(baseline_logits[i]) for i in range(n_tokens)]

    def ce_for_targets(logits, targets):
        total_ce = 0.0
        for i in range(n_tokens):
            row = logits[i]
            max_val = max(row)
            sum_exp = sum(math.exp(x - max_val) for x in row)
            log_sum_exp = max_val + math.log(sum_exp)
            target_logit = row[targets[i]]
            total_ce += -(target_logit - log_sum_exp)
        return total_ce / n_tokens

    baseline_ce = ce_for_targets(baseline_logits, baseline_argmax)
    candidate_ce = ce_for_targets(candidate_logits, baseline_argmax)

    baseline_ppl = math.exp(min(baseline_ce, 100.0))
    candidate_ppl = math.exp(min(candidate_ce, 100.0))

    return {
        "baseline_perplexity": round(baseline_ppl, 4),
        "candidate_perplexity": round(candidate_ppl, 4),
        "perplexity_delta": round(candidate_ppl - baseline_ppl, 4),
        "perplexity_ratio": round(candidate_ppl / baseline_ppl, 6) if baseline_ppl > 0 else None,
    }


def compute_perplexity_delta_np(baseline_logits, candidate_logits):
    """Compute perplexity delta — numpy."""
    n_tokens = baseline_logits.shape[0]
    baseline_argmax = np.argmax(baseline_logits, axis=-1)

    def ce_for_targets(logits, targets):
        max_vals = logits.max(axis=-1)
        shifted = logits - max_vals[:, None]
        log_sum_exp = max_vals + np.log(np.sum(np.exp(shifted), axis=-1))
        target_logits = logits[np.arange(n_tokens), targets]
        return float(np.mean(-(target_logits - log_sum_exp)))

    baseline_ce = ce_for_targets(baseline_logits, baseline_argmax)
    candidate_ce = ce_for_targets(candidate_logits, baseline_argmax)

    baseline_ppl = math.exp(min(baseline_ce, 100.0))
    candidate_ppl = math.exp(min(candidate_ce, 100.0))

    return {
        "baseline_perplexity": round(baseline_ppl, 4),
        "candidate_perplexity": round(candidate_ppl, 4),
        "perplexity_delta": round(candidate_ppl - baseline_ppl, 4),
        "perplexity_ratio": round(candidate_ppl / baseline_ppl, 6) if baseline_ppl > 0 else None,
    }


def compare_binary(baseline_path: str, candidate_path: str, top_k: int = 10) -> dict:
    """Load two binary logit files and compute full comparison metrics."""
    b_vocab, b_tokens, baseline_logits = load_binary_logits(baseline_path)
    c_vocab, c_tokens, candidate_logits = load_binary_logits(candidate_path)

    if b_vocab != c_vocab:
        print(
            f"ERROR: vocab size mismatch: baseline={b_vocab}, candidate={c_vocab}",
            file=sys.stderr,
        )
        sys.exit(2)

    n_tokens = min(b_tokens, c_tokens)
    if b_tokens != c_tokens:
        print(
            f"WARNING: token count mismatch (baseline={b_tokens}, candidate={c_tokens}), "
            f"using first {n_tokens} positions",
            file=sys.stderr,
        )
        if HAS_NUMPY:
            baseline_logits = baseline_logits[:n_tokens]
            candidate_logits = candidate_logits[:n_tokens]
        else:
            baseline_logits = baseline_logits[:n_tokens]
            candidate_logits = candidate_logits[:n_tokens]

    if HAS_NUMPY:
        kl = compute_kl_divergence_np(baseline_logits, candidate_logits)
        tj = compute_topk_jaccard_np(baseline_logits, candidate_logits, top_k)
        rd = compute_rank_delta_np(baseline_logits, candidate_logits)
        ta = compute_token_agreement_np(baseline_logits, candidate_logits)
        pd = compute_perplexity_delta_np(baseline_logits, candidate_logits)
    else:
        kl = compute_kl_divergence_pure(baseline_logits, candidate_logits, n_tokens)
        tj = compute_topk_jaccard_pure(baseline_logits, candidate_logits, n_tokens, top_k)
        rd = compute_rank_delta_pure(baseline_logits, candidate_logits, n_tokens)
        ta = compute_token_agreement_pure(baseline_logits, candidate_logits, n_tokens)
        pd = compute_perplexity_delta_pure(baseline_logits, candidate_logits, n_tokens)

    return {
        "n_vocab": int(b_vocab),
        "n_tokens": int(n_tokens),
        "kl_divergence": kl,
        "topk_jaccard": tj,
        "rank_delta": rd,
        "token_agreement": ta,
        "perplexity_delta": pd,
    }


# --- JSONL mode (original) ---


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

    parser = argparse.ArgumentParser(
        description="Compute quality metrics from teacher_eval output"
    )
    subparsers = parser.add_subparsers(dest="mode")

    # JSONL mode
    jsonl_parser = subparsers.add_parser("jsonl", help="Compute metrics from JSONL output")
    jsonl_parser.add_argument("jsonl_file", help="Path to teacher_eval JSONL output")
    jsonl_parser.add_argument("--baseline", help="Path to baseline JSON for threshold comparison")
    jsonl_parser.add_argument("--output", "-o", help="Write result JSON to this path")
    jsonl_parser.add_argument(
        "--tolerance", type=float, help="Override tolerance (default from baseline or 0.05)"
    )

    # Binary comparison mode
    cmp_parser = subparsers.add_parser(
        "compare", help="Compare two binary logit files (llama.cpp format)"
    )
    cmp_parser.add_argument("baseline_bin", help="Path to baseline binary logit file")
    cmp_parser.add_argument("candidate_bin", help="Path to candidate binary logit file")
    cmp_parser.add_argument(
        "--top-k", type=int, default=10, help="k for top-k Jaccard overlap (default: 10)"
    )
    cmp_parser.add_argument("--output", "-o", help="Write result JSON to this path")
    cmp_parser.add_argument(
        "--max-kl-mean", type=float, default=None,
        help="Fail if KL divergence mean exceeds this threshold"
    )
    cmp_parser.add_argument(
        "--min-top1-agreement", type=float, default=None,
        help="Fail if top-1 token agreement drops below this (0.0-1.0)"
    )
    cmp_parser.add_argument(
        "--min-jaccard-mean", type=float, default=None,
        help="Fail if mean top-k Jaccard overlap drops below this (0.0-1.0)"
    )

    # Backwards compatibility: if first arg isn't a known subcommand, treat as legacy JSONL
    if len(sys.argv) > 1 and sys.argv[1] not in ("jsonl", "compare", "-h", "--help"):
        legacy = argparse.ArgumentParser(description="Compute quality metrics (legacy mode)")
        legacy.add_argument("jsonl", help="Path to teacher_eval JSONL output")
        legacy.add_argument("--baseline", help="Path to baseline JSON for threshold comparison")
        legacy.add_argument("--output", "-o", help="Write result JSON to this path")
        legacy.add_argument(
            "--tolerance", type=float, help="Override tolerance (default from baseline or 0.05)"
        )
        args = legacy.parse_args()
        args.mode = "jsonl"
        args.jsonl_file = args.jsonl
    else:
        args = parser.parse_args()
        if args.mode is None:
            parser.print_help()
            sys.exit(2)

    if args.mode == "jsonl":
        if not Path(args.jsonl_file).exists():
            print(f"ERROR: input file not found: {args.jsonl_file}", file=sys.stderr)
            sys.exit(2)

        metrics = compute_metrics(args.jsonl_file)

        print("=== Quality Metrics ===")
        print(f"  tokens evaluated:    {metrics['n_tokens']}")
        print(f"  mean cross-entropy:  {metrics['mean_cross_entropy']:.6f}")
        print(f"  median cross-entropy:{metrics['median_cross_entropy']:.6f}")
        print(f"  CE stddev:           {metrics['cross_entropy_stddev']:.6f}")
        print(f"  perplexity:          {metrics['perplexity']:.4f}")
        print(f"  top-1 accuracy:      {metrics['top1_accuracy']*100:.2f}%")
        print(f"  top-5 accuracy:      {metrics['top5_accuracy']*100:.2f}%")

        result = {"metrics": metrics, "status": "PASS", "failures": []}

        if hasattr(args, "baseline") and args.baseline:
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
                    print("\n=== QUALITY REGRESSION DETECTED ===")
                    for fail in failures:
                        print(f"  FAIL: {fail}")
                else:
                    print("\n=== All metrics within thresholds ===")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
                f.write("\n")

        sys.exit(1 if result["status"] == "FAIL" else 0)

    elif args.mode == "compare":
        if not Path(args.baseline_bin).exists():
            print(f"ERROR: baseline file not found: {args.baseline_bin}", file=sys.stderr)
            sys.exit(2)
        if not Path(args.candidate_bin).exists():
            print(f"ERROR: candidate file not found: {args.candidate_bin}", file=sys.stderr)
            sys.exit(2)

        metrics = compare_binary(args.baseline_bin, args.candidate_bin, top_k=args.top_k)

        print("=== Binary Logit Comparison ===")
        print(f"  vocab size:          {metrics['n_vocab']}")
        print(f"  tokens compared:     {metrics['n_tokens']}")
        print()
        kl = metrics["kl_divergence"]
        print("  KL divergence:")
        print(f"    mean:              {kl['mean']:.6f}")
        print(f"    max:               {kl['max']:.6f}")
        print(f"    p95:               {kl['p95']:.6f}")
        print(f"    median:            {kl['median']:.6f}")
        print()
        tj = metrics["topk_jaccard"]
        print(f"  Top-{tj['k']} Jaccard overlap:")
        print(f"    mean:              {tj['mean']:.4f}")
        print(f"    min:               {tj['min']:.4f}")
        print(f"    p5:                {tj['p5']:.4f}")
        print()
        rd = metrics["rank_delta"]
        print("  Rank delta (baseline argmax in candidate):")
        print(f"    mean:              {rd['mean']:.2f}")
        print(f"    max:               {rd['max']:.0f}")
        print(f"    median:            {rd['median']:.1f}")
        print(f"    p95:               {rd['p95']:.1f}")
        print()
        ta = metrics["token_agreement"]
        print("  Token agreement:")
        print(f"    top-1 agreement:   {ta['top1_agreement']*100:.2f}%")
        print()
        pd = metrics["perplexity_delta"]
        print("  Perplexity (using baseline argmax as reference):")
        print(f"    baseline:          {pd['baseline_perplexity']:.4f}")
        print(f"    candidate:         {pd['candidate_perplexity']:.4f}")
        print(f"    delta:             {pd['perplexity_delta']:+.4f}")
        if pd["perplexity_ratio"] is not None:
            print(f"    ratio:             {pd['perplexity_ratio']:.6f}")

        # Threshold checks for golden comparison gating
        failures = []
        if args.max_kl_mean is not None and kl["mean"] > args.max_kl_mean:
            failures.append(
                f"kl_divergence.mean: {kl['mean']:.6f} > threshold {args.max_kl_mean:.6f}"
            )
        if args.min_top1_agreement is not None and ta["top1_agreement"] < args.min_top1_agreement:
            failures.append(
                f"top1_agreement: {ta['top1_agreement']:.4f} < threshold {args.min_top1_agreement:.4f}"
            )
        if args.min_jaccard_mean is not None and tj["mean"] < args.min_jaccard_mean:
            failures.append(
                f"topk_jaccard.mean: {tj['mean']:.4f} < threshold {args.min_jaccard_mean:.4f}"
            )

        if failures:
            print("\n=== GOLDEN COMPARISON REGRESSION DETECTED ===")
            for fail in failures:
                print(f"  FAIL: {fail}")
            result = {"metrics": metrics, "status": "FAIL", "failures": failures}
        else:
            if args.max_kl_mean is not None or args.min_top1_agreement is not None or args.min_jaccard_mean is not None:
                print("\n=== Golden comparison: all metrics within thresholds ===")
            result = {"metrics": metrics, "status": "PASS", "failures": []}

        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
                f.write("\n")

        sys.exit(1 if result["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
