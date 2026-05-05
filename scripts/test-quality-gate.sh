#!/usr/bin/env bash
# Integration test for the quality gate pipeline.
#
# Validates that:
#   1. teacher_eval builds successfully
#   2. quality_metrics.py handles valid/invalid input correctly
#   3. quality-gate.sh orchestrator runs end-to-end (if models present)
#
# Exit codes:
#   0  all tests pass
#   1  test failure
#   2  environment error

set -u
cd "$(dirname "$0")/.."

PASS=0
FAIL=0

ok() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=== Quality Gate Integration Tests ==="
echo

# ── Test 1: teacher_eval builds ─────────────────────────────────────────
echo "-- Test 1: teacher_eval builds"
if cargo build --release --example teacher_eval --features deltanet 2>&1 | tail -3; then
    if [ -x "./target/release/examples/teacher_eval" ]; then
        ok "teacher_eval binary built"
    else
        fail "teacher_eval binary not executable"
    fi
else
    fail "teacher_eval build failed"
fi
echo

# ── Test 2: quality_metrics.py with synthetic data ──────────────────────
echo "-- Test 2: quality_metrics.py with valid JSONL"
tmpdir=$(mktemp -d)
cat > "$tmpdir/test.jsonl" <<'EOF'
{"pos":0,"ref_token":100,"top1":100,"top1_correct":true,"top5_correct":true,"cross_entropy":0.5,"top1_logit":5.0,"ref_logit":5.0,"ref_rank":1}
{"pos":1,"ref_token":200,"top1":201,"top1_correct":false,"top5_correct":true,"cross_entropy":1.2,"top1_logit":4.0,"ref_logit":3.8,"ref_rank":3}
{"pos":2,"ref_token":300,"top1":300,"top1_correct":true,"top5_correct":true,"cross_entropy":0.3,"top1_logit":6.0,"ref_logit":6.0,"ref_rank":1}
{"pos":3,"ref_token":400,"top1":401,"top1_correct":false,"top5_correct":false,"cross_entropy":3.5,"top1_logit":2.0,"ref_logit":0.5,"ref_rank":12}
{"pos":4,"ref_token":500,"top1":500,"top1_correct":true,"top5_correct":true,"cross_entropy":0.8,"top1_logit":4.5,"ref_logit":4.5,"ref_rank":1}
EOF

if python3 scripts/quality_metrics.py "$tmpdir/test.jsonl" -o "$tmpdir/result.json" 2>/dev/null; then
    # Verify expected metrics
    top1=$(python3 -c "import json; d=json.load(open('$tmpdir/result.json')); print(d['metrics']['top1_accuracy'])")
    if [ "$top1" = "0.6" ]; then
        ok "quality_metrics.py computes correct top-1 accuracy (0.6)"
    else
        fail "quality_metrics.py top-1 accuracy wrong: got $top1, expected 0.6"
    fi
else
    fail "quality_metrics.py failed on valid input"
fi
echo

# ── Test 3: quality_metrics.py threshold check (pass) ───────────────────
echo "-- Test 3: quality_metrics.py threshold pass"
cat > "$tmpdir/baseline.json" <<'EOF'
{"tolerance": 0.10, "mean_cross_entropy": 1.3, "perplexity": 3.7, "top1_accuracy": 0.55, "top5_accuracy": 0.55}
EOF

if python3 scripts/quality_metrics.py "$tmpdir/test.jsonl" --baseline "$tmpdir/baseline.json" 2>/dev/null; then
    ok "quality_metrics.py passes within threshold"
else
    fail "quality_metrics.py should have passed (metrics within tolerance)"
fi
echo

# ── Test 4: quality_metrics.py threshold check (fail) ───────────────────
echo "-- Test 4: quality_metrics.py threshold fail"
cat > "$tmpdir/strict_baseline.json" <<'EOF'
{"tolerance": 0.01, "mean_cross_entropy": 0.5, "perplexity": 1.6, "top1_accuracy": 0.9, "top5_accuracy": 0.95}
EOF

if python3 scripts/quality_metrics.py "$tmpdir/test.jsonl" --baseline "$tmpdir/strict_baseline.json" 2>/dev/null; then
    fail "quality_metrics.py should have failed (metrics exceed strict baseline)"
else
    ok "quality_metrics.py correctly detects regression"
fi
echo

# ── Test 5: quality_metrics.py with empty input ─────────────────────────
echo "-- Test 5: quality_metrics.py rejects empty input"
echo "" > "$tmpdir/empty.jsonl"
if python3 scripts/quality_metrics.py "$tmpdir/empty.jsonl" 2>/dev/null; then
    fail "quality_metrics.py should reject empty input"
else
    ok "quality_metrics.py correctly rejects empty input"
fi
echo

# ── Test 6: reference texts exist ──────────────────────────────────────
echo "-- Test 6: reference texts present"
ref_count=$(find benchmarks/quality-gate -name "*.txt" 2>/dev/null | wc -l)
if [ "$ref_count" -gt 0 ]; then
    ok "found $ref_count reference text(s) in benchmarks/quality-gate/"
else
    fail "no reference texts in benchmarks/quality-gate/"
fi
echo

# ── Test 7: quality-gate.sh is executable and parses args ───────────────
echo "-- Test 7: quality-gate.sh help"
if bash scripts/quality-gate.sh --help >/dev/null 2>&1; then
    ok "quality-gate.sh --help works"
else
    fail "quality-gate.sh --help failed"
fi
echo

# ── Cleanup ─────────────────────────────────────────────────────────────
rm -rf "$tmpdir"

# ── Summary ─────────────────────────────────────────────────────────────
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
