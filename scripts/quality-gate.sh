#!/usr/bin/env bash
# quality-gate.sh — Teacher-forced quality regression gate.
#
# Runs teacher_eval on reference texts against each available model and
# compares cross-entropy/accuracy metrics to committed baselines.
# Catches quality regressions from quantization changes, kernel bugs,
# or numerical drift that wouldn't trigger the coherence gate.
#
# Exit codes:
#   0  all metrics within thresholds
#   1  quality regression detected
#   2  build or environment error
#
# Modes:
#   ./scripts/quality-gate.sh              # default (smallest available model)
#   ./scripts/quality-gate.sh --full       # all available models
#   ./scripts/quality-gate.sh --update     # capture new baselines
#   ./scripts/quality-gate.sh --tolerance 0.08  # override tolerance (default 0.05)
#
# Reference texts live in benchmarks/quality-gate/. Baselines in
# tests/quality-baselines/<arch>/<model>.json.

set -u
cd "$(dirname "$0")/.."

FULL=0
UPDATE=0
TOLERANCE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --full) FULL=1 ;;
        --update|--update-baselines) UPDATE=1 ;;
        --tolerance) TOLERANCE="$2"; shift ;;
        -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

EXE="./target/release/examples/teacher_eval"
LOCK_SCRIPT="./scripts/gpu-lock.sh"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
REFERENCE_DIR="benchmarks/quality-gate"
BASELINE_DIR="tests/quality-baselines"

# ── Arch detection (reuse from speed-gate pattern) ──────────────────────
BASELINE_ARCH=""
if [ -n "${HIPFIRE_BASELINE_ARCH:-}" ]; then
    BASELINE_ARCH="$HIPFIRE_BASELINE_ARCH"
else
    for probe in amdgpu-arch offload-arch \
                 /opt/rocm/bin/amdgpu-arch /opt/rocm/bin/offload-arch \
                 /opt/rocm/llvm/bin/amdgpu-arch; do
        if command -v "$probe" >/dev/null 2>&1 || [ -x "$probe" ]; then
            BASELINE_ARCH="$("$probe" 2>/dev/null | head -1)"
            if [ -n "$BASELINE_ARCH" ]; then break; fi
        fi
    done
    if [ -z "$BASELINE_ARCH" ]; then
        for node_props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
            [ -f "$node_props" ] || continue
            ver=$(awk '/gfx_target_version/ {print $2; exit}' "$node_props" 2>/dev/null || true)
            case "$ver" in
                90006)          BASELINE_ARCH="gfx906";  break ;;
                90008)          BASELINE_ARCH="gfx908";  break ;;
                100100)         BASELINE_ARCH="gfx1010"; break ;;
                100300|100302)  BASELINE_ARCH="gfx1030"; break ;;
                110000|110001)  BASELINE_ARCH="gfx1100"; break ;;
                110501)         BASELINE_ARCH="gfx1151"; break ;;
                120000)         BASELINE_ARCH="gfx1200"; break ;;
                120001)         BASELINE_ARCH="gfx1201"; break ;;
            esac
        done
    fi
    if [ -z "$BASELINE_ARCH" ] && command -v rocminfo >/dev/null 2>&1; then
        BASELINE_ARCH="$(rocminfo 2>/dev/null | awk '/^  Name:/ && $2 ~ /^gfx/ {print $2; exit}')"
    fi
fi
case "${HSA_OVERRIDE_GFX_VERSION:-}" in
    9.0.6|9.0) BASELINE_ARCH="gfx906" ;;
    10.1.0|10.1) BASELINE_ARCH="gfx1010" ;;
    10.3.0|10.3) BASELINE_ARCH="gfx1030" ;;
    11.0.0|11.0) BASELINE_ARCH="gfx1100" ;;
esac
if [ -z "$BASELINE_ARCH" ]; then
    echo "quality-gate: cannot detect GPU arch — set HIPFIRE_BASELINE_ARCH=gfxNNNN" >&2
    exit 2
fi
echo "quality-gate: arch=$BASELINE_ARCH"

# ── Build teacher_eval ──────────────────────────────────────────────────
rebuild=0
if [ ! -x "$EXE" ]; then
    rebuild=1
else
    for src in crates/hipfire-arch-qwen35/src/qwen35.rs crates/hipfire-runtime/src/llama.rs \
               crates/hipfire-runtime/src/hfq.rs crates/hipfire-runtime/examples/teacher_eval.rs \
               crates/rdna-compute/src/dispatch.rs; do
        if [ -f "$src" ] && [ "$src" -nt "$EXE" ]; then
            rebuild=1
            break
        fi
    done
fi
if [ "$rebuild" -eq 1 ]; then
    echo "quality-gate: building teacher_eval..."
    if ! cargo build --release --example teacher_eval --features deltanet >&2; then
        echo "quality-gate: build failed" >&2
        exit 2
    fi
fi

# ── Reference texts ─────────────────────────────────────────────────────
if [ ! -d "$REFERENCE_DIR" ]; then
    echo "quality-gate: reference dir $REFERENCE_DIR not found" >&2
    exit 2
fi

# ── GPU lock ────────────────────────────────────────────────────────────
if [ -r "$LOCK_SCRIPT" ]; then
    # shellcheck disable=SC1090
    . "$LOCK_SCRIPT"
    gpu_acquire "quality-gate" || { echo "could not acquire GPU lock" >&2; exit 2; }
    trap 'gpu_release 2>/dev/null || true' EXIT
fi

# ── Model matrix ────────────────────────────────────────────────────────
# Models to evaluate. Short mode: smallest available. Full: all.
if [ "$FULL" -eq 1 ]; then
    MODEL_SIZES=("0.8b" "4b" "9b")
else
    MODEL_SIZES=("0.8b")
fi

# ── Run evaluations ─────────────────────────────────────────────────────
pass=0
fail=0
skip=0

for size in "${MODEL_SIZES[@]}"; do
    model_path="$MODELS_DIR/qwen3.5-${size}.mq4"
    if [ ! -f "$model_path" ]; then
        echo "  qwen3.5-${size}.mq4: SKIP (model not present)"
        skip=$((skip + 1))
        continue
    fi

    for ref_file in "$REFERENCE_DIR"/*.txt; do
        [ -f "$ref_file" ] || continue
        ref_name=$(basename "$ref_file" .txt)
        baseline_file="$BASELINE_DIR/$BASELINE_ARCH/${size}_${ref_name}.json"
        out_file="/tmp/quality-gate_${size}_${ref_name}_$$.jsonl"

        echo "== qwen3.5-${size}.mq4 / $ref_name =="

        # Run teacher_eval
        if ! "$EXE" "$model_path" "$ref_file" "$out_file" 2>&1 | grep -E "^  (pos|tokens|QUALITY)" | tail -3; then
            echo "  HARD_ERROR: teacher_eval crashed"
            fail=$((fail + 1))
            rm -f "$out_file"
            continue
        fi

        if [ ! -s "$out_file" ]; then
            echo "  HARD_ERROR: no output produced"
            fail=$((fail + 1))
            rm -f "$out_file"
            continue
        fi

        if [ "$UPDATE" -eq 1 ]; then
            # Capture baseline
            mkdir -p "$(dirname "$baseline_file")"
            tol_arg=""
            [ -n "$TOLERANCE" ] && tol_arg="--tolerance $TOLERANCE"
            python3 scripts/quality_metrics.py "$out_file" -o "$baseline_file" $tol_arg
            # Inject tolerance into baseline
            if [ -n "$TOLERANCE" ]; then
                python3 -c "
import json, sys
with open('$baseline_file') as f: d = json.load(f)
d['tolerance'] = $TOLERANCE
d.update(d.pop('metrics', {}))
with open('$baseline_file','w') as f: json.dump(d, f, indent=2); f.write('\n')
"
            else
                python3 -c "
import json
with open('$baseline_file') as f: d = json.load(f)
d['tolerance'] = 0.05
d.update(d.pop('metrics', {}))
with open('$baseline_file','w') as f: json.dump(d, f, indent=2); f.write('\n')
"
            fi
            echo "  baseline written: $baseline_file"
        else
            # Check against baseline
            if [ ! -f "$baseline_file" ]; then
                echo "  NO BASELINE (generate with --update)"
                skip=$((skip + 1))
            else
                tol_arg=""
                [ -n "$TOLERANCE" ] && tol_arg="--tolerance $TOLERANCE"
                if python3 scripts/quality_metrics.py "$out_file" --baseline "$baseline_file" $tol_arg; then
                    pass=$((pass + 1))
                else
                    fail=$((fail + 1))
                fi
            fi
        fi

        rm -f "$out_file"
    done
done

# ── Summary ─────────────────────────────────────────────────────────────
echo
if [ "$UPDATE" -eq 1 ]; then
    echo "quality-gate: baselines updated in $BASELINE_DIR/$BASELINE_ARCH/"
    echo "Review with: git diff $BASELINE_DIR/"
    exit 0
fi

if [ "$fail" -eq 0 ]; then
    if [ "$pass" -eq 0 ] && [ "$skip" -gt 0 ]; then
        echo "quality-gate: NO METRICS CHECKED (${skip} skipped)"
        exit 0
    fi
    echo "quality-gate: PASSED (${pass} checks, ${skip} skipped)"
    exit 0
fi

echo "quality-gate: FAILED (${fail} regression(s), ${pass} passed, ${skip} skipped)"
exit 1
