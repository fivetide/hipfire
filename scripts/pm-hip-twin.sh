#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Kaden Schutt
# hipfire — see LICENSE and NOTICE in the project root.
#
# Route A seal of the gfx1100 hipcc twins into embedded `kernels/*.hxaco`:
#   kernels/gated_delta_vector_pm_gfx1100.hip          (RDA serial + depthwise conv rows)
#   kernels/gemm_mq4g256v2_moe_grouped_k2_pm_gfx1100.hip
#
# Each source is compiled with EXACTLY the runtime JIT argv (crates/rdna-compute/src/compiler.rs
# `core_hipcc_flags` + the rocm path flags the card judge recorded in candidate.co.json), the
# shipped incumbent source (kernels/src) is compiled the same way as the kernarg `--abi-ref`, and
# the `hip_twin` example seals every twin symbol (lift identity via emit::checked, hazards,
# SGPR >= 16, kernarg ABI == incumbent). Each twin symbol's `.text` sha256 must equal the
# card-judged candidate's `code_sha256`; a mismatch aborts (the rename must not change code).
#
# Writes kernels/<module>.hxaco, kernels/<module>.hxaco.sha256 and kernels/<module>.twin.json.
#
# Env: ROCM_PATH (default /opt/rocm/core), CARGO_TARGET_DIR, PM_TWIN_WORK (scratch dir, default
#      <repo>/target/pm-hip-twin), PM_TWIN_JUDGE_OBJ (the r1card obj/ directory holding
#      <node>-<shape>-<cand>/candidate.co.json).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM="${ROCM_PATH:-/opt/rocm/core}"
WORK="${PM_TWIN_WORK:-$REPO/target/pm-hip-twin}"
JUDGE="${PM_TWIN_JUDGE_OBJ:-/home/kaden/qcal/release-0.4.3/raon/scratch-pm/r1card/obj}"
ARCH=gfx1100
mkdir -p "$WORK"

# Mirror of compiler.rs `core_hipcc_flags(arch)` followed by the recorded rocm path flags.
HIPCC_FLAGS=(
  --genco "--offload-arch=$ARCH" -O3 --no-offload-compress -fuse-cuid=none -mcode-object-version=6
  "--rocm-path=$ROCM" "--hip-path=$ROCM" "-I$ROCM/include"
)

hipcc_compile() { # src out
  local src="$1" out="$2"
  "$ROCM/bin/hipcc" "${HIPCC_FLAGS[@]}" "-fmacro-prefix-map=$(dirname "$src")/=" -o "$out" "$src"
}

( cd "$REPO" && cargo build --release -p peacemaker-isa --example hip_twin )
TWIN_BIN="${CARGO_TARGET_DIR:-$REPO/target}/release/examples/hip_twin"

# seal_module <module> <twin-src> <spec>...   spec = twin_sym:incumbent_src:incumbent_sym:judge_dir:judge_sym
seal_module() {
  local module="$1" src="$2"
  shift 2
  local dir="$WORK/$module"
  mkdir -p "$dir"
  hipcc_compile "$REPO/kernels/$src" "$dir/twin.co"
  local args=(--in "$dir/twin.co" --arch "$ARCH" --out "$REPO/kernels/$module.hxaco" --report "$dir/report.json")
  local spec i=0 tsym isrc isym
  for spec in "$@"; do
    IFS=: read -r tsym isrc isym _ _ <<<"$spec"
    hipcc_compile "$REPO/kernels/src/$isrc" "$dir/incumbent$i.co"
    args+=(--kernel "$tsym" --abi-ref "$dir/incumbent$i.co" --abi-ref-symbol "$isym")
    i=$((i + 1))
  done
  "$TWIN_BIN" "${args[@]}" >/dev/null
  python3 - "$dir/report.json" "$REPO/kernels/$module.twin.json" "$module" "$JUDGE" "$@" <<'PY'
import json, sys
report_path, out_path, module, judge, *specs = sys.argv[1:]
reports = json.load(open(report_path))
if isinstance(reports, dict):
    reports = [reports]
assert len(reports) == len(specs), (len(reports), len(specs))
judged, bad = [], []
for rep, spec in zip(reports, specs):
    tsym, _isrc, _isym, jdir, jsym = spec.split(":")
    want = json.load(open(f"{judge}/{jdir}/candidate.co.json"))["kernels"][jsym]["code_sha256"]
    got = rep["text_sha256"]
    judged.append({"symbol": tsym, "judged_object": jdir, "judged_symbol": jsym,
                   "judged_code_sha256": want, "twin_text_sha256": got, "equal": want == got})
    if want != got:
        bad.append(f"{tsym}: twin .text {got} != judged {want} ({jdir})")
with open(out_path, "w") as f:
    json.dump({"module": module, "arch": "gfx1100", "reports": reports, "judged_text_check": judged}, f, indent=2)
    f.write("\n")
if bad:
    sys.exit("pm-hip-twin: rename/compile changed .text vs the card-judged candidate:\n  " + "\n  ".join(bad))
PY
  echo "sealed kernels/$module.hxaco sha256 $(cat "$REPO/kernels/$module.hxaco.sha256")"
}

seal_module gated_delta_vector_pm_gfx1100 gated_delta_vector_pm_gfx1100.hip \
  "gated_delta_net_vector_serial_f32_pm_gfx1100:gated_delta_net_vector_serial.hip:gated_delta_net_vector_serial_f32:rda_serial-pp512-R1Rda-05-dpp-lockstep-lanestore:gated_delta_net_vector_serial_f32" \
  "causal_conv_depthwise_silu_rows_f32_pm_gfx1100:causal_conv_depthwise_rows.hip:causal_conv_depthwise_silu_rows_f32:conv_rows-pp512-R1Conv-04-rpw32-k4:causal_conv_depthwise_silu_rows_f32"

seal_module gemm_mq4g256v2_moe_grouped_k2_pm_gfx1100 gemm_mq4g256v2_moe_grouped_k2_pm_gfx1100.hip \
  "gemm_mq4g256v2_moe_grouped_wmma_k2_pm_gfx1100:gemm_mq4g256v2_moe_grouped_wmma_k2.hip:gemm_mq4g256v2_moe_grouped_wmma_k2:moe_k2-gu512-R1Moe-04-xlds-w4-t2:gemm_mq4g256v2_moe_grouped_wmma_k2"
