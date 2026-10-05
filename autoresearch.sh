#!/usr/bin/env bash
# Online DFlash draft tuning (halo, gfx1151): Qwen3.5-9B MQ4 + 9B DFlash MQ4 draft, greedy,
# `hipfire bench --spec dflash` with HIPFIRE_DFLASH_ONLINE_TUNE=on (per-request tuner, no
# cross-request learning). Primary: mean acceptance tau over three committed prose prompts
# (greedy => deterministic). Secondaries: prose decode tok/s, code tau / tok/s (control),
# generated token counts (lossless proxy: must not change).
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=$HOME/.hipfire/rocm-merged/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export HIPFIRE_DFLASH_ONLINE_TUNE=${HIPFIRE_DFLASH_ONLINE_TUNE:-on}

cargo build -q --release 2>/dev/null
HF=target/release/hipfire
MODEL=$HOME/.hipfire/models/qwen3.5-9b.mq4

# Pause the user's llama-server (GPU contention) for the run.
kill -STOP 1500 2>/dev/null || true
trap 'kill -CONT 1500 2>/dev/null || true' EXIT

out=$(mktemp -d)
bench() { # name prompt max_tokens
  timeout --foreground 900 "$HF" bench "$MODEL" --spec dflash --runs 2 --warmups 1 \
    --max-tokens "$3" --backend noslots --workload stateless \
    --prompt-file "benchmarks/prompts/$2" --json >"$out/$1.json" 2>"$out/$1.err" \
    || { tail -30 "$out/$1.err" >&2; echo "bench $1 FAILED" >&2; exit 1; }
}
bench river prose_river_short.txt 1024
bench light fiction_lighthouse.txt 1024
bench essay prose_long_lighthouses.txt 1536
bench code merge_sort_thinking_off.txt 1024

python3 - "$out" <<'EOF'
import json, re, statistics as st, sys, os
d = sys.argv[1]
def load(n):
    j = json.load(open(os.path.join(d, n + ".json")))
    err = open(os.path.join(d, n + ".err")).read()
    toks = re.findall(r"tau=[0-9.]+ tok/s=[0-9.]+ decode \((\d+) tok", err)[-2:]
    return st.mean(j["spec_tau"]), j["decode_tok_s"]["median"], toks, j["prompt_md5"]
prose = {n: load(n) for n in ("river", "light", "essay")}
code = load("code")
print("METRIC prose_tau=%.4f" % st.mean(v[0] for v in prose.values()))
print("METRIC prose_tok_s=%.2f" % st.mean(v[1] for v in prose.values()))
for n, v in prose.items():
    print("METRIC %s_tau=%.4f" % (n, v[0]))
    print("METRIC %s_tok_s=%.2f" % (n, v[1]))
print("METRIC code_tau=%.4f" % code[0])
print("METRIC code_tok_s=%.2f" % code[1])
for n, v in list(prose.items()) + [("code", code)]:
    print("ASI %s_tokens=%s" % (n, "/".join(v[2])))
    print("ASI %s_prompt_md5=%s" % (n, v[3]))
EOF
grep -h 'dflash-online\] request end' "$out"/*.err | tail -8 >&2 || true
rm -rf "$out"
