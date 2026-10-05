#!/usr/bin/env bash
# Online DFlash draft tuning (halo, gfx1151): Qwen3.5-9B MQ4 + 9B DFlash MQ4 draft, greedy.
#
# Primary (deterministic): replay_tau_ratio = geomean over the 9 committed long-session prompts
#   (benchmarks/prompts/online_tune/) of tuner-picked tau / argmax tau on the recorded argmax
#   session (same text, same block starts; dumps made once with HIPFIRE_DFLASH_ONLINE_TUNE=stats).
# Online check: paired A/B (A = stats/argmax, B = on) `hipfire bench --spec dflash`, alternating
#   order, on 3 prompts -> geomean tau ratio and decode tok/s ratio. Online tau compares different
#   texts (batched-verify numerics flip near-ties when acceptance moves), so it is noisier.
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=$HOME/.hipfire/rocm-merged/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
unset HIPFIRE_DFLASH_ONLINE_TUNE HIPFIRE_DFLASH_ONLINE_HP HIPFIRE_DFLASH_ONLINE_DUMP

cargo build -q --release 2>/dev/null
cargo build -q --release -p hipfire-runtime --example dflash_online_replay 2>/dev/null
HF=target/release/hipfire
REPLAY=target/release/examples/dflash_online_replay
MODEL=$HOME/.hipfire/models/qwen3.5-9b.mq4
P=benchmarks/prompts/online_tune
DUMPS=$HOME/.cache/hipfire-online/v2

# Pause the user's llama-server (GPU contention) for the run.
kill -STOP 1500 2>/dev/null || true
trap 'kill -CONT 1500 2>/dev/null || true' EXIT

out=$(mktemp -d)
bench() { # tag prompt-file mode [dump]
  env HIPFIRE_DFLASH_ONLINE_TUNE="$3" ${4:+HIPFIRE_DFLASH_ONLINE_DUMP=$4} \
    timeout --foreground 900 "$HF" bench "$MODEL" --spec dflash --runs 1 --warmups 0 \
    --max-tokens 1536 --backend noslots --workload stateless \
    --prompt-file "$2" --json >"$out/$1.json" 2>"$out/$1.err" \
    || { tail -30 "$out/$1.err" >&2; echo "bench $1 FAILED" >&2; exit 1; }
}

# One-time argmax session dumps (keyed by prompt md5).
mkdir -p "$DUMPS"
dumps=()
for f in "$P"/*.txt; do
  d="$DUMPS/$(basename "$f" .txt)-$(md5sum <"$f" | cut -c1-12).bin"
  [ -s "$d" ] || bench "dump-$(basename "$f" .txt)" "$f" stats "$d"
  dumps+=("$d")
done
"$REPLAY" "${dumps[@]}" >"$out/replay.txt"
cat "$out/replay.txt" >&2

# Online paired A/B, alternating order.
i=0
for n in prose_lighthouses prose_letter code_inventory; do
  if (( i % 2 == 0 )); then bench "$n.A" "$P/$n.txt" stats; bench "$n.B" "$P/$n.txt" on
  else bench "$n.B" "$P/$n.txt" on; bench "$n.A" "$P/$n.txt" stats; fi
  i=$((i + 1))
done

python3 - "$out" <<'EOF'
import json, math, os, re, sys
d = sys.argv[1]
rep = open(os.path.join(d, "replay.txt")).read()
geo = float(re.search(r"GEOMEAN picked/argmax tau over \d+ sessions: ([0-9.]+)", rep).group(1))
print("METRIC replay_tau_ratio=%.4f" % geo)
for path, base, pick in re.findall(r"/([a-z_]+)-[0-9a-f]+\.bin: cycles=\d+ argmax_tau=([0-9.]+) picked_tau=([0-9.]+)", rep):
    print("METRIC replay_%s=%.4f" % (path, float(pick) / float(base)))
def run(tag):
    j = json.load(open(os.path.join(d, tag + ".json")))
    err = open(os.path.join(d, tag + ".err")).read()
    toks = re.findall(r"decode \((\d+) tok", err)[-1]
    return j["spec_tau"][-1], j["decode_tok_s"]["median"], toks
tr, sr = [], []
for n in ("prose_lighthouses", "prose_letter", "code_inventory"):
    a, b = run(n + ".A"), run(n + ".B")
    tr.append(b[0] / a[0]); sr.append(b[1] / a[1])
    print("METRIC %s_tau_A=%.3f" % (n, a[0])); print("METRIC %s_tau_B=%.3f" % (n, b[0]))
    print("METRIC %s_tok_s_A=%.2f" % (n, a[1])); print("METRIC %s_tok_s_B=%.2f" % (n, b[1]))
    print("ASI %s_tokens_A_B=%s/%s" % (n, a[2], b[2]))
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
print("METRIC online_tau_ratio=%.4f" % g(tr))
print("METRIC online_tok_s_ratio=%.4f" % g(sr))
EOF
rm -rf "$out"
