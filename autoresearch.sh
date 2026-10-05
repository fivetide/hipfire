#!/usr/bin/env bash
# Online DFlash draft tuning (halo, gfx1151), two target/draft pairs, greedy:
#   q9: Qwen3.5-9B MQ4 + qwen35-9b-dflash-mq4 (qwen35 chain DFlash, strong draft)
#   q8: Qwen3-8B MQ4 + qwen3-8b-dflash        (generic llama-family DFlash, weak draft)
#
# Primary (deterministic): replay_tau_ratio = geomean of the two pairs' replay ratios. Per pair:
#   geomean over the 10 committed long-session prompts (benchmarks/prompts/online_tune/) of
#   tuner-picked tau / argmax tau on the recorded argmax session (same text and block starts,
#   prompt-seeded; dumps made once with HIPFIRE_DFLASH_ONLINE_TUNE=stats), each session replayed
#   after the other 9 in one tuner (--carry-loo: weights carried as in a long-running daemon,
#   never trained on the scored session). *_cold = fresh tuner per request.
# Online check: paired A/B (A = shipping DFlash, tuner unset; B = on), alternating order,
#   fresh daemon per run -> geomean tau and decode tok/s ratios (texts may differ at near-ties:
#   batched-verify numerics, so online tau is noisier than replay).
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=$HOME/.hipfire/rocm-merged/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
unset HIPFIRE_DFLASH_ONLINE_TUNE HIPFIRE_DFLASH_ONLINE_HP HIPFIRE_DFLASH_ONLINE_DUMP HIPFIRE_HOME

cargo build -q --release 2>/dev/null
cargo build -q --release -p hipfire-runtime --example dflash_online_replay 2>/dev/null
HF=target/release/hipfire
REPLAY=target/release/examples/dflash_online_replay
P=benchmarks/prompts/online_tune
C=$HOME/.cache/hipfire-online

# q8 needs a non-registry draft: an isolated HIPFIRE_HOME (links to ~/.hipfire) whose
# config sets developer.dflash_draft and speculation.dflash = "on".
Q8HOME=$C/hf8home
if [ ! -f "$Q8HOME/config.toml" ]; then
  mkdir -p "$Q8HOME"
  for e in "$HOME"/.hipfire/*; do [ "$(basename "$e")" = config.toml ] || ln -sfn "$e" "$Q8HOME/$(basename "$e")"; done
  { sed 's/^dflash = "off"/dflash = "on"/' "$HOME/.hipfire/config.toml"
    printf '\n[developer]\ndflash_draft = "%s"\n' "$HOME/.hipfire/models/qwen3-8b-dflash.hfq"; } >"$Q8HOME/config.toml"
fi

declare -A MODEL=([q9]=$HOME/.hipfire/models/qwen3.5-9b.mq4 [q8]=$HOME/.hipfire/models/qwen3-8b.mq4)
declare -A HOMEDIR=([q9]="" [q8]=$Q8HOME)
declare -A DUMPS=([q9]=$C/v3 [q8]=$C/q8)
declare -A MAXTOK=([q9]=1536 [q8]=1024)

kill -STOP 1500 2>/dev/null || true   # user's llama-server: GPU contention
trap 'kill -CONT 1500 2>/dev/null || true' EXIT

out=$(mktemp -d)
bench() { # pair tag prompt-file mode(stats|on|off) [dump]
  local tune=(); [ "$4" = off ] || tune=(HIPFIRE_DFLASH_ONLINE_TUNE="$4")
  env ${HOMEDIR[$1]:+HIPFIRE_HOME=${HOMEDIR[$1]}} "${tune[@]}" ${5:+HIPFIRE_DFLASH_ONLINE_DUMP=$5} \
    timeout --foreground 900 "$HF" bench "${MODEL[$1]}" --spec dflash --runs 1 --warmups 0 \
    --max-tokens "${MAXTOK[$1]}" --backend noslots --workload stateless \
    --prompt-file "$3" --json >"$out/$1.$2.json" 2>"$out/$1.$2.err" \
    || { tail -30 "$out/$1.$2.err" >&2; echo "bench $1 $2 FAILED" >&2; exit 1; }
}

for pair in q9 q8; do
  mkdir -p "${DUMPS[$pair]}"
  dumps=()
  for f in "$P"/*.txt; do
    d="${DUMPS[$pair]}/$(basename "$f" .txt)-$(md5sum <"$f" | cut -c1-12).bin"
    [ -s "$d" ] || bench "$pair" "dump-$(basename "$f" .txt)" "$f" stats "$d"
    dumps+=("$d")
  done
  "$REPLAY" --carry-loo "${dumps[@]}" >"$out/$pair.replay.txt"
  "$REPLAY" --hp carry=0 "${dumps[@]}" >"$out/$pair.replay_cold.txt"
  cat "$out/$pair.replay.txt" >&2
done

# Online paired A/B, alternating order.
ab=(q9:prose_letter q9:code_inventory q9:code_edit_typehints q8:mixed_tcp q8:code_inventory q8:prose_snowstorm)
i=0
for pn in "${ab[@]}"; do
  pair=${pn%%:*}; n=${pn#*:}
  if (( i % 2 == 0 )); then bench "$pair" "$n.A" "$P/$n.txt" off; bench "$pair" "$n.B" "$P/$n.txt" on
  else bench "$pair" "$n.B" "$P/$n.txt" on; bench "$pair" "$n.A" "$P/$n.txt" off; fi
  i=$((i + 1))
done

python3 - "$out" "${ab[@]}" <<'EOF'
import json, math, os, re, sys
d, ab = sys.argv[1], sys.argv[2:]
g = lambda xs: math.exp(sum(map(math.log, xs)) / len(xs))
geo = lambda t: float(re.search(r"GEOMEAN picked/argmax tau over \d+ sessions: ([0-9.]+)", t).group(1))
pairs = {}
for pair in ("q9", "q8"):
    rep = open(os.path.join(d, pair + ".replay.txt")).read()
    pairs[pair] = geo(rep)
    print("METRIC %s_replay=%.4f" % (pair, pairs[pair]))
    print("METRIC %s_replay_cold=%.4f" % (pair, geo(open(os.path.join(d, pair + ".replay_cold.txt")).read())))
    for name, base, pick in re.findall(r"/([a-z_]+)-[0-9a-f]+\.bin: cycles=\d+ argmax_tau=([0-9.]+) picked_tau=([0-9.]+)", rep):
        print("METRIC %s_replay_%s=%.4f" % (pair, name, float(pick) / float(base)))
print("METRIC replay_tau_ratio=%.4f" % g(list(pairs.values())))
def run(tag):
    j = json.load(open(os.path.join(d, tag + ".json")))
    toks = re.findall(r"decode \((\d+) tok", open(os.path.join(d, tag + ".err")).read())[-1]
    return j["spec_tau"][-1], j["decode_tok_s"]["median"], toks
tr, sr = {"q9": [], "q8": []}, {"q9": [], "q8": []}
for pn in ab:
    pair, n = pn.split(":")
    a, b = run(f"{pair}.{n}.A"), run(f"{pair}.{n}.B")
    tr[pair].append(b[0] / a[0]); sr[pair].append(b[1] / a[1])
    print("METRIC %s_%s_tok_s_A=%.2f" % (pair, n, a[1])); print("METRIC %s_%s_tok_s_B=%.2f" % (pair, n, b[1]))
    print("ASI %s_%s_tau_A_B=%.2f/%.2f tokens=%s/%s" % (pair, n, a[0], b[0], a[2], b[2]))
for pair in ("q9", "q8"):
    print("METRIC %s_online_tau_ratio=%.4f" % (pair, g(tr[pair])))
    print("METRIC %s_online_tok_s_ratio=%.4f" % (pair, g(sr[pair])))
print("METRIC online_tok_s_ratio=%.4f" % g(sr["q9"] + sr["q8"]))
EOF
rm -rf "$out"
