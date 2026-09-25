#!/usr/bin/env bash
# Prompt-processing (prefill) throughput for Qwen3.8-Flash-Next (qwen4, mq6q8-pleq8)
# on gfx1151, through the native daemon protocol (the product path). `hipfire bench`
# cannot load this model (max_seq pinned to 2048), so we drive the daemon directly.
#
# Workload: fresh daemon, load (max_seq 2048, kv q8, mtp off, graph off), 1 warmup
# generate + RUNS measured generates of a committed ~1k-token prompt, greedy,
# max_tokens 16. Prefill tok/s = prefill_tokens / prefill_ms from the daemon.
# Guard: greedy token IDs must match the recorded baseline (REF_IDS).
set -euo pipefail
cd "$(dirname "$0")"

flock -w 3600 /tmp/hipfire-build.lock cargo build --release -p hipfire-daemon -q >/tmp/autoresearch-build.log 2>&1 \
    || { tail -40 /tmp/autoresearch-build.log; exit 1; }

exec flock -w 3600 /tmp/hipfire-gpu.lock python3 - <<'PY'
import hashlib, json, os, select, statistics, subprocess, sys, time

MODEL = os.path.expanduser("~/.hipfire/models/qwen3.8-flash-next.mq6q8-pleq8.hfq")
PROMPT_PATH = "benchmarks/prompts/glimmer_prefill_1024.txt"
PROMPT_MD5 = "0ee8f86ada3683eda452bc294ec824a9"
RUNS = int(os.environ.get("AR_RUNS", "3"))
MAX_TOKENS = 16
# Greedy token IDs recorded on the baseline build (text: "The text you provided contains a
# repeated block of prose and Python code. Here is").
REF_IDS = [760, 1414, 488, 3766, 5435, 264, 11173, 2424, 314, 1414, 321, 12654, 1970, 13, 5514, 369]  # segment 2: F16 WMMA gate/up + down, KLD-gated
MIN_MATCH = 12  # leading tokens that must match REF_IDS

prompt = open(PROMPT_PATH, "rb").read()
assert hashlib.md5(prompt).hexdigest() == PROMPT_MD5, "prompt bytes changed"
prompt = prompt.decode()

env = dict(os.environ, HIPFIRE_EMIT_TOKEN_IDS="1", HIPFIRE_GRAPH="0",
           HIPFIRE_AR_GRAPH="0", HIPFIRE_CASK_OFF="1", HIPFIRE_DPM_WARMUP_SECS="10")
log = open("/tmp/autoresearch-daemon.log", "w")
proc = subprocess.Popen(["target/release/daemon"], env=env, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                        start_new_session=True)

def send(msg):
    proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
    proc.stdin.flush()

STALE = set()

def read_until(stop, seconds):
    out, deadline = [], time.time() + seconds
    while True:
        left = deadline - time.time()
        if left <= 0:
            raise TimeoutError(f"no {stop} within {seconds}s")
        if not select.select([proc.stdout], [], [], min(left, 5.0))[0]:
            continue
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("daemon closed stdout (see /tmp/autoresearch-daemon.log)")
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t == "done" and ev.get("id") in STALE:
            continue  # previous generate's done: the daemon emits it only once the next command arrives
        out.append(ev)
        if t == "commit_ready":
            send({"type": "commit", "id": ev.get("id"), "attempt_id": ev.get("attempt_id", 1)})
            STALE.add(ev.get("id"))
        if t in stop:
            return out

def generate(gid):
    send({"type": "generate", "id": gid, "prompt": prompt, "temperature": 0.0,
          "max_tokens": MAX_TOKENS, "max_think_tokens": 1, "attempt_id": 1})
    evs = read_until({"commit_ready", "done", "error"}, 600)
    errs = [e for e in evs if e.get("type") == "error"]
    if errs:
        raise RuntimeError(f"generate error: {errs}")
    done = evs[-1]
    ids = [e.get("tok_id") for e in evs if e.get("type") == "committed"]
    text = "".join(e.get("text", "") for e in evs if e.get("type") == "token")
    return done, ids, text

try:
    t0 = time.time()
    send({"type": "load", "model": MODEL,
          "params": {"max_seq": 2048, "kv_mode": "q8", "mtp_mode": "off"}})
    loaded = read_until({"loaded", "load_error", "error"}, 1800)
    if loaded[-1].get("type") != "loaded":
        raise RuntimeError(f"load failed: {loaded[-1]}")
    load_s = time.time() - t0
    generate("warmup")
    rows = []
    for i in range(RUNS):
        done, ids, text = generate(f"r{i}")
        rows.append((done, ids, text))
finally:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    log.close()

pp = [d["prefill_tokens"] / d["prefill_ms"] * 1000.0 for d, _, _ in rows]
dec = [d.get("decode_tok_s") or 0.0 for d, _, _ in rows]
ttft = [d.get("ttft_ms") or d["prefill_ms"] for d, _, _ in rows]
ids = rows[-1][1]
print(f"prefill_tokens={rows[0][0]['prefill_tokens']} samples_pp={[round(x, 2) for x in pp]}")
print(f"text={rows[-1][2]!r}")
print(f"ids={ids}")
if any(r[1] != ids for r in rows):
    print("FAIL: token IDs differ between runs"); sys.exit(1)
match = MAX_TOKENS
if REF_IDS is not None:
    match = next((i for i, (a, b) in enumerate(zip(ids, REF_IDS)) if a != b), min(len(ids), len(REF_IDS)))
    if match < MIN_MATCH:
        print(f"FAIL: only {match} leading tokens match baseline"); sys.exit(1)
print(f"METRIC prefill_tok_s={statistics.median(pp):.2f}")
print(f"METRIC ttft_ms={statistics.median(ttft):.1f}")
print(f"METRIC decode_tok_s={statistics.median(dec):.2f}")
print(f"METRIC token_match={match}")
print(f"METRIC load_s={load_s:.1f}")
PY
