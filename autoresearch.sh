#!/usr/bin/env bash
# Long-context prefill + MTP decode throughput for Qwen3.8-Flash-Next (qwen4,
# the qwen3.8-flash-next.mq4 pin) on gfx1151 through the native daemon protocol,
# gated on the prefill-route KLD of the first 2048 tokens against the
# BF16-source teacher.
#
# Perf: fresh daemon, load (max_seq 65536, mtp on, graph off, prompt cache off),
# 1 warmup + RUNS measured greedy generates of a committed ~32k-token prompt,
# max_tokens MAX_TOKENS. Metric = geomean(median prefill tok/s, median decode tok/s).
# Guards (non-zero exit on violation):
#   - every run prefills the whole prompt; greedy ids identical across runs and
#     the leading MIN_MATCH match REF_IDS (garbage guard; KLD decides quality);
#   - prefill-route KLD (qwen4_kld eval: production forward_chunk, all-row
#     logits, KLD_CHUNKS x 512 = 2048 tokens of wikitext-2) must not exceed KLD_MAX.
set -euo pipefail
cd "$(dirname "$0")"

# Two invocations: `--example` restricts target selection, so a combined
# `-p hipfire-daemon --example qwen4_kld` never relinks the daemon binary.
{ flock -w 3600 /tmp/hipfire-build.lock cargo build --release -q -p hipfire-daemon \
    && flock -w 3600 /tmp/hipfire-build.lock cargo build --release -q -p hipfire-arch-qwen4 \
        --features lab --example qwen4_kld; } >/tmp/autoresearch-build.log 2>&1 \
    || { tail -40 /tmp/autoresearch-build.log; exit 1; }

exec flock -w 3600 /tmp/hipfire-gpu.lock python3 - <<'PY'
import hashlib, json, os, re, select, statistics, subprocess, sys, time

MODEL = os.path.expanduser("~/.hipfire/models/qwen3.8-flash-next.mq4.hfq")
KLD_REF = "/home/bjoern/hipfire-qwen4-kld/.codeinsight+research/qwen4-kld/source-teacher/bf16src-wt2-c512x32.kldref"
KLD_CHUNKS = 4  # first 2048 tokens
# Segment baseline prefill-route KLD; a candidate above it fails (fixed, per the goal).
KLD_MAX = 0.083874  # segment-2 baseline, 4 chunks, prefill logits sha256 7670240f...
PROMPT_PATH = "benchmarks/prompts/qwen4_longcode_32k.txt"
PROMPT_MD5 = "70ffb7d29325a2e2fa032b0911592c6b"
MIN_PREFILL = 31900
MAX_SEQ = 65536
RUNS = int(os.environ.get("AR_RUNS", "3"))
MAX_TOKENS = 256
# Greedy ids at baseline (text: "Based on the provided source code and design notes, here is a summary of the **Helix**").
REF_IDS = [27775, 383, 279, 3766, 2450, 1970, 321, 2790, 8129, 11, 1532, 369, 264, 11782, 314, 279]
MIN_MATCH = 16  # leading tokens that must match REF_IDS (garbage guard)

ENV = dict(os.environ, HIPFIRE_EMIT_TOKEN_IDS="1", HIPFIRE_GRAPH="0",
           HIPFIRE_AR_GRAPH="0", HIPFIRE_CASK_OFF="1", HIPFIRE_DPM_WARMUP_SECS="10",
           HIPFIRE_QWEN_PROMPT_CACHE="0")

prompt = open(PROMPT_PATH, "rb").read()
assert hashlib.md5(prompt).hexdigest() == PROMPT_MD5, "prompt bytes changed"
prompt = prompt.decode()

# ---- perf through the daemon ----
log = open("/tmp/autoresearch-daemon.log", "w")
proc = subprocess.Popen(["target/release/daemon"], env=ENV, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                        start_new_session=True)
OUT_FD = proc.stdout.fileno()
PENDING = b""
STALE = set()

def send(msg):
    proc.stdin.write(json.dumps(msg, separators=(",", ":")) + "\n")
    proc.stdin.flush()

def read_line(deadline):
    # Own line buffer over the raw fd: select() on a buffered file object misses
    # lines already pulled into Python's buffer.
    global PENDING
    while b"\n" not in PENDING:
        left = deadline - time.time()
        if left <= 0:
            raise TimeoutError("daemon output timeout")
        if not select.select([OUT_FD], [], [], min(left, 5.0))[0]:
            continue
        chunk = os.read(OUT_FD, 1 << 16)
        if not chunk:
            raise RuntimeError("daemon closed stdout (see /tmp/autoresearch-daemon.log)")
        PENDING += chunk
    line, PENDING = PENDING.split(b"\n", 1)
    return line.decode(errors="replace")

def read_until(stop, seconds):
    out, deadline = [], time.time() + seconds
    while True:
        line = read_line(deadline)
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t == "done" and ev.get("id") in STALE:
            continue  # previous generate's done: emitted only once the next command arrives
        out.append(ev)
        if t == "commit_ready":
            send({"type": "commit", "id": ev.get("id"), "attempt_id": ev.get("attempt_id", 1)})
            STALE.add(ev.get("id"))
        if t in stop:
            return out

def generate(gid):
    send({"type": "generate", "id": gid, "prompt": prompt, "temperature": 0.0,
          "max_tokens": MAX_TOKENS, "max_think_tokens": 1, "attempt_id": 1})
    evs = read_until({"commit_ready", "done", "error"}, 1800)
    errs = [e for e in evs if e.get("type") == "error"]
    if errs:
        raise RuntimeError(f"generate error: {errs}")
    ids = [e.get("tok_id") for e in evs if e.get("type") == "committed"]
    text = "".join(e.get("text", "") for e in evs if e.get("type") == "token")
    return evs[-1], ids, text

try:
    t0 = time.time()
    send({"type": "load", "model": MODEL,
          "params": {"max_seq": MAX_SEQ, "mtp_mode": "on"}})
    loaded = read_until({"loaded", "load_error", "error"}, 1800)
    if loaded[-1].get("type") != "loaded":
        raise RuntimeError(f"load failed: {loaded[-1]}")
    load_s = time.time() - t0
    generate("warmup")
    rows = [generate(f"r{i}") for i in range(RUNS)]
finally:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    log.close()

pp = [d["prefill_tokens"] / d["prefill_ms"] * 1000.0 for d, _, _ in rows]
ttft = [d.get("ttft_ms") or d["prefill_ms"] for d, _, _ in rows]
dec = [d.get("decode_tok_s") or 0.0 for d, _, _ in rows]
ids = rows[-1][1]
print(f"done={ {k: v for k, v in rows[-1][0].items() if not isinstance(v, (list, dict))} }")
print(f"prefill_tokens={[d['prefill_tokens'] for d, _, _ in rows]} samples_pp={[round(x, 2) for x in pp]}")
print(f"text={rows[-1][2]!r}")
print(f"ids={ids}")
if any(d["prefill_tokens"] < MIN_PREFILL for d, _, _ in rows):
    print("FAIL: a run did not prefill the whole prompt (prompt cache hit?)"); sys.exit(1)
if len(ids) < MIN_MATCH:
    print(f"FAIL: only {len(ids)} tokens generated"); sys.exit(1)
if any(r[1] != ids for r in rows):
    print("FAIL: token IDs differ between runs"); sys.exit(1)
match = len(ids)
if REF_IDS is not None:
    match = next((i for i, (a, b) in enumerate(zip(ids, REF_IDS)) if a != b), min(len(ids), len(REF_IDS)))
    if match < MIN_MATCH:
        print(f"FAIL: only {match} leading tokens match baseline"); sys.exit(1)

# ---- prefill-route KLD gate ----
kld_log = "/tmp/autoresearch-kld.log"
with open(kld_log, "w") as f:
    rc = subprocess.run(["target/release/examples/qwen4_kld", "eval", "--model", MODEL,
                         "--ref", KLD_REF, "--output", "/tmp/autoresearch-kld.kldseq",
                         "--max-chunks", str(KLD_CHUNKS)],
                        env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode
tail = open(kld_log).read()
m = re.search(r"mean KLD = ([0-9.]+)\s+mean NLL = ([0-9.]+).*top1 = ([0-9.]+).*logits sha256 ([0-9a-f]+)", tail)
if rc != 0 or not m:
    print(tail[-3000:]); print("FAIL: qwen4_kld eval"); sys.exit(1)
kld, nll, top1, sha = float(m.group(1)), float(m.group(2)), float(m.group(3)), m.group(4)
print(f"prefill_logits_sha256={sha}")
print(f"METRIC score={(statistics.median(pp) * statistics.median(dec)) ** 0.5:.3f}")
print(f"METRIC prefill_tok_s={statistics.median(pp):.2f}")
print(f"METRIC ttft_ms={statistics.median(ttft):.1f}")
print(f"METRIC prefill_kld={kld:.6f}")
print(f"METRIC prefill_nll={nll:.6f}")
print(f"METRIC prefill_top1={top1:.4f}")
print(f"METRIC decode_tok_s={statistics.median(dec):.2f}")
print(f"samples_dec={[round(x, 2) for x in dec]}")
print(f"METRIC token_match={match}")
print(f"METRIC load_s={load_s:.1f}")
if KLD_MAX is not None and kld > KLD_MAX:
    print(f"FAIL: prefill KLD {kld:.6f} > baseline {KLD_MAX:.6f}"); sys.exit(1)
PY
