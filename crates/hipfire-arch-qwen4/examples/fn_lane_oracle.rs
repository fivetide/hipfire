// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Flash-Next exact AR lane oracle (fn-batch PLAN §2, §6 S2 G0).
//!
//! fn_lane_oracle MODEL.mq4 TEXT.txt OUT_DIR [--ks 1,2,4,8] [--contexts 512,2048,8192]
//!     [--steps 64] [--max-seq 16384] [--policy production|per-lane|<spec>]
//!     [--g0] [--controls] [--bench N]
//!
//! The model loads once. Request `i` gets a unique prompt of exactly
//! `contexts[i % len]` tokens (TEXT tokens from offset `i * 997`, wrapping).
//!
//! Singleton reference (resident state, `prefill_final` +
//! `forward_token_or_argmax`, retained replay included): per request the
//! prefill logits row (`logits_0.bin`), the logits row after each of `steps`
//! greedy tokens (`logits_{s}.bin`), the fed token ids, and the full state
//! bytes at the checkpoints (after prefill, after step 1, after the last step)
//! are written under `OUT/ref/req{i}/`.
//!
//! Lane phase: the same requests run as lanes (`admit_lane`, hand-built
//! `BatchStepPlan`s through `lane_executor`): one prefill tile per step for
//! the first unfinished lane plus every decoding lane's AR row, so lanes
//! finish prefill on different steps and prefill tiles share steps with AR
//! rows. A lane may run at most three AR steps ahead before every lane has
//! finished prefill; after that all lanes decode together. Every AR row's
//! logits row, every committed id and every checkpoint's state bytes are
//! compared with the singleton reference (first differing byte reported).
//!
//! `--g0`: per-stage probe (each non-stateful stage alone at once-over-all-rows,
//! k=2 and k=8, `min(steps, 16)` steps) and the combined policy at every k.
//! `--controls`: negative controls that MUST be detected (lane swap, wrong
//! position). `--bench N`: lane step vs singleton step wall time.
//!
//! Exit 0 only when every compared run passes and every control is detected.
//! Nothing is written outside OUT_DIR.

use hipfire_arch_qwen4::admit_hfqm_artifact;
use hipfire_arch_qwen4::bundle::Qwen4Bundle;
use hipfire_arch_qwen4::gpu_forward_lanes::{LaneStagePolicy, StageName, StageShare};
use hipfire_arch_qwen4::state::Qwen4State;
use hipfire_runtime::device_mesh::DeviceMesh;
use hipfire_runtime::hfq::{HfqFile, HfqModelSource};
use hipfire_runtime::model_source::SourcePayload;
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestAdvance, RequestEpoch, RequestRows, RequestStepKind, RowRange, SlotBatch,
};
use hipfire_runtime::tokenizer::Tokenizer;
use hipfire_runtime::weight_store::{fulfill_manifest_from_payloads, WeightOrigin};
use rdna_compute::slot_pool::SlotId;
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

type R<T> = Result<T, String>;

fn e<E: std::fmt::Display>(x: E) -> String {
    x.to_string()
}

/// A lane may run this many AR steps before every lane has finished prefill.
const EARLY_AR_STEPS: usize = 3;

// ── arguments ────────────────────────────────────────────────────────

struct Args {
    model: PathBuf,
    text: PathBuf,
    out: PathBuf,
    ks: Vec<usize>,
    contexts: Vec<usize>,
    steps: usize,
    max_seq: usize,
    policy: String,
    g0: bool,
    controls: bool,
    bench: usize,
}

fn parse_list(s: &str, what: &str) -> R<Vec<usize>> {
    s.split(',')
        .filter(|v| !v.trim().is_empty())
        .map(|v| v.trim().parse::<usize>().map_err(|x| format!("bad {what} '{v}': {x}")))
        .collect()
}

fn parse_args() -> R<Args> {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: fn_lane_oracle MODEL.mq4 TEXT.txt OUT_DIR [--ks 1,2,4,8] [--contexts 512,2048,8192] \
                 [--steps 64] [--max-seq 16384] [--policy production|per-lane|<spec>] [--g0] [--controls] [--bench N]";
    let mut pos = Vec::new();
    let mut a = Args {
        model: PathBuf::new(),
        text: PathBuf::new(),
        out: PathBuf::new(),
        ks: vec![1, 2, 4, 8],
        contexts: vec![512, 2048, 8192],
        steps: 64,
        max_seq: 16384,
        policy: "production".into(),
        g0: false,
        controls: false,
        bench: 0,
    };
    let mut i = 0;
    let value = |i: &mut usize, flag: &str| -> R<String> {
        *i += 1;
        raw.get(*i).cloned().ok_or_else(|| format!("{flag} needs a value\n{usage}"))
    };
    while i < raw.len() {
        match raw[i].as_str() {
            "--ks" => a.ks = parse_list(&value(&mut i, "--ks")?, "--ks")?,
            "--contexts" => a.contexts = parse_list(&value(&mut i, "--contexts")?, "--contexts")?,
            "--steps" => a.steps = value(&mut i, "--steps")?.parse().map_err(e)?,
            "--max-seq" => a.max_seq = value(&mut i, "--max-seq")?.parse().map_err(e)?,
            "--policy" => a.policy = value(&mut i, "--policy")?,
            "--bench" => a.bench = value(&mut i, "--bench")?.parse().map_err(e)?,
            "--g0" => a.g0 = true,
            "--controls" => a.controls = true,
            flag if flag.starts_with("--") => return Err(format!("unknown flag {flag}\n{usage}")),
            other => pos.push(other.to_string()),
        }
        i += 1;
    }
    if pos.len() != 3 {
        return Err(usage.to_string());
    }
    a.model = pos[0].clone().into();
    a.text = pos[1].clone().into();
    a.out = pos[2].clone().into();
    if a.ks.is_empty() || a.ks.contains(&0) {
        return Err("--ks must be non-empty and positive".into());
    }
    if a.contexts.is_empty() || a.contexts.contains(&0) {
        return Err("--contexts must be non-empty and positive".into());
    }
    if a.steps == 0 {
        return Err("--steps must be >= 1".into());
    }
    Ok(a)
}

// ── log / summary ────────────────────────────────────────────────────

#[derive(Default)]
struct Log {
    lines: Vec<String>,
}

impl Log {
    fn say(&mut self, line: String) {
        println!("{line}");
        self.lines.push(line);
    }
}

// ── byte comparison ──────────────────────────────────────────────────

#[derive(Default)]
struct Diff {
    items: usize,
    bytes: u64,
    bad: usize,
    first: Option<String>,
    shown: Vec<String>,
}

impl Diff {
    fn note(&mut self, msg: String) {
        self.bad += 1;
        if self.first.is_none() {
            self.first = Some(msg.clone());
        }
        if self.shown.len() < 8 {
            self.shown.push(msg);
        }
    }

    /// Compare `reference` with `got`; `f32_view` adds the float at the first
    /// differing byte to the report.
    fn check(&mut self, label: &str, reference: &[u8], got: &[u8], f32_view: bool) {
        self.items += 1;
        self.bytes += reference.len() as u64;
        if reference.len() != got.len() {
            self.note(format!("{label}: length ref {} got {}", reference.len(), got.len()));
            return;
        }
        if reference == got {
            return;
        }
        let i = reference.iter().zip(got).position(|(a, b)| a != b).unwrap_or(0);
        let n = reference.iter().zip(got).filter(|(a, b)| a != b).count();
        let mut msg = format!(
            "{label}: first differing byte {i} (ref 0x{:02x} got 0x{:02x}), {n}/{} bytes differ",
            reference[i],
            got[i],
            reference.len()
        );
        if f32_view {
            let j = i / 4 * 4;
            if j + 4 <= reference.len() {
                let r = f32::from_le_bytes(reference[j..j + 4].try_into().unwrap());
                let g = f32::from_le_bytes(got[j..j + 4].try_into().unwrap());
                msg += &format!("; f32[{}] ref {r:e} got {g:e}", j / 4);
            }
        }
        self.note(msg);
    }

    fn ids(&mut self, label: &str, reference: u32, got: u32) {
        self.items += 1;
        self.bytes += 4;
        if reference != got {
            self.note(format!("{label}: token id ref {reference} got {got}"));
        }
    }
}

// ── device reads / state capture ─────────────────────────────────────

fn read_prefix(gpu: &Gpu, t: &GpuTensor, bytes: usize) -> R<Vec<u8>> {
    if bytes == 0 {
        return Ok(Vec::new());
    }
    if bytes > t.buf.size() {
        return Err(format!("read of {bytes} bytes exceeds the {}-byte device buffer", t.buf.size()));
    }
    let mut out = vec![0u8; bytes];
    gpu.hip.memcpy_dtoh(&mut out, &t.buf).map_err(e)?;
    Ok(out)
}

/// First `rows` rows of an arena holding `capacity` rows.
fn read_rows(gpu: &Gpu, t: &GpuTensor, rows: usize, capacity: usize) -> R<Vec<u8>> {
    if rows == 0 || capacity == 0 {
        return Ok(Vec::new());
    }
    let total = t.byte_size();
    if total % capacity != 0 {
        return Err(format!("arena of {total} bytes does not hold {capacity} equal rows"));
    }
    read_prefix(gpu, t, rows.min(capacity) * (total / capacity))
}

/// `elems` elements of `t` starting at element `offset`.
fn read_view(gpu: &Gpu, t: &GpuTensor, offset: usize, elems: usize) -> R<Vec<u8>> {
    let v = t.sub_offset(offset, elems);
    read_prefix(gpu, &v, v.byte_size())
}

struct Item {
    name: String,
    bytes: Vec<u8>,
}

fn le_u64s(v: &[usize]) -> Vec<u8> {
    v.iter().flat_map(|x| (*x as u64).to_le_bytes()).collect()
}

/// Every byte family of a request state a step may write: GDN recurrent
/// (live slot) and conv per layer; QSA full K/V, raw/pooled index rows below
/// their marks, partial rows, selected indices and the marks per layer; PLE
/// conv and history; hyper feedback; position. `[n]` is the ordinal within
/// the family's layer list.
fn capture(gpu: &Gpu, st: &Qwen4State) -> R<Vec<Item>> {
    let mut items = Vec::new();
    let mut push = |name: String, bytes: Vec<u8>| items.push(Item { name, bytes });
    for (i, g) in st.gdn.iter().enumerate() {
        push(format!("gdn_recurrent[{i}]"), read_prefix(gpu, &g.recurrent, g.recurrent.byte_size())?);
        push(format!("gdn_conv[{i}]"), read_prefix(gpu, &g.conv, g.conv.byte_size())?);
    }
    for (i, q) in st.qsa.iter().enumerate() {
        push(format!("qsa_full_keys[{i}]"), read_rows(gpu, &q.full_keys, q.full_len, q.full_capacity)?);
        push(format!("qsa_full_values[{i}]"), read_rows(gpu, &q.full_values, q.full_len, q.full_capacity)?);
        push(
            format!("qsa_raw_index_keys[{i}]"),
            read_rows(gpu, &q.raw_index_keys, q.raw_len, q.raw_capacity)?,
        );
        push(
            format!("qsa_pooled_keys[{i}]"),
            read_rows(gpu, &q.pooled_keys, q.pooled_len, q.pooled_capacity)?,
        );
        push(
            format!("qsa_partial_keys[{i}]"),
            read_rows(gpu, &q.partial_keys, q.partial_len, q.partial_capacity)?,
        );
        push(
            format!("qsa_partial_values[{i}]"),
            read_rows(gpu, &q.partial_values, q.partial_len, q.partial_capacity)?,
        );
        push(
            format!("qsa_selected_indices[{i}]"),
            read_prefix(gpu, &q.selected_indices, (q.selected_len * 4).min(q.selected_indices.byte_size()))?,
        );
        push(
            format!("qsa_marks[{i}]"),
            le_u64s(&[q.full_len, q.raw_len, q.pooled_len, q.partial_len, q.selected_len, q.position]),
        );
    }
    push("ple_conv".into(), read_prefix(gpu, &st.ple_conv, st.ple_conv.byte_size())?);
    push("hyper_feedback".into(), read_prefix(gpu, &st.hyper_feedback, st.hyper_feedback.byte_size())?);
    let prev = st.ple_history.previous();
    let mut hist = Vec::with_capacity(12);
    for v in [prev[0], prev[1], st.ple_history.eos_token_id()] {
        hist.extend_from_slice(&v.to_le_bytes());
    }
    push("ple_history".into(), hist);
    push("position".into(), le_u64s(&[st.position]));
    Ok(items)
}

// ── reference store ──────────────────────────────────────────────────

struct RefCkpt {
    step: usize,
    path: PathBuf,
    /// (name, offset, len) of every item in the `.bin`.
    index: Vec<(String, u64, u64)>,
}

struct RefReq {
    prompt: Vec<u32>,
    /// Token fed at step `s` (1-based): `ids[s - 1]`.
    ids: Vec<u32>,
    dir: PathBuf,
    ckpts: Vec<RefCkpt>,
}

fn ckpt_tag(step: usize) -> String {
    if step == 0 {
        "prefill".into()
    } else {
        format!("step{step}")
    }
}

fn write_ckpt(dir: &Path, step: usize, items: &[Item]) -> R<RefCkpt> {
    let tag = ckpt_tag(step);
    let path = dir.join(format!("state_{tag}.bin"));
    let mut w = BufWriter::new(File::create(&path).map_err(e)?);
    let mut index = Vec::with_capacity(items.len());
    let mut offset = 0u64;
    for it in items {
        w.write_all(&it.bytes).map_err(e)?;
        index.push((it.name.clone(), offset, it.bytes.len() as u64));
        offset += it.bytes.len() as u64;
    }
    w.flush().map_err(e)?;
    let idx: Vec<Value> = index
        .iter()
        .map(|(n, o, l)| json!({"name": n, "offset": o, "len": l}))
        .collect();
    fs::write(dir.join(format!("state_{tag}.json")), serde_json::to_vec_pretty(&idx).map_err(e)?)
        .map_err(e)?;
    Ok(RefCkpt { step, path, index })
}

fn compare_ckpt(diff: &mut Diff, ctx: &str, reference: &RefCkpt, got: &[Item]) -> R<()> {
    let mut f = File::open(&reference.path).map_err(e)?;
    if reference.index.len() != got.len() {
        diff.note(format!("{ctx}: state item count ref {} got {}", reference.index.len(), got.len()));
    }
    for (idx, (name, offset, len)) in reference.index.iter().enumerate() {
        let Some(g) = got.get(idx) else { break };
        if &g.name != name {
            diff.note(format!("{ctx}: state item {idx} is '{}' but the reference has '{name}'", g.name));
            continue;
        }
        let mut buf = vec![0u8; *len as usize];
        f.seek(SeekFrom::Start(*offset)).map_err(e)?;
        f.read_exact(&mut buf).map_err(e)?;
        diff.check(&format!("{ctx} state {name}"), &buf, &g.bytes, false);
    }
    Ok(())
}

// ── world: model, singleton reference, lane plumbing ────────────────

struct World {
    gpu: Gpu,
    bundle: Qwen4Bundle,
    vocab: usize,
    /// Singleton logits row.
    row: GpuTensor,
    max_lanes: usize,
}

fn read_row(w: &World) -> R<Vec<u8>> {
    w.gpu.hip.device_synchronize().map_err(e)?;
    read_prefix(&w.gpu, &w.row, w.vocab * 4)
}

fn record_ref(w: &mut World, i: usize, prompt: Vec<u32>, steps: usize, ckpt_steps: &[usize], dir: &Path) -> R<RefReq> {
    fs::create_dir_all(dir).map_err(e)?;
    let mut rq = RefReq { prompt, ids: Vec::with_capacity(steps), dir: dir.to_path_buf(), ckpts: Vec::new() };
    w.bundle.reset(&mut w.gpu).map_err(e)?;
    let prompt = rq.prompt.clone();
    w.bundle.prefill_final(&mut w.gpu, &prompt, 0, &w.row).map_err(e)?;
    fs::write(dir.join("logits_0.bin"), read_row(w)?).map_err(e)?;
    let items = capture(&w.gpu, &w.bundle.state)?;
    rq.ckpts.push(write_ckpt(dir, 0, &items)?);
    for s in 1..=steps {
        let id = w
            .bundle
            .forward_token_or_argmax(&mut w.gpu, None, &w.row)
            .map_err(|x| format!("request {i} singleton step {s}: {x}"))?;
        rq.ids.push(id);
        fs::write(dir.join(format!("logits_{s}.bin")), read_row(w)?).map_err(e)?;
        if ckpt_steps.contains(&s) {
            let items = capture(&w.gpu, &w.bundle.state)?;
            rq.ckpts.push(write_ckpt(dir, s, &items)?);
        }
    }
    fs::write(
        dir.join("ids.txt"),
        rq.ids.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("\n"),
    )
    .map_err(e)?;
    Ok(rq)
}

// ── plans ────────────────────────────────────────────────────────────

struct RowSpec {
    slot: usize,
    epoch: RequestEpoch,
    tokens: Vec<u32>,
    start: usize,
    kind: RequestStepKind,
}

/// One step's plan: per-slot rows for every slot `0..max_lanes` (idle slots
/// empty), requests in slot order, absolute positions.
fn make_plan(max_lanes: usize, mut rows: Vec<RowSpec>) -> BatchStepPlan {
    rows.sort_by_key(|r| r.slot);
    let mut per: Vec<(SlotId, Vec<u32>, usize)> = (0..max_lanes).map(|s| (SlotId(s), Vec::new(), 0)).collect();
    for r in &rows {
        per[r.slot] = (SlotId(r.slot), r.tokens.clone(), r.start);
    }
    let triples: Vec<(SlotId, &[u32], usize)> = per.iter().map(|(s, t, p)| (*s, t.as_slice(), *p)).collect();
    let batch = SlotBatch::build(&triples);
    let mut plan = BatchStepPlan { batch, ..Default::default() };
    let mut begin = 0;
    for r in &rows {
        plan.requests.push(RequestRows { epoch: r.epoch, rows: RowRange { begin, len: r.tokens.len() }, kind: r.kind });
        match r.kind {
            RequestStepKind::Ar => plan.decode_rows += r.tokens.len(),
            RequestStepKind::Prefill => plan.prefill_rows += r.tokens.len(),
            RequestStepKind::Verify { .. } => plan.verify_rows += r.tokens.len(),
            RequestStepKind::Forced => plan.forced_rows += r.tokens.len(),
        }
        begin += r.tokens.len();
    }
    plan
}

struct StepErr {
    stage: &'static str,
    msg: String,
}

impl StepErr {
    fn text(&self) -> String {
        format!("{} failed: {}", self.stage, self.msg)
    }
}

struct Stepped {
    /// Logits row of every AR request and every prompt-completing prefill tile.
    logits: Vec<(RequestEpoch, Vec<u8>)>,
    advances: Vec<RequestAdvance>,
}

fn read_step_logits(
    gpu: &Gpu,
    bundle: &Qwen4Bundle,
    plan: &BatchStepPlan,
    picks: &[u32],
    vocab: usize,
    max_lanes: usize,
) -> R<Vec<(RequestEpoch, Vec<u8>)>> {
    gpu.hip.device_synchronize().map_err(e)?;
    let store = bundle.lanes().ok_or("lane store vanished")?;
    let t = store.logits();
    let mut out = Vec::new();
    let (mut ar, mut pre) = (0usize, 0usize);
    for rr in &plan.requests {
        match rr.kind {
            RequestStepKind::Ar => {
                out.push((rr.epoch, read_view(gpu, t, ar * vocab, vocab)?));
                ar += 1;
            }
            RequestStepKind::Prefill => {
                if picks.get(rr.rows.end() - 1).is_some_and(|&p| p != u32::MAX) {
                    out.push((rr.epoch, read_view(gpu, t, (max_lanes + pre) * vocab, vocab)?));
                }
                pre += 1;
            }
            _ => {}
        }
    }
    Ok(out)
}

/// provision + forward + (logits read) + commit of one plan.
fn exec_step(w: &mut World, plan: &BatchStepPlan, read_logits: bool) -> Result<Stepped, StepErr> {
    let World { gpu, bundle, vocab, max_lanes, .. } = w;
    let se = |stage: &'static str, msg: String| StepErr { stage, msg };
    let mut ex = bundle.lane_executor().map_err(|x| se("provision", x.to_string()))?;
    ex.provision_step(gpu, plan).map_err(|m| se("provision", m))?;
    let out = match ex.forward_step(gpu, plan) {
        Ok(o) => o,
        Err(m) => {
            ex.abort_step(plan);
            return Err(se("forward", m));
        }
    };
    drop(ex);
    let logits = if read_logits {
        match read_step_logits(gpu, bundle, plan, &out.target_picks, *vocab, *max_lanes) {
            Ok(l) => l,
            Err(m) => {
                if let Ok(mut ex) = bundle.lane_executor() {
                    ex.abort_step(plan);
                }
                return Err(se("logits read", m));
            }
        }
    } else {
        Vec::new()
    };
    let mut ex = bundle.lane_executor().map_err(|x| se("commit", x.to_string()))?;
    match ex.commit_step(gpu, plan, out) {
        Ok(advances) => Ok(Stepped { logits, advances }),
        Err(m) => {
            ex.abort_step(plan);
            Err(se("commit", m))
        }
    }
}

// ── lane runs ────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Control {
    None,
    /// Two AR rows carry each other's tokens.
    SwapTokens,
    /// Two AR rows carry each other's row_slot.
    SwapRowSlot,
    /// Two AR rows swap token, position and row_slot (each request's rows
    /// hold the other lane's step).
    SwapRows,
    /// One lane runs an extra AR row before a step (position ahead).
    WrongPosition,
}

impl Control {
    fn name(self) -> &'static str {
        match self {
            Control::None => "none",
            Control::SwapTokens => "lane_swap_tokens",
            Control::SwapRowSlot => "lane_swap_row_slot",
            Control::SwapRows => "lane_swap_rows",
            Control::WrongPosition => "wrong_position",
        }
    }
}

struct Lane {
    req: usize,
    epoch: RequestEpoch,
    slot: usize,
    prompt_len: usize,
    prefilled: usize,
    committed: Vec<u32>,
    ar_steps: usize,
}

struct Run {
    lanes: Vec<Lane>,
    diff: Diff,
    /// Mismatches attributable to the prefill (logits row 0, first id, prefill state).
    pre_bad: usize,
    pre_checked: usize,
    lane_steps: usize,
    ckpts: usize,
    error: Option<String>,
    control: Control,
    fired: bool,
    refused: Option<String>,
    control_note: Option<String>,
    release_errors: Vec<String>,
}

struct RunOpts {
    prefill_ckpt: bool,
}

fn ar_spec(w: &World, l: &Lane) -> R<RowSpec> {
    let lane = w
        .bundle
        .lanes()
        .and_then(|s| s.lane(&l.epoch))
        .ok_or_else(|| format!("lane of request {} vanished", l.req))?;
    let seed = lane
        .host
        .pending_seed
        .ok_or_else(|| format!("lane of request {} has no pending seed", l.req))?;
    Ok(RowSpec {
        slot: l.slot,
        epoch: l.epoch,
        tokens: vec![seed],
        start: lane.state.position,
        kind: RequestStepKind::Ar,
    })
}

fn ref_logits(rq: &RefReq, s: usize) -> R<Vec<u8>> {
    fs::read(rq.dir.join(format!("logits_{s}.bin"))).map_err(e)
}

fn lane_items(w: &World, l: &Lane) -> R<Vec<Item>> {
    let lane = w
        .bundle
        .lanes()
        .and_then(|s| s.lane(&l.epoch))
        .ok_or_else(|| format!("lane of request {} vanished", l.req))?;
    capture(&w.gpu, &lane.state)
}

/// Compare one committed step with the references.
fn absorb(
    w: &World,
    refs: &[RefReq],
    run: &mut Run,
    plan: &BatchStepPlan,
    res: &Stepped,
    steps: usize,
    ckpt_steps: &[usize],
    opts: &RunOpts,
) -> R<()> {
    if res.advances.len() != plan.requests.len() {
        return Err(format!("{} advances for {} plan requests", res.advances.len(), plan.requests.len()));
    }
    for (rr, adv) in plan.requests.iter().zip(&res.advances) {
        let idx = run
            .lanes
            .iter()
            .position(|l| l.epoch == rr.epoch)
            .ok_or("plan epoch not in this run")?;
        if adv.epoch != rr.epoch {
            return Err("advance order differs from the plan".into());
        }
        let (req, prompt_len) = (run.lanes[idx].req, run.lanes[idx].prompt_len);
        let rq = &refs[req];
        let logits_of = |epoch: RequestEpoch| res.logits.iter().find(|(e2, _)| *e2 == epoch).map(|(_, v)| v);
        match rr.kind {
            RequestStepKind::Prefill => {
                run.lanes[idx].prefilled += rr.rows.len;
                if run.lanes[idx].prefilled < prompt_len {
                    if !adv.committed_ids.is_empty() {
                        run.diff.note(format!("req {req}: non-final prefill tile committed {:?}", adv.committed_ids));
                    }
                    continue;
                }
                let before = run.diff.bad;
                let got = logits_of(rr.epoch).ok_or("prompt-completing tile has no logits row")?;
                run.diff.check(&format!("req {req} prefill logits"), &ref_logits(rq, 0)?, got, true);
                match adv.committed_ids.as_slice() {
                    [id] => {
                        run.diff.ids(&format!("req {req} first token"), rq.ids[0], *id);
                        run.lanes[idx].committed.push(*id);
                    }
                    other => run.diff.note(format!("req {req}: prefill commit returned {other:?}, expected 1 id")),
                }
                if opts.prefill_ckpt {
                    let items = lane_items(w, &run.lanes[idx])?;
                    let rc = rq.ckpts.iter().find(|c| c.step == 0).ok_or("no prefill checkpoint")?;
                    compare_ckpt(&mut run.diff, &format!("req {req} prefill"), rc, &items)?;
                    run.ckpts += 1;
                }
                run.pre_checked += 1;
                run.pre_bad += run.diff.bad - before;
            }
            RequestStepKind::Ar => {
                run.lanes[idx].ar_steps += 1;
                let s = run.lanes[idx].ar_steps;
                let got = logits_of(rr.epoch).ok_or("AR row has no logits row")?;
                run.diff.check(&format!("req {req} step {s} logits"), &ref_logits(rq, s)?, got, true);
                run.lane_steps += 1;
                match adv.committed_ids.as_slice() {
                    [id] => {
                        // Commit of step `s` is the token fed at step `s + 1`.
                        if s < steps {
                            run.diff.ids(&format!("req {req} step {s} pick"), rq.ids[s], *id);
                        }
                        run.lanes[idx].committed.push(*id);
                    }
                    other => run.diff.note(format!("req {req} step {s}: AR commit returned {other:?}, expected 1 id")),
                }
                if adv.committed_position != prompt_len + s {
                    run.diff.note(format!(
                        "req {req} step {s}: committed position {} expected {}",
                        adv.committed_position,
                        prompt_len + s
                    ));
                }
                if ckpt_steps.contains(&s) {
                    let items = lane_items(w, &run.lanes[idx])?;
                    let rc = rq.ckpts.iter().find(|c| c.step == s).ok_or("missing reference checkpoint")?;
                    compare_ckpt(&mut run.diff, &format!("req {req} step {s}"), rc, &items)?;
                    run.ckpts += 1;
                }
            }
            other => run.diff.note(format!("req {req}: unexpected plan row kind {other:?}")),
        }
    }
    Ok(())
}

fn drive(
    w: &mut World,
    refs: &[RefReq],
    run: &mut Run,
    steps: usize,
    ckpt_steps: &[usize],
    opts: &RunOpts,
) -> R<()> {
    let early_cap = EARLY_AR_STEPS.min(steps);
    loop {
        if run.diff.bad > 0 || run.error.is_some() {
            break;
        }
        let all_prefilled = run.lanes.iter().all(|l| l.prefilled == l.prompt_len);
        let pre = run.lanes.iter().position(|l| l.prefilled < l.prompt_len);
        let ar: Vec<usize> = run
            .lanes
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                l.prefilled == l.prompt_len && l.ar_steps < steps && (all_prefilled || l.ar_steps < early_cap)
            })
            .map(|(i, _)| i)
            .collect();
        if pre.is_none() && ar.is_empty() {
            break;
        }
        let mut specs = Vec::new();
        if let Some(i) = pre {
            let l = &run.lanes[i];
            let len = w.bundle.lane_executor().map_err(e)?.exact_prefill_chunk_len(&l.epoch)?;
            if len == 0 || l.prefilled + len > l.prompt_len {
                return Err(format!("req {}: prefill tile of {len} rows at {}/{}", l.req, l.prefilled, l.prompt_len));
            }
            specs.push(RowSpec {
                slot: l.slot,
                epoch: l.epoch,
                tokens: refs[l.req].prompt[l.prefilled..l.prefilled + len].to_vec(),
                start: l.prefilled,
                kind: RequestStepKind::Prefill,
            });
        }
        for &i in &ar {
            specs.push(ar_spec(w, &run.lanes[i])?);
        }

        // Wrong position: one extra AR row for lane 0 before its third step.
        if run.control == Control::WrongPosition && !run.fired {
            if let Some(&i) = ar.iter().find(|&&i| run.lanes[i].req == 0 && run.lanes[i].ar_steps == 2) {
                let solo = make_plan(w.max_lanes, vec![ar_spec(w, &run.lanes[i])?]);
                run.fired = true;
                match exec_step(w, &solo, false) {
                    Ok(_) => {
                        run.control_note = Some(format!(
                            "request 0 ran one extra AR row before step 3 (position {} instead of {})",
                            run.lanes[i].prompt_len + 3,
                            run.lanes[i].prompt_len + 2
                        ));
                    }
                    Err(x) => {
                        run.error = Some(format!("extra AR row: {}", x.text()));
                    }
                }
                continue;
            }
        }

        let mut plan = make_plan(w.max_lanes, specs);

        // Lane swap controls fire on the first plan with two AR rows past step 2.
        let swap = matches!(run.control, Control::SwapTokens | Control::SwapRowSlot | Control::SwapRows);
        let mut swapped = false;
        if swap && !run.fired {
            let eligible: Vec<usize> = ar.iter().copied().filter(|&i| run.lanes[i].ar_steps >= 2).collect();
            if eligible.len() >= 2 {
                let find = |lane: &Lane| plan.requests.iter().position(|r| r.epoch == lane.epoch).unwrap();
                let (ra, rb) = (find(&run.lanes[eligible[0]]), find(&run.lanes[eligible[1]]));
                let (ba, bb) = (plan.requests[ra].rows.begin, plan.requests[rb].rows.begin);
                if plan.batch.tokens[ba] != plan.batch.tokens[bb] {
                    let b = &mut plan.batch;
                    if matches!(run.control, Control::SwapTokens | Control::SwapRows) {
                        b.tokens.swap(ba, bb);
                    }
                    if matches!(run.control, Control::SwapRowSlot | Control::SwapRows) {
                        b.row_slot.swap(ba, bb);
                    }
                    if run.control == Control::SwapRows {
                        b.positions.swap(ba, bb);
                    }
                    swapped = true;
                    run.fired = true;
                    run.control_note = Some(format!(
                        "requests {} and {} swapped ({}) at their step {}",
                        run.lanes[eligible[0]].req,
                        run.lanes[eligible[1]].req,
                        run.control.name(),
                        run.lanes[eligible[0]].ar_steps + 1
                    ));
                }
            }
        }

        let res = match exec_step(w, &plan, true) {
            Ok(r) => r,
            Err(x) if swapped && x.stage == "provision" => {
                // Refusal at provision is the detection; nothing ran, the
                // run continues and its comparisons cover "state untouched".
                run.refused = Some(x.msg);
                continue;
            }
            Err(x) => {
                run.error = Some(x.text());
                if swapped {
                    run.control_note = Some(format!("{} (executor error counts as detection)", run.control_note.clone().unwrap_or_default()));
                }
                break;
            }
        };
        absorb(w, refs, run, &plan, &res, steps, ckpt_steps, opts)?;
    }
    if run.diff.bad == 0 && run.error.is_none() {
        for l in &run.lanes {
            if l.prefilled != l.prompt_len
                || l.ar_steps != steps
                || l.committed.len() != steps + 1
                || l.committed[..steps] != refs[l.req].ids[..steps]
            {
                run.diff.note(format!(
                    "req {}: run ended at prefill {}/{}, {}/{steps} AR steps, {} committed ids (expected {})",
                    l.req,
                    l.prefilled,
                    l.prompt_len,
                    l.ar_steps,
                    l.committed.len(),
                    steps + 1
                ));
            }
        }
    }
    Ok(())
}

struct RunReport {
    k: usize,
    label: String,
    pass: bool,
    steps: usize,
    lane_steps: usize,
    ckpts: usize,
    first: Option<String>,
    prefill_pass: bool,
    json: Value,
    run: Run,
}

struct App {
    w: World,
    refs: Vec<RefReq>,
    steps: usize,
    ckpt_steps: Vec<usize>,
    tag: u64,
    log: Log,
}

impl App {
    /// Admit `k` lanes, drive them against the references, release them.
    fn run_lanes(
        &mut self,
        k: usize,
        steps: usize,
        policy: LaneStagePolicy,
        label: &str,
        control: Control,
        opts: &RunOpts,
    ) -> RunReport {
        let started = Instant::now();
        let mut run = Run {
            lanes: Vec::new(),
            diff: Diff::default(),
            pre_bad: 0,
            pre_checked: 0,
            lane_steps: 0,
            ckpts: 0,
            error: None,
            control,
            fired: false,
            refused: None,
            control_note: None,
            release_errors: Vec::new(),
        };
        let res = (|| -> R<()> {
            self.w
                .bundle
                .lanes_mut()
                .ok_or("lane store is not staged")?
                .set_stage_policy(policy);
            for req in 0..k {
                self.tag += 1;
                let epoch = RequestEpoch { request_tag: self.tag, owner_generation: 1 };
                let prompt = self.refs[req].prompt.clone();
                let prompt_len = prompt.len();
                let slot = self
                    .w
                    .bundle
                    .admit_lane(&mut self.w.gpu, epoch, prompt)
                    .map_err(|x| format!("admit request {req}: {x}"))?;
                run.lanes.push(Lane { req, epoch, slot, prompt_len, prefilled: 0, committed: Vec::new(), ar_steps: 0 });
            }
            drive(&mut self.w, &self.refs, &mut run, steps, &self.ckpt_steps, opts)
        })();
        if let Err(m) = res {
            run.error = Some(m);
        }
        for l in &run.lanes {
            if let Err(x) = self.w.bundle.release_lane(&l.epoch) {
                run.release_errors.push(format!("release request {}: {x}", l.req));
            }
        }
        let clean = run.diff.bad == 0 && run.error.is_none() && run.release_errors.is_empty();
        let first = run
            .error
            .clone()
            .map(|m| format!("executor error: {m}"))
            .or_else(|| run.diff.first.clone())
            .or_else(|| run.release_errors.first().cloned());
        let prefill_pass = run.pre_bad == 0 && run.pre_checked == k && run.error.is_none();
        let json = json!({
            "k": k,
            "policy": label,
            "control": control.name(),
            "pass": clean,
            "steps": steps,
            "lane_steps_compared": run.lane_steps,
            "state_checkpoints_compared": run.ckpts,
            "items_compared": run.diff.items,
            "bytes_compared": run.diff.bytes,
            "differences": run.diff.bad,
            "first_divergence": first,
            "first_divergences": run.diff.shown,
            "executor_error": run.error,
            "provision_refusal": run.refused,
            "control_note": run.control_note,
            "release_errors": run.release_errors,
            "prefill_pass": prefill_pass,
            "wall_s": started.elapsed().as_secs_f64(),
        });
        RunReport {
            k,
            label: label.to_string(),
            pass: clean,
            steps,
            lane_steps: run.lane_steps,
            ckpts: run.ckpts,
            first,
            prefill_pass,
            json,
            run,
        }
    }

    fn say_run(&mut self, r: &RunReport) {
        let verdict = if r.pass { "PASS" } else { "FAIL" };
        let mut line = format!(
            "run k={} policy={}: {verdict} (steps {}, lane-steps compared {}, state checkpoints {})",
            r.k, r.label, r.steps, r.lane_steps, r.ckpts
        );
        if let Some(first) = &r.first {
            line += &format!("\n    first divergence: {first}");
        }
        self.log.say(line);
    }
}

// ── bench ────────────────────────────────────────────────────────────

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

impl App {
    /// Wall time per lane step at `k` lanes (production policy), median over `n`.
    fn bench_lanes(&mut self, k: usize, n: usize) -> R<f64> {
        self.w
            .bundle
            .lanes_mut()
            .ok_or("lane store is not staged")?
            .set_stage_policy(LaneStagePolicy::production());
        let mut lanes: Vec<Lane> = Vec::new();
        let result = (|| -> R<f64> {
            for req in 0..k {
                self.tag += 1;
                let epoch = RequestEpoch { request_tag: self.tag, owner_generation: 1 };
                let prompt = self.refs[req].prompt.clone();
                let prompt_len = prompt.len();
                let slot = self.w.bundle.admit_lane(&mut self.w.gpu, epoch, prompt).map_err(e)?;
                lanes.push(Lane { req, epoch, slot, prompt_len, prefilled: 0, committed: Vec::new(), ar_steps: 0 });
            }
            while let Some(i) = lanes.iter().position(|l| l.prefilled < l.prompt_len) {
                let l = &lanes[i];
                let len = self.w.bundle.lane_executor().map_err(e)?.exact_prefill_chunk_len(&l.epoch)?;
                let plan = make_plan(
                    self.w.max_lanes,
                    vec![RowSpec {
                        slot: l.slot,
                        epoch: l.epoch,
                        tokens: self.refs[l.req].prompt[l.prefilled..l.prefilled + len].to_vec(),
                        start: l.prefilled,
                        kind: RequestStepKind::Prefill,
                    }],
                );
                exec_step(&mut self.w, &plan, false).map_err(|x| x.text())?;
                lanes[i].prefilled += len;
            }
            let mut times = Vec::with_capacity(n);
            for it in 0..n + 3 {
                let specs = lanes.iter().map(|l| ar_spec(&self.w, l)).collect::<R<Vec<_>>>()?;
                let plan = make_plan(self.w.max_lanes, specs);
                let t = Instant::now();
                exec_step(&mut self.w, &plan, false).map_err(|x| x.text())?;
                self.w.gpu.hip.device_synchronize().map_err(e)?;
                if it >= 3 {
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            Ok(median(&mut times))
        })();
        for l in &lanes {
            let _ = self.w.bundle.release_lane(&l.epoch);
        }
        result
    }

    /// Wall time per singleton `forward_token_or_argmax`, median over `n`.
    fn bench_singleton(&mut self, n: usize) -> R<f64> {
        let prompt = self.refs[0].prompt.clone();
        self.w.bundle.reset(&mut self.w.gpu).map_err(e)?;
        self.w.bundle.prefill_final(&mut self.w.gpu, &prompt, 0, &self.w.row).map_err(e)?;
        let mut times = Vec::with_capacity(n);
        for it in 0..n + 3 {
            self.w.gpu.hip.device_synchronize().map_err(e)?;
            let t = Instant::now();
            self.w.bundle.forward_token_or_argmax(&mut self.w.gpu, None, &self.w.row).map_err(e)?;
            self.w.gpu.hip.device_synchronize().map_err(e)?;
            if it >= 3 {
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
        }
        Ok(median(&mut times))
    }
}

// ── model load ───────────────────────────────────────────────────────

fn load(path: &Path, max_seq: usize, max_k: usize) -> R<(HfqModelSource, World, Tokenizer)> {
    let mut hfq = HfqFile::open(path).map_err(e)?;
    let tokenizer = Tokenizer::from_hfq_metadata(&hfq.metadata_json).map_err(e)?;
    let receipt = admit_hfqm_artifact(&hfq).map_err(e)?;
    let mut gpu = Gpu::init().map_err(e)?;
    if gpu.is_uma() {
        hfq.drop_mmap();
    }
    let mesh = DeviceMesh::single().map_err(e)?;
    let expected = WeightOrigin::for_single(&mesh, &gpu);
    let source = HfqModelSource::from_hfq(hfq);
    let transaction = fulfill_manifest_from_payloads(
        &receipt.manifest.weights,
        &mesh,
        receipt.config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            source
                .tensor_range(&entry.name)
                .map_err(|x| x.to_string())?
                .map(SourcePayload::Range)
                .ok_or_else(|| format!("missing tensor '{}'", entry.name))
        },
    )
    .map_err(e)?;
    let vocab = receipt.config.vocab_size;
    let state_format = hipfire_arch_qwen4::resolve_state_format(
        &hipfire_runtime::config::get().kv_mode,
        &std::env::var("HIPFIRE_STATE_QUANT").unwrap_or_default(),
        &gpu,
        &receipt.config,
    )?;
    let backend = hipfire_arch_qwen4::Qwen4KvBackend::automatic(&gpu);
    let mut bundle = Qwen4Bundle::assemble_with_metadata(
        receipt.config,
        transaction,
        &receipt.placements,
        &mut gpu,
        max_seq,
        receipt.ple,
        state_format,
        backend,
    )
    .map_err(e)?;
    bundle.attach_forward(&mut gpu, max_seq).map_err(e)?;
    bundle.stage_lanes(&mut gpu, max_k, max_k).map_err(e)?;
    let row = gpu.zeros(&[vocab], DType::F32).map_err(e)?;
    let max_lanes = bundle.lanes().map(|s| s.max_lanes()).ok_or("lane store is not staged")?;
    Ok((source, World { gpu, bundle, vocab, row, max_lanes }, tokenizer))
}

fn prompt_for(tokens: &[u32], i: usize, ctx: usize) -> Vec<u32> {
    (0..ctx).map(|j| tokens[(i * 997 + j) % tokens.len()]).collect()
}

fn stage_rust_line(stage: StageName, share: StageShare) -> String {
    format!("    (StageName::{stage:?}, StageShare::{share:?}),")
}

// ── main ─────────────────────────────────────────────────────────────

fn main() {
    match real_main() {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(m) => {
            eprintln!("fn_lane_oracle: {m}");
            std::process::exit(2);
        }
    }
}

fn real_main() -> R<bool> {
    let args = parse_args()?;
    fs::create_dir_all(&args.out).map_err(e)?;
    let primary = LaneStagePolicy::parse(&args.policy)?;
    let mut max_k = *args.ks.iter().max().unwrap();
    if args.g0 {
        max_k = max_k.max(8);
    }
    if args.controls {
        max_k = max_k.max(2);
    }
    let max_ctx = *args.contexts.iter().max().unwrap();
    let longest_run = args.steps.max(args.bench + 3);
    if max_ctx + longest_run + 2 > args.max_seq {
        return Err(format!(
            "--max-seq {} cannot hold the longest context {max_ctx} + {longest_run} decode steps",
            args.max_seq
        ));
    }

    let mut log = Log::default();
    let file_size = fs::metadata(&args.model).map_err(e)?.len();
    let (_source, mut w, tokenizer) = load(&args.model, args.max_seq, max_k)?;
    log.say(format!("gpu arch: {}", w.gpu.arch));
    log.say(format!("model: {} ({file_size} bytes)", args.model.display()));
    {
        let store = w.bundle.lanes().ok_or("lane store is not staged")?;
        let r = store.receipt(&w.gpu);
        log.say(format!(
            "lanes: kv_backend={} qsa={} gdn={} ring_rows={} max_lanes={} row_budget={} mapped_bytes={} stage_policy={} evidence={}",
            r.kv_backend, r.qsa_format, r.gdn_format, r.ring_rows, r.max_lanes, r.row_budget, r.mapped_bytes,
            r.stage_policy, r.stage_evidence
        ));
    }
    log.say(format!(
        "args: ks={:?} contexts={:?} steps={} max_seq={} policy={} ({}) g0={} controls={} bench={}",
        args.ks, args.contexts, args.steps, args.max_seq, args.policy, primary.describe(), args.g0,
        args.controls, args.bench
    ));
    if !primary.evidence().is_empty() {
        log.say(format!("policy evidence: {}", primary.evidence()));
    }

    let text = fs::read_to_string(&args.text).map_err(e)?;
    let toks = tokenizer.encode(&text);
    if toks.is_empty() {
        return Err("TEXT tokenized to nothing".into());
    }
    if toks.len() < max_ctx {
        log.say(format!(
            "warning: TEXT has {} tokens, fewer than the longest context {max_ctx}; prompts wrap",
            toks.len()
        ));
    }

    // Checkpoints: prefill, step 1, the shortened g0/control run, the last step.
    let short_steps = args.steps.min(16);
    let mut ckpt_steps = vec![1, args.steps];
    if args.g0 || args.controls {
        ckpt_steps.push(short_steps);
    }
    ckpt_steps.sort_unstable();
    ckpt_steps.dedup();

    // ── singleton references ───────────────────────────────────────
    let mut refs = Vec::with_capacity(max_k);
    for i in 0..max_k {
        let ctx = args.contexts[i % args.contexts.len()];
        let prompt = prompt_for(&toks, i, ctx);
        let dir = args.out.join("ref").join(format!("req{i}"));
        let t = Instant::now();
        let rq = record_ref(&mut w, i, prompt, args.steps, &ckpt_steps, &dir)?;
        log.say(format!(
            "reference req{i}: context {ctx}, {} steps, ids[0..4]={:?} ({:.1}s)",
            args.steps,
            &rq.ids[..rq.ids.len().min(4)],
            t.elapsed().as_secs_f64()
        ));
        refs.push(rq);
    }

    let mut app = App { w, refs, steps: args.steps, ckpt_steps, tag: 0, log };
    let full = RunOpts { prefill_ckpt: true };
    let mut all_pass = true;
    let mut runs: Vec<Value> = Vec::new();
    let mut s1: Option<bool> = None;

    // ── main matrix ────────────────────────────────────────────────
    for &k in &args.ks {
        let r = app.run_lanes(k, args.steps, primary, &args.policy, Control::None, &full);
        app.say_run(&r);
        all_pass &= r.pass;
        if k == 2 && s1.is_none() {
            s1 = Some(r.prefill_pass);
        }
        runs.push(r.json);
    }
    if s1.is_none() {
        let steps = app.steps.min(1);
        let r = app.run_lanes(2, steps, primary, &args.policy, Control::None, &full);
        s1 = Some(r.prefill_pass);
        all_pass &= r.pass;
        runs.push(r.json);
    }
    let s1_pass = s1.unwrap_or(false);
    all_pass &= s1_pass;
    app.log.say(format!("S1 prefill parity k=2: {}", if s1_pass { "PASS" } else { "FAIL" }));

    // ── G0 per-stage probe ─────────────────────────────────────────
    let mut g0_json = Value::Null;
    if args.g0 {
        let probe = RunOpts { prefill_ckpt: false };
        let mut verdicts = Vec::new();
        let mut combined = LaneStagePolicy::all_per_lane();
        let mut identical_stages = Vec::new();
        for stage in StageName::ALL {
            if stage.stateful() {
                app.log.say(format!("g0 stage {}: per lane by construction (stateful)", stage.key()));
                verdicts.push(json!({"stage": stage.key(), "verdict": "stateful", "identical": false}));
                continue;
            }
            let policy = match LaneStagePolicy::all_per_lane().with(stage, StageShare::OnceAllRows) {
                Ok(p) => p,
                Err(m) => {
                    app.log.say(format!("g0 stage {}: policy refused: {m}", stage.key()));
                    verdicts.push(json!({"stage": stage.key(), "verdict": format!("refused: {m}"), "identical": false}));
                    continue;
                }
            };
            let mut identical = true;
            let mut first: Option<String> = None;
            let mut per_k = Vec::new();
            for k in [2usize, 8] {
                let label = format!("g0:{}=once", stage.key());
                let r = app.run_lanes(k, short_steps, policy, &label, Control::None, &probe);
                if !r.pass {
                    identical = false;
                    first.get_or_insert_with(|| format!("k={k}: {}", r.first.clone().unwrap_or_default()));
                }
                per_k.push(r.json);
            }
            if identical {
                app.log.say(format!("g0 stage {}: identical", stage.key()));
                identical_stages.push(stage);
                combined = combined.with(stage, StageShare::OnceAllRows)?;
            } else {
                app.log.say(format!(
                    "g0 stage {}: differs (first divergence: {})",
                    stage.key(),
                    first.clone().unwrap_or_default()
                ));
            }
            verdicts.push(json!({
                "stage": stage.key(),
                "verdict": if identical { "identical" } else { "differs" },
                "identical": identical,
                "first_divergence": first,
                "runs": per_k,
            }));
        }
        app.log.say(format!("g0 combined policy: {}", combined.describe()));
        app.log.say("g0 combined STAGE_SHARE:".to_string());
        let mut table = Vec::new();
        for stage in StageName::ALL {
            let line = stage_rust_line(stage, combined.share(stage));
            app.log.say(line.clone());
            table.push(line);
        }
        let mut combined_runs = Vec::new();
        for &k in &args.ks {
            let r = app.run_lanes(k, args.steps, combined, "g0:combined", Control::None, &full);
            app.say_run(&r);
            all_pass &= r.pass;
            combined_runs.push(r.json);
        }
        g0_json = json!({
            "stages": verdicts,
            "identical_stages": identical_stages.iter().map(|s| s.key()).collect::<Vec<_>>(),
            "combined_policy": combined.describe(),
            "combined_stage_share": table,
            "combined_runs": combined_runs,
        });
    }

    // ── negative controls ──────────────────────────────────────────
    let mut controls_json = Vec::new();
    if args.controls {
        if short_steps < 4 {
            return Err("--controls needs --steps >= 4".into());
        }
        let probe = RunOpts { prefill_ckpt: true };
        for control in [Control::SwapTokens, Control::SwapRowSlot, Control::SwapRows, Control::WrongPosition] {
            let r = app.run_lanes(2, short_steps, primary, &args.policy, control, &probe);
            let refused = r.run.refused.clone();
            let diverged = r.run.diff.bad > 0 || r.run.error.is_some();
            let (detected, how) = if !r.run.fired {
                (false, "control was not exercised (no eligible step)".to_string())
            } else if let Some(m) = &refused {
                (true, format!("refused at provision: {m}"))
            } else if diverged {
                (true, format!("comparison failed: {}", r.first.clone().unwrap_or_default()))
            } else {
                (false, "comparison still passed".to_string())
            };
            // A refused plan must leave every lane byte-exact for the rest of the run.
            let clean_after_refusal = refused.is_none() || r.pass;
            let ok = detected && clean_after_refusal;
            all_pass &= ok;
            let mut line = format!(
                "control {}: {}",
                control.name(),
                if detected { "detected" } else { "MISSED" }
            );
            line += &format!(" ({how})");
            if !clean_after_refusal {
                line += &format!(
                    "; FAIL: the refusal left lane state modified: {}",
                    r.first.clone().unwrap_or_default()
                );
            }
            app.log.say(line);
            controls_json.push(json!({
                "control": control.name(),
                "detected": detected,
                "how": how,
                "refused_at_provision": refused,
                "clean_after_refusal": clean_after_refusal,
                "note": r.run.control_note,
                "run": r.json,
            }));
        }
        app.log.say(
            "control lane_swap (device level): no test-visible path exchanges two lanes' Qwen4State; \
             the plan-level swaps above are the lane-swap controls"
                .to_string(),
        );
    }

    // ── bench ──────────────────────────────────────────────────────
    let mut bench_json = Vec::new();
    if args.bench > 0 {
        let n = args.bench;
        let singleton_ms = app.bench_singleton(n)?;
        app.log.say(format!("bench singleton: {singleton_ms:.3} ms/step (median of {n}, request 0)"));
        for &k in &args.ks {
            let step_ms = app.bench_lanes(k, n)?;
            let tok_s = k as f64 * 1000.0 / step_ms;
            let speedup = tok_s / (1000.0 / singleton_ms);
            app.log.say(format!(
                "bench k={k}, step_ms={step_ms:.3}, tok_s={tok_s:.2}, singleton_ms={singleton_ms:.3}, speedup={speedup:.3}x"
            ));
            bench_json.push(json!({
                "k": k, "step_ms": step_ms, "tok_s": tok_s, "singleton_ms": singleton_ms, "speedup": speedup
            }));
        }
    }

    app.log.say(format!("fn_lane_oracle: {}", if all_pass { "PASS" } else { "FAIL" }));
    let summary = json!({
        "pass": all_pass,
        "gpu_arch": app.w.gpu.arch,
        "model": args.model.display().to_string(),
        "model_bytes": file_size,
        "args": {
            "ks": args.ks, "contexts": args.contexts, "steps": args.steps, "max_seq": args.max_seq,
            "policy": args.policy, "policy_described": primary.describe(), "g0": args.g0,
            "controls": args.controls, "bench": args.bench,
        },
        "policy_evidence": primary.evidence(),
        "requests": app.refs.iter().map(|r| json!({"context": r.prompt.len(), "dir": r.dir.display().to_string()})).collect::<Vec<_>>(),
        "s1_prefill_parity_k2": s1_pass,
        "runs": runs,
        "g0": g0_json,
        "controls": controls_json,
        "bench": bench_json,
        "log": app.log.lines,
    });
    fs::write(args.out.join("summary.json"), serde_json::to_vec_pretty(&summary).map_err(e)?).map_err(e)?;
    Ok(all_pass)
}
