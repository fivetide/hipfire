// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! raon_oracle `state` subcommand (PLAN §6.2, slice S gate).
//!
//! State-only checks that exercise the production owners in
//! `hipfire_arch_raon::state` on the real device:
//!
//! * `slot_isolation`      two slots, unequal lengths, interleaved decode-style
//!                         writes, poisoned rows beyond each length, per-slot
//!                         reset/abort, snapshot/restore, failed admission;
//! * `ctx_reservation_*`   virtual reservations at ctx 512/8192/32768/65536/98304,
//!                         boundary growth with data integrity across the chunk
//!                         edge, physical bytes versus the ledger, recurrent bytes
//!                         constant in max_seq;
//! * `admission_*`         clean refusals (oversized reservation, 262144 admitted
//!                         only if the device backs it, growth refused under
//!                         memory pressure without mutating the slot);
//! * `scratch_*`           bounded scratch at 1/512/1024 rows vs plan bytes;
//! * `expanded_mirror`     admission limits and ledger;
//! * `vmm_owner_table`     every VMM owner released.
//!
//! `--mode serial|batched|graph` additionally requires the integrated forward:
//! [`run_forward_mode`] returns `Err("forward not integrated")` until slice I
//! lands, and `run` then reports `pass=false` — never a placeholder pass.

use std::collections::BTreeMap;
use std::path::Path;

use hip_bridge::DeviceBuffer;
use hipfire_arch_raon::state::{
    auto_backend, expanded_mirror_bytes, scratch_plan, state_budget, LatentBackend, RaonPrefillScratch,
    RaonState, RaonStateDims, ADMISSION_HEADROOM_BYTES, EXPANDED_MIRROR_MAX_ROWS,
};
use rdna_compute::{DType, Gpu};
use serde_json::{json, Value};

const DEFAULT_CTXS: [usize; 5] = [512, 8192, 32768, 65536, 98304];
const MIB: usize = 1 << 20;

type Check = Result<Value, String>;

/// Entry point shared with `raon_oracle state`.
pub fn run(_fixtures: &Path, opts: &BTreeMap<String, String>) -> Result<Value, String> {
    let mode = opts.get("mode").map(String::as_str).unwrap_or("state-only");
    let forward_mode = match mode {
        "state-only" | "state" => None,
        "serial" | "batched" | "graph" => Some(mode),
        other => return Err(format!("state: unknown --mode {other} (serial|batched|graph|state-only)")),
    };
    let ctxs = match opts.get("ctxs") {
        Some(s) => s
            .split(',')
            .map(|x| x.trim().parse::<usize>().map_err(|e| format!("state: bad ctx {x:?}: {e}")))
            .collect::<Result<Vec<_>, _>>()?,
        None => DEFAULT_CTXS.to_vec(),
    };
    let dims = RaonStateDims::raon();
    dims.validate()?;
    let mut gpu = Gpu::init().map_err(|e| format!("state: gpu init: {e:?}"))?;

    let mut checks: Vec<Value> = Vec::new();
    let mut record = |name: String, r: Check| {
        checks.push(match r {
            Ok(detail) => json!({"name": name, "pass": true, "detail": detail}),
            Err(e) => json!({"name": name, "pass": false, "error": e}),
        });
    };

    record("slot_isolation".into(), check_slot_isolation(&mut gpu, &dims));
    let mut recurrent_bytes: Vec<(usize, usize)> = Vec::new();
    for &ctx in &ctxs {
        let r = check_ctx_reservation(&mut gpu, &dims, ctx, &mut recurrent_bytes);
        record(format!("ctx_reservation_{ctx}"), r);
    }
    record(
        "rda_bytes_constant_in_max_seq".into(),
        check_recurrent_constant(&recurrent_bytes),
    );
    record("admission_refusal".into(), check_admission_refusal(&mut gpu, &dims));
    record("admission_262144".into(), check_admission_262144(&mut gpu, &dims));
    record(
        "admission_growth_under_pressure".into(),
        check_growth_pressure(&mut gpu, &dims),
    );
    for rows in [1usize, 512, 1024] {
        record(format!("scratch_{rows}"), check_scratch(&mut gpu, &dims, rows));
    }
    record("scratch_rows_limit".into(), check_scratch_limit(&mut gpu, &dims));
    record("expanded_mirror".into(), check_mirror(&mut gpu, &dims));
    gpu.drain_pool();
    record("vmm_owner_table".into(), check_vmm_clean(&mut gpu));

    let state_pass = checks.iter().all(|c| c["pass"].as_bool() == Some(true));
    let mut out = json!({
        "module": "state",
        "arch": gpu.arch,
        "mode": mode,
        "ctxs": ctxs,
        "state_checks_pass": state_pass,
        "checks": checks,
    });
    let mut pass = state_pass;
    if let Some(m) = forward_mode {
        match run_forward_mode(&mut gpu, m, opts) {
            Ok(v) => {
                out["forward"] = json!({"status": "ok", "result": v});
            }
            Err(e) => {
                pass = false;
                out["forward"] = json!({"status": "error", "error": e});
            }
        }
    }
    out["pass"] = json!(pass);
    Ok(out)
}

/// Full-forward state parity (`serial|batched|graph`): composes embedding,
/// layers and head through the production forward and compares every RDA
/// state/conv window/live latent row. Filled in by integration (slice I/B/G).
pub fn run_forward_mode(
    _gpu: &mut Gpu,
    _mode: &str,
    _opts: &BTreeMap<String, String>,
) -> Result<Value, String> {
    Err("forward not integrated".to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// Byte patterns
// ─────────────────────────────────────────────────────────────────────────────

fn pat_byte(seed: u64, idx: usize) -> u8 {
    let mut x = seed ^ (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    (x & 0xFF) as u8
}

fn view(buf: &DeviceBuffer, off: usize, len: usize, label: &str) -> Result<DeviceBuffer, String> {
    if off.checked_add(len).is_none_or(|e| e > buf.size()) {
        return Err(format!(
            "{label}: range [{off}, {off}+{len}) exceeds device extent {}",
            buf.size()
        ));
    }
    Ok(buf.byte_view(off, len))
}

fn hip_err<T>(r: hip_bridge::HipResult<T>, what: &str) -> Result<T, String> {
    r.map_err(|e| format!("{what}: {e:?}"))
}

/// Write `len` bytes at `off` where byte `k` is `pat(seed, idx0 + k)`.
fn write_pat(
    gpu: &Gpu,
    buf: &DeviceBuffer,
    off: usize,
    len: usize,
    seed: u64,
    idx0: usize,
    label: &str,
) -> Result<(), String> {
    let host: Vec<u8> = (0..len).map(|k| pat_byte(seed, idx0 + k)).collect();
    let v = view(buf, off, len, label)?;
    hip_err(gpu.hip.memcpy_htod(&v, &host), label)
}

fn fill_const(gpu: &Gpu, buf: &DeviceBuffer, off: usize, len: usize, val: u8, label: &str) -> Result<(), String> {
    let host = vec![val; len];
    let v = view(buf, off, len, label)?;
    hip_err(gpu.hip.memcpy_htod(&v, &host), label)
}

fn read_bytes(gpu: &Gpu, buf: &DeviceBuffer, off: usize, len: usize, label: &str) -> Result<Vec<u8>, String> {
    let v = view(buf, off, len, label)?;
    let mut host = vec![0u8; len];
    hip_err(gpu.hip.memcpy_dtoh(&mut host, &v), label)?;
    Ok(host)
}

fn check_pat(
    gpu: &Gpu,
    buf: &DeviceBuffer,
    off: usize,
    len: usize,
    seed: u64,
    idx0: usize,
    label: &str,
) -> Result<(), String> {
    let host = read_bytes(gpu, buf, off, len, label)?;
    for (k, &got) in host.iter().enumerate() {
        let want = pat_byte(seed, idx0 + k);
        if got != want {
            return Err(format!(
                "{label}: first mismatch at byte {k} (abs {}): got 0x{got:02x} want 0x{want:02x}",
                off + k
            ));
        }
    }
    Ok(())
}

fn check_const(gpu: &Gpu, buf: &DeviceBuffer, off: usize, len: usize, val: u8, label: &str) -> Result<(), String> {
    let host = read_bytes(gpu, buf, off, len, label)?;
    if let Some(k) = host.iter().position(|&b| b != val) {
        return Err(format!(
            "{label}: first mismatch at byte {k} (abs {}): got 0x{:02x} want 0x{val:02x}",
            off + k,
            host[k]
        ));
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Slot content helpers
// ─────────────────────────────────────────────────────────────────────────────

fn rda_seed(slot: u64, layer: usize) -> u64 {
    slot ^ 0xA5A5_0000 ^ ((layer as u64) << 8)
}
fn conv_seed(slot: u64, layer: usize) -> u64 {
    slot ^ 0x5A5A_0000 ^ ((layer as u64) << 8) ^ 0x77
}
fn row_seed(slot: u64, layer: usize) -> u64 {
    slot ^ 0x1234_0000 ^ ((layer as u64) << 8) ^ 0x31
}

fn fill_recurrent(gpu: &Gpu, st: &RaonState, slot: u64) -> Result<(), String> {
    for i in 0..st.dims().n_rda {
        let r = &st.rda[i].buf;
        write_pat(gpu, r, 0, r.size(), rda_seed(slot, i), 0, &format!("fill rda[{i}]"))?;
        for (p, c) in [&st.conv_q[i], &st.conv_k[i], &st.conv_v[i]].into_iter().enumerate() {
            let c = &c.buf;
            write_pat(gpu, c, 0, c.size(), conv_seed(slot, i) ^ ((p as u64) << 40), 0, &format!("fill conv{p}[{i}]"))?;
        }
    }
    Ok(())
}

fn verify_recurrent(gpu: &Gpu, st: &RaonState, slot: u64, label: &str) -> Result<(), String> {
    for i in 0..st.dims().n_rda {
        let r = &st.rda[i].buf;
        check_pat(gpu, r, 0, r.size(), rda_seed(slot, i), 0, &format!("{label}: rda[{i}]"))?;
        for (p, c) in [&st.conv_q[i], &st.conv_k[i], &st.conv_v[i]].into_iter().enumerate() {
            let c = &c.buf;
            check_pat(gpu, c, 0, c.size(), conv_seed(slot, i) ^ ((p as u64) << 40), 0, &format!("{label}: conv{p}[{i}]"))?;
        }
    }
    Ok(())
}

fn verify_recurrent_zero(gpu: &Gpu, st: &RaonState, label: &str) -> Result<(), String> {
    for i in 0..st.dims().n_rda {
        let r = &st.rda[i].buf;
        check_const(gpu, r, 0, r.size(), 0, &format!("{label}: rda[{i}]"))?;
        for (p, c) in [&st.conv_q[i], &st.conv_k[i], &st.conv_v[i]].into_iter().enumerate() {
            let c = &c.buf;
            check_const(gpu, c, 0, c.size(), 0, &format!("{label}: conv{p}[{i}]"))?;
        }
    }
    Ok(())
}

fn write_rows(gpu: &Gpu, st: &RaonState, slot: u64, row0: usize, nrows: usize) -> Result<(), String> {
    let rb = st.dims().latent_row_bytes();
    for (l, t) in st.mla_cache.iter().enumerate() {
        write_pat(
            gpu,
            &t.buf,
            row0 * rb,
            nrows * rb,
            row_seed(slot, l),
            row0 * rb,
            &format!("write latent l{l} rows [{row0},+{nrows})"),
        )?;
    }
    Ok(())
}

fn verify_rows(gpu: &Gpu, st: &RaonState, slot: u64, row0: usize, nrows: usize, label: &str) -> Result<(), String> {
    let rb = st.dims().latent_row_bytes();
    for (l, t) in st.mla_cache.iter().enumerate() {
        check_pat(
            gpu,
            &t.buf,
            row0 * rb,
            nrows * rb,
            row_seed(slot, l),
            row0 * rb,
            &format!("{label}: latent l{l} rows [{row0},+{nrows})"),
        )?;
    }
    Ok(())
}

fn poison_rows(gpu: &Gpu, st: &RaonState, row0: usize, row1: usize) -> Result<(), String> {
    let rb = st.dims().latent_row_bytes();
    for (l, t) in st.mla_cache.iter().enumerate() {
        fill_const(gpu, &t.buf, row0 * rb, (row1 - row0) * rb, 0xFF, &format!("poison l{l}"))?;
    }
    Ok(())
}

fn verify_poison(gpu: &Gpu, st: &RaonState, row0: usize, row1: usize, label: &str) -> Result<(), String> {
    let rb = st.dims().latent_row_bytes();
    for (l, t) in st.mla_cache.iter().enumerate() {
        check_const(gpu, &t.buf, row0 * rb, (row1 - row0) * rb, 0xFF, &format!("{label}: poison l{l}"))?;
    }
    Ok(())
}

fn device_seq_len(gpu: &Gpu, st: &RaonState) -> Result<i32, String> {
    let b = read_bytes(gpu, &st.seq_len.buf, 0, 4, "read seq_len")?;
    Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn expect_seq(gpu: &Gpu, st: &RaonState, want: usize, label: &str) -> Result<(), String> {
    let dev = device_seq_len(gpu, st)?;
    if dev as usize != want || st.n_tokens() != want {
        return Err(format!(
            "{label}: seq_len device {dev} / host {} != {want}",
            st.n_tokens()
        ));
    }
    Ok(())
}

fn mapped_bytes(gpu: &Gpu, st: &RaonState) -> Vec<usize> {
    st.mla_cache
        .iter()
        .map(|t| gpu.vmm_mapped_bytes(t).unwrap_or(t.buf.size()))
        .collect()
}

fn free_bytes(gpu: &Gpu) -> Result<usize, String> {
    hip_err(gpu.device_mem_info(), "device_mem_info").map(|(f, _)| f)
}

/// Physical delta must cover the ledger and not exceed it by more than
/// allocator rounding. Skipped on unified-memory parts whose free figure is a
/// host-RAM estimate.
fn check_delta(gpu: &Gpu, before: usize, after: usize, ledger_physical: usize, label: &str) -> Result<Value, String> {
    let delta = before.saturating_sub(after);
    if gpu.arch == "gfx1151" {
        return Ok(json!({"label": label, "delta_bytes": delta, "ledger_bytes": ledger_physical, "enforced": false}));
    }
    let slack = 64 * MIB;
    if delta + 4 * MIB < ledger_physical || delta > ledger_physical + slack {
        return Err(format!(
            "{label}: physical delta {delta} B vs ledger {ledger_physical} B (allowed -4 MiB..+64 MiB)"
        ));
    }
    Ok(json!({"label": label, "delta_bytes": delta, "ledger_bytes": ledger_physical, "enforced": true}))
}

// ─────────────────────────────────────────────────────────────────────────────
// Checks
// ─────────────────────────────────────────────────────────────────────────────

fn check_slot_isolation(gpu: &mut Gpu, dims: &RaonStateDims) -> Check {
    const A: u64 = 0xA000;
    const B: u64 = 0xB000;
    let max_seq = 4096;
    let mut a = RaonState::new(gpu, dims, max_seq)?;
    let mut b = match RaonState::new(gpu, dims, max_seq) {
        Ok(b) => b,
        Err(e) => {
            let _ = a.free_gpu(gpu);
            return Err(e);
        }
    };
    let r = (|| -> Check {
        let (la, lb) = (700usize, 3usize);
        fill_recurrent(gpu, &a, A)?;
        fill_recurrent(gpu, &b, B)?;
        a.ensure_capacity(gpu, la)?;
        b.ensure_capacity(gpu, lb)?;
        // Poison every mapped row past each slot's length first, then write
        // the live rows over the front of it.
        let (ca, cb) = (a.capacity_rows(), b.capacity_rows());
        poison_rows(gpu, &a, 0, ca)?;
        poison_rows(gpu, &b, 0, cb)?;
        write_rows(gpu, &a, A, 0, la)?;
        write_rows(gpu, &b, B, 0, lb)?;
        a.set_n_tokens(gpu, la)?;
        b.set_n_tokens(gpu, lb)?;
        expect_seq(gpu, &a, la, "a initial")?;
        expect_seq(gpu, &b, lb, "b initial")?;

        // Interleaved decode-style steps: each slot appends one row and bumps
        // its own length; a's recurrent block is rewritten each step.
        let steps = 8usize;
        let mut la_now = la;
        let mut lb_now = lb;
        for s in 0..steps {
            write_rows(gpu, &a, A, la_now, 1)?;
            la_now += 1;
            a.set_n_tokens(gpu, la_now)?;
            write_pat(gpu, &a.rda[0].buf, 0, a.rda[0].buf.size(), rda_seed(A, 0) ^ (s as u64 + 1), 0, "a rda step")?;
            write_rows(gpu, &b, B, lb_now, 1)?;
            lb_now += 1;
            b.set_n_tokens(gpu, lb_now)?;
            if s % 2 == 1 {
                // extra b-only step: unequal progress
                write_rows(gpu, &b, B, lb_now, 1)?;
                lb_now += 1;
                b.set_n_tokens(gpu, lb_now)?;
            }
        }
        // restore a's layer-0 expected content for the plain verify below.
        write_pat(gpu, &a.rda[0].buf, 0, a.rda[0].buf.size(), rda_seed(A, 0), 0, "a rda restore")?;
        expect_seq(gpu, &a, la_now, "a after interleave")?;
        expect_seq(gpu, &b, lb_now, "b after interleave")?;
        if la_now == lb_now {
            return Err("isolation scenario degenerated to equal lengths".into());
        }
        verify_recurrent(gpu, &a, A, "a after interleave")?;
        verify_recurrent(gpu, &b, B, "b after interleave")?;
        verify_rows(gpu, &a, A, 0, la_now, "a live rows")?;
        verify_rows(gpu, &b, B, 0, lb_now, "b live rows")?;
        verify_poison(gpu, &a, la_now, ca, "a beyond length")?;
        verify_poison(gpu, &b, lb_now, cb, "b beyond length")?;

        // Snapshot / restore on b (graph-warmup contract).
        let mut snap = b.snapshot_alloc(gpu)?;
        b.snapshot_into(gpu, &mut snap)?;
        fill_recurrent(gpu, &b, B ^ 0xFFFF)?;
        b.set_n_tokens(gpu, lb_now + 2)?;
        b.restore(gpu, &snap)?;
        expect_seq(gpu, &b, lb_now, "b after restore")?;
        verify_recurrent(gpu, &b, B, "b after restore")?;
        snap.free_gpu(gpu)?;

        // Failed admission: beyond the reservation. Slot is untouched.
        let mapped_a = mapped_bytes(gpu, &a);
        let ptrs_a: Vec<usize> = a.mla_cache.iter().map(|t| t.buf.as_ptr() as usize).collect();
        match a.ensure_capacity(gpu, max_seq + 1) {
            Err(e) if e.contains("max_seq") => {}
            other => return Err(format!("ensure_capacity beyond reservation must be refused, got {other:?}")),
        }
        if a.set_n_tokens(gpu, max_seq + 1).is_ok() {
            return Err("set_n_tokens beyond reservation must be refused".into());
        }
        if mapped_bytes(gpu, &a) != mapped_a || a.capacity_rows() != ca {
            return Err("failed admission changed mapped extent".into());
        }
        expect_seq(gpu, &a, la_now, "a after failed admission")?;
        verify_recurrent(gpu, &a, A, "a after failed admission")?;
        verify_rows(gpu, &a, A, 0, la_now, "a rows after failed admission")?;

        // Abort/reset b: recurrent+conv zero, length 0, mappings and addresses
        // unchanged, neighbor a bit-identical.
        let mapped_b = mapped_bytes(gpu, &b);
        let ptrs_b: Vec<usize> = b.mla_cache.iter().map(|t| t.buf.as_ptr() as usize).collect();
        b.reset(gpu)?;
        expect_seq(gpu, &b, 0, "b after reset")?;
        verify_recurrent_zero(gpu, &b, "b after reset")?;
        if mapped_bytes(gpu, &b) != mapped_b
            || b.mla_cache.iter().map(|t| t.buf.as_ptr() as usize).collect::<Vec<_>>() != ptrs_b
        {
            return Err("reset unmapped or moved a latent cache".into());
        }
        verify_recurrent(gpu, &a, A, "a after b reset")?;
        verify_rows(gpu, &a, A, 0, la_now, "a rows after b reset")?;
        expect_seq(gpu, &a, la_now, "a after b reset")?;
        if a.mla_cache.iter().map(|t| t.buf.as_ptr() as usize).collect::<Vec<_>>() != ptrs_a {
            return Err("a latent cache address moved".into());
        }
        // Reuse b for a short new request and make sure nothing of the old
        // request is visible through the recurrent block, then scrub.
        b.scrub_latent(gpu)?;
        let cb2 = b.capacity_rows();
        check_const(gpu, &b.mla_cache[0].buf, 0, cb2 * dims.latent_row_bytes(), 0, "b scrub")?;
        write_rows(gpu, &b, B ^ 0x99, 0, 2)?;
        b.set_n_tokens(gpu, 2)?;
        verify_rows(gpu, &b, B ^ 0x99, 0, 2, "b new request")?;
        verify_recurrent_zero(gpu, &b, "b new request")?;
        verify_recurrent(gpu, &a, A, "a after b reuse")?;

        // Abort a mid-request too.
        a.reset(gpu)?;
        expect_seq(gpu, &a, 0, "a after reset")?;
        verify_recurrent_zero(gpu, &a, "a after reset")?;
        Ok(json!({
            "slot_a_len": la_now,
            "slot_b_len": lb_now,
            "capacity_rows_a": ca,
            "capacity_rows_b": cb,
            "interleaved_steps": steps,
            "backend": a.backend().as_str(),
        }))
    })();
    let fa = a.free_gpu(gpu);
    let fb = b.free_gpu(gpu);
    match (r, fa, fb) {
        (Ok(v), Ok(()), Ok(())) => Ok(v),
        (Err(e), _, _) => Err(e),
        (Ok(_), Err(e), _) | (Ok(_), _, Err(e)) => Err(e),
    }
}

fn check_ctx_reservation(
    gpu: &mut Gpu,
    dims: &RaonStateDims,
    ctx: usize,
    recurrent: &mut Vec<(usize, usize)>,
) -> Check {
    gpu.drain_pool();
    let free0 = free_bytes(gpu)?;
    let mut st = RaonState::new(gpu, dims, ctx)?;
    let r = (|| -> Check {
        let free1 = free_bytes(gpu)?;
        let ledger = st.bytes_ledger(gpu);
        let backend = st.backend();
        let gran = match backend {
            LatentBackend::Vmm => gpu.vmm_granularity(&st.mla_cache[0]).unwrap_or(0),
            LatentBackend::Dense => 1,
        };
        let budget = state_budget(dims, ctx, backend, gran.max(1))?;
        let initial = ledger.physical_total();
        if initial != budget.initial_physical(dims) {
            return Err(format!(
                "initial ledger {initial} B != budget {} B",
                budget.initial_physical(dims)
            ));
        }
        let delta_initial = check_delta(gpu, free0, free1, initial, "initial")?;
        let rda_conv = ledger.physical_with_prefix("rda_state") + ledger.physical_with_prefix("conv_history");
        recurrent.push((ctx, rda_conv));
        let latent_virtual: usize = (0..dims.n_mla)
            .map(|i| {
                ledger
                    .entries
                    .iter()
                    .find(|e| e.owner == format!("latent_cache.mla{i}"))
                    .map_or(0, |e| e.virtual_bytes)
            })
            .sum();
        if latent_virtual < ctx * dims.latent_row_bytes() * dims.n_mla {
            return Err(format!(
                "virtual reservation {latent_virtual} B < {} B needed for {ctx} tokens",
                ctx * dims.latent_row_bytes() * dims.n_mla
            ));
        }

        // Boundary growth with data integrity across the chunk edge.
        let cap0 = st.capacity_rows();
        let mut growth = json!({"grew": false});
        if backend == LatentBackend::Vmm && cap0 < ctx {
            let head = cap0.min(16);
            write_rows(gpu, &st, 0x51, 0, head)?;
            write_rows(gpu, &st, 0x51, cap0 - 1, 1)?;
            if st.ensure_capacity(gpu, cap0)? {
                return Err("ensure_capacity(cap) must not grow".into());
            }
            if !st.ensure_capacity(gpu, cap0 + 1)? {
                return Err("ensure_capacity(cap+1) must grow".into());
            }
            let cap1 = st.capacity_rows();
            if cap1 <= cap0 {
                return Err(format!("capacity did not increase: {cap0} -> {cap1}"));
            }
            verify_rows(gpu, &st, 0x51, 0, head, "head after first growth")?;
            verify_rows(gpu, &st, 0x51, cap0 - 1, 1, "boundary row after growth")?;
            write_rows(gpu, &st, 0x52, cap0, 1)?;
            verify_rows(gpu, &st, 0x52, cap0, 1, "first new row")?;
            st.ensure_capacity(gpu, ctx)?;
            if st.capacity_rows() != ctx {
                return Err(format!("capacity {} != ctx {ctx} after full growth", st.capacity_rows()));
            }
            write_rows(gpu, &st, 0x53, ctx - 1, 1)?;
            verify_rows(gpu, &st, 0x53, ctx - 1, 1, "last row")?;
            verify_rows(gpu, &st, 0x51, 0, head, "head after full growth")?;
            verify_rows(gpu, &st, 0x51, cap0 - 1, 1, "boundary row after full growth")?;
            verify_rows(gpu, &st, 0x52, cap0, 1, "first new row after full growth")?;
            if st.ensure_capacity(gpu, ctx + 1).is_ok() {
                return Err("ensure_capacity(ctx+1) must be refused".into());
            }
            let full_ledger = st.bytes_ledger(gpu);
            if full_ledger.physical_total() != budget.full_physical(dims) {
                return Err(format!(
                    "full ledger {} B != budget {} B",
                    full_ledger.physical_total(),
                    budget.full_physical(dims)
                ));
            }
            let free2 = free_bytes(gpu)?;
            let delta_full = check_delta(gpu, free0, free2, full_ledger.physical_total(), "full")?;
            growth = json!({
                "grew": true,
                "initial_capacity_rows": cap0,
                "after_first_growth_rows": cap1,
                "full_ledger_bytes": full_ledger.physical_total(),
                "delta_full": delta_full,
            });
        } else if !st.ensure_capacity(gpu, ctx)? && st.capacity_rows() < ctx {
            return Err("capacity short of ctx without growth".into());
        }
        Ok(json!({
            "ctx": ctx,
            "backend": backend.as_str(),
            "granularity": gran,
            "virtual_latent_bytes": latent_virtual,
            "initial_physical_bytes": initial,
            "rda_plus_conv_bytes": rda_conv,
            "delta_initial": delta_initial,
            "growth": growth,
            "ledger_initial": ledger.to_json(),
        }))
    })();
    let freed = st.free_gpu(gpu);
    gpu.drain_pool();
    match (r, freed) {
        (Ok(v), Ok(())) => Ok(v),
        (Err(e), _) => Err(e),
        (Ok(_), Err(e)) => Err(e),
    }
}

fn check_recurrent_constant(per_ctx: &[(usize, usize)]) -> Check {
    let Some(&(_, first)) = per_ctx.first() else {
        return Err("no ctx reservation produced a recurrent byte count".into());
    };
    let d = RaonStateDims::raon();
    let expected_rda = d.n_rda * d.rda_state_elems() * 4;
    if first < expected_rda {
        return Err(format!("recurrent bytes {first} < 45 MiB ({expected_rda})"));
    }
    for &(ctx, b) in per_ctx {
        if b != first {
            return Err(format!("recurrent bytes at ctx {ctx} = {b}, at ctx {} = {first}", per_ctx[0].0));
        }
    }
    Ok(json!({"rda_plus_conv_bytes": first, "rda_bytes": expected_rda, "ctxs": per_ctx.iter().map(|c| c.0).collect::<Vec<_>>()}))
}

fn check_admission_refusal(gpu: &mut Gpu, dims: &RaonStateDims) -> Check {
    let free0 = free_bytes(gpu)?;
    let row_all = dims.latent_row_bytes() * dims.n_mla;
    let ctx = (free0 / row_all).saturating_mul(2).saturating_add(4096);
    match RaonState::new(gpu, dims, ctx) {
        Ok(st) => {
            let _ = st.free_gpu(gpu);
            Err(format!("max_seq {ctx} ({} MiB latent) must be refused with {} MiB free", ctx * row_all / MIB, free0 / MIB))
        }
        Err(e) => {
            if !e.contains("refused") {
                return Err(format!("refusal must be a clean named refusal, got: {e}"));
            }
            let free1 = free_bytes(gpu)?;
            if free0.abs_diff(free1) > 16 * MIB {
                return Err(format!("refused admission changed free memory {free0} -> {free1}"));
            }
            Ok(json!({"max_seq": ctx, "message": e}))
        }
    }
}

fn check_admission_262144(gpu: &mut Gpu, dims: &RaonStateDims) -> Check {
    let ctx = 262_144;
    let backend = auto_backend(gpu);
    let gran = match backend {
        LatentBackend::Vmm => gpu.vmm_recommended_granularity().map_err(|e| format!("granularity: {e:?}"))?,
        LatentBackend::Dense => 1,
    };
    let budget = state_budget(dims, ctx, backend, gran)?;
    let need = budget.full_physical(dims) + ADMISSION_HEADROOM_BYTES;
    gpu.drain_pool();
    let free = free_bytes(gpu)?;
    let supported = need <= free;
    match RaonState::new(gpu, dims, ctx) {
        Ok(mut st) => {
            if !supported && gpu.arch != "gfx1151" {
                let _ = st.free_gpu(gpu);
                return Err(format!("262144 admitted although need {need} B > free {free} B"));
            }
            let r = (|| -> Check {
                st.ensure_capacity(gpu, ctx)?;
                if st.capacity_rows() != ctx {
                    return Err("full growth to 262144 incomplete".into());
                }
                write_rows(gpu, &st, 0x262, ctx - 1, 1)?;
                verify_rows(gpu, &st, 0x262, ctx - 1, 1, "row 262143")?;
                Ok(json!({"outcome": "admitted", "need_bytes": need, "free_bytes": free, "latent_mib": st.bytes_ledger(gpu).physical_with_prefix("latent_cache") / MIB}))
            })();
            let f = st.free_gpu(gpu);
            gpu.drain_pool();
            match (r, f) {
                (Ok(v), Ok(())) => Ok(v),
                (Err(e), _) => Err(e),
                (Ok(_), Err(e)) => Err(e),
            }
        }
        Err(e) => {
            if supported {
                return Err(format!("262144 refused although need {need} B <= free {free} B: {e}"));
            }
            if !e.contains("refused") {
                return Err(format!("262144 refusal is not a named refusal: {e}"));
            }
            Ok(json!({"outcome": "refused", "need_bytes": need, "free_bytes": free, "message": e}))
        }
    }
}

fn check_growth_pressure(gpu: &mut Gpu, dims: &RaonStateDims) -> Check {
    let ctx = 65_536;
    gpu.drain_pool();
    let mut st = RaonState::new(gpu, dims, ctx)?;
    if st.backend() != LatentBackend::Vmm {
        let f = st.free_gpu(gpu);
        f?;
        return Ok(json!({"skipped": "dense latent backend has no growth"}));
    }
    let r = (|| -> Check {
        let budget = state_budget(
            dims,
            ctx,
            LatentBackend::Vmm,
            gpu.vmm_granularity(&st.mla_cache[0]).unwrap_or(1),
        )?;
        let needed = budget.full_physical(dims) - st.bytes_ledger(gpu).physical_total();
        let free_now = free_bytes(gpu)?;
        let keep = needed / 2 + 64 * MIB;
        if free_now <= keep + 64 * MIB || gpu.arch == "gfx1151" {
            return Ok(json!({"skipped": "not enough free memory for a ballast test or UMA device", "free_bytes": free_now}));
        }
        let ballast = hip_err(gpu.alloc_tensor_exact(&[free_now - keep], DType::Raw), "alloc ballast")?;
        let outcome = (|| -> Check {
            let before = mapped_bytes(gpu, &st);
            let cap = st.capacity_rows();
            write_rows(gpu, &st, 0x77, 0, 4)?;
            match st.ensure_capacity(gpu, ctx) {
                Err(e) if e.contains("refused") => {}
                other => return Err(format!("growth under pressure must be refused, got {other:?}")),
            }
            if mapped_bytes(gpu, &st) != before || st.capacity_rows() != cap {
                return Err("refused growth mutated the mapped extent".into());
            }
            verify_rows(gpu, &st, 0x77, 0, 4, "rows after refused growth")?;
            st.reset(gpu)?;
            Ok(json!({"needed_bytes": needed, "free_with_ballast": free_bytes(gpu)?}))
        })();
        gpu.bind_thread().map_err(|e| format!("bind: {e:?}"))?;
        hip_err(gpu.hip.free(ballast.buf), "free ballast")?;
        let v = outcome?;
        if !st.ensure_capacity(gpu, ctx)? || st.capacity_rows() != ctx {
            return Err("growth must succeed after pressure is released".into());
        }
        verify_rows(gpu, &st, 0x77, 0, 4, "rows after retried growth")?;
        Ok(v)
    })();
    let f = st.free_gpu(gpu);
    gpu.drain_pool();
    match (r, f) {
        (Ok(v), Ok(())) => Ok(v),
        (Err(e), _) => Err(e),
        (Ok(_), Err(e)) => Err(e),
    }
}

fn check_scratch(gpu: &mut Gpu, dims: &RaonStateDims, rows: usize) -> Check {
    gpu.drain_pool();
    let plan = scratch_plan(dims, rows)?;
    let free0 = free_bytes(gpu)?;
    let sc = RaonPrefillScratch::new(gpu, dims, rows)?;
    let r = (|| -> Check {
        let free1 = free_bytes(gpu)?;
        if sc.bytes() != plan.total_bytes || sc.bytes() != RaonPrefillScratch::required_bytes(dims, rows)? {
            return Err("scratch bytes != plan bytes".into());
        }
        let delta = check_delta(gpu, free0, free1, sc.bytes(), "scratch")?;
        // Views follow the plan and start zeroed.
        let base = sc.token_ids.buf.as_ptr() as usize;
        for (name, ptr, len) in [
            ("positions", sc.positions.buf.as_ptr() as usize, sc.positions.buf.size()),
            ("stream_a", sc.stream_a.buf.as_ptr() as usize, sc.stream_a.buf.size()),
            ("stream_b", sc.stream_b.buf.as_ptr() as usize, sc.stream_b.buf.size()),
            ("mla_kv_tile", sc.mla_kv_tile.buf.as_ptr() as usize, sc.mla_kv_tile.buf.size()),
            ("logits", sc.logits.buf.as_ptr() as usize, sc.logits.buf.size()),
        ] {
            let e = plan.entry(name).ok_or_else(|| format!("plan lacks {name}"))?;
            if ptr != base + e.offset || len != e.bytes {
                return Err(format!("{name}: view does not match plan (ptr off {} vs {}, len {len} vs {})", ptr.wrapping_sub(base), e.offset, e.bytes));
            }
        }
        check_const(gpu, &sc.logits.buf, 0, sc.logits.buf.size(), 0, "logits zero-init")?;
        check_const(gpu, &sc.mla_kv_tile.buf, 0, sc.mla_kv_tile.buf.size(), 0, "mla_kv_tile zero-init")?;
        // A write to one buffer must not leak into a neighbor.
        fill_const(gpu, &sc.stream_a.buf, 0, sc.stream_a.buf.size(), 0xAB, "stream_a fill")?;
        check_const(gpu, &sc.stream_b.buf, 0, sc.stream_b.buf.size(), 0, "stream_b untouched")?;
        check_const(gpu, &sc.positions.buf, 0, sc.positions.buf.size(), 0, "positions untouched")?;
        Ok(json!({"rows": rows, "bytes": sc.bytes(), "mib": sc.bytes() / MIB, "delta": delta, "buffers": plan.entries.len()}))
    })();
    let f = sc.free_gpu(gpu);
    gpu.drain_pool();
    match (r, f) {
        (Ok(v), Ok(())) => Ok(v),
        (Err(e), _) => Err(e),
        (Ok(_), Err(e)) => Err(e),
    }
}

fn check_scratch_limit(gpu: &mut Gpu, dims: &RaonStateDims) -> Check {
    if RaonPrefillScratch::new(gpu, dims, 0).is_ok() {
        return Err("0-row scratch must be refused".into());
    }
    if RaonPrefillScratch::new(gpu, dims, 1 << 20).is_ok() {
        return Err("oversized scratch must be refused".into());
    }
    Ok(json!({"zero_rows": "refused", "oversized": "refused"}))
}

fn check_mirror(gpu: &mut Gpu, dims: &RaonStateDims) -> Check {
    gpu.drain_pool();
    let mut st = RaonState::new(gpu, dims, 16_384)?;
    let r = (|| -> Check {
        if st.admit_expanded_mirror(gpu, EXPANDED_MIRROR_MAX_ROWS + 1).is_ok() {
            return Err(format!("mirror above {EXPANDED_MIRROR_MAX_ROWS} rows must be refused"));
        }
        if st.admit_expanded_mirror(gpu, 0).is_ok() {
            return Err("0-row mirror must be refused".into());
        }
        let rows = 2048;
        let bytes = expanded_mirror_bytes(dims, rows)?;
        let before = st.bytes_ledger(gpu).physical_total();
        st.admit_expanded_mirror(gpu, rows)?;
        let after = st.bytes_ledger(gpu);
        let mirror = after.physical_with_prefix("expanded_mirror");
        if mirror != bytes || after.physical_total() != before + bytes {
            return Err(format!("mirror ledger {mirror} B != {bytes} B"));
        }
        {
            let m = st.expanded_mirror().ok_or("mirror missing after admit")?;
            if m.knope.len() != dims.n_mla || m.vmir.len() != dims.n_mla || m.rows != rows {
                return Err("mirror view count/rows mismatch".into());
            }
            if m.knope[0].shape != vec![dims.heads, rows, 128] || m.vmir[0].shape != vec![dims.heads, rows, 128] {
                return Err("mirror views are not head-major [heads, rows, 128]".into());
            }
            write_pat(gpu, &m.knope[3].buf, 0, m.knope[3].buf.size(), 0x3C, 0, "mirror knope write")?;
            check_const(gpu, &m.vmir[3].buf, 0, m.vmir[3].buf.size(), 0, "mirror vmir untouched")?;
            check_const(gpu, &m.knope[4].buf, 0, m.knope[4].buf.size(), 0, "mirror next-layer knope untouched")?;
        }
        st.release_expanded_mirror(gpu)?;
        if st.expanded_mirror().is_some() || st.bytes_ledger(gpu).physical_total() != before {
            return Err("mirror release did not return the ledger".into());
        }
        Ok(json!({"rows": rows, "bytes": bytes, "per_token_bytes": bytes / rows}))
    })();
    let f = st.free_gpu(gpu);
    gpu.drain_pool();
    match (r, f) {
        (Ok(v), Ok(())) => Ok(v),
        (Err(e), _) => Err(e),
        (Ok(_), Err(e)) => Err(e),
    }
}

fn check_vmm_clean(gpu: &mut Gpu) -> Check {
    hip_err(gpu.ensure_vmm_cleaned(), "ensure_vmm_cleaned")?;
    let n = gpu.vmm_allocation_count();
    if n != 0 {
        return Err(format!("{n} VMM owner(s) still registered"));
    }
    Ok(json!({"vmm_allocation_count": n}))
}
