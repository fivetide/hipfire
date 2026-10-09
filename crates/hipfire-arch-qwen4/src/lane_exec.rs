// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! The §4.2 step executor over the exact AR lanes
//! ([`crate::lane::Qwen4LaneStore`]).
//!
//! A plan carries `Ar` requests (one row: the lane's pending seed at the
//! lane's position) and `Prefill` requests (one singleton prefill tile of
//! the lane's prompt, [`Qwen4Bundle::lane_prefill_tile_len`] rows).
//! `Verify` and `Forced` rows are refused. Per step:
//!
//! 1. [`Qwen4LaneExecutor::provision_step`] validates the plan against the
//!    lane table and maps every lane's QSA arenas up to the step's end
//!    position.
//! 2. [`Qwen4LaneExecutor::forward_step`] runs the AR rows of all lanes as
//!    one exact trunk step (`forward_lanes`), then each prefill tile through
//!    the singleton chunk program on the lane's own state, and downloads the
//!    greedy picks once.
//! 3. [`Qwen4LaneExecutor::commit_step`] applies the lane bookkeeping. The
//!    forwards publish the device and host trunk state (QSA lengths, PLE
//!    history, `state.position`) themselves, exactly as the singleton route
//!    does; commit never touches device state.
//!
//! Invariants of the lane plan (PLAN §3.2) that apply to AR lanes, and how
//! each is enforced:
//!
//! - I1 (`state.position == mtp.position`): AR lanes carry no MTP state
//!   (`mtp` is `None`); the position every row is checked against is the
//!   lane's `state.position`.
//! - I2 (one snapshot ticket, only in a verify): no lane step snapshots; no
//!   verify row is admitted.
//! - I3 (`row_capture_armed` only during a verify forward): AR lanes arm no
//!   ring (`ring_rows == 0`) and nothing in a lane step sets the flag.
//! - I4 (mapping only at provision): `ensure_mapped_capacity` is called
//!   in `provision_step` and nowhere else; the forwards run on mapped
//!   arenas.
//! - I5 (one plan identity across provision, forward, commit): the
//!   [`StepKey`] recorded at provision must equal the plan at forward and at
//!   commit, and commit also needs the step id forward issued. A failed
//!   forward poisons its participants and returns to idle.
//! - I6 (a lane step never touches `bundle.state`/`bundle.mtp`):
//!   `forward_lanes` works on the lanes' own states; a prefill tile swaps
//!   the lane's state in for the call only.
//! - I7 (rings are never shared): none exist on AR lanes.
//! - I8 (session bookkeeping keyed by epoch): lanes keep no session cache in
//!   this route; every lookup here is by the full epoch (tag and generation).
//! - I9 (arenas cover the step's end position before the first launch):
//!   `provision_step` maps `state.position + rows` for every request and
//!   refuses ends past `state.max_seq_len`.

use crate::bundle::Qwen4Bundle;
use crate::gpu_forward_lanes::{forward_lanes, LaneOutputs, LaneRowKind, LaneRows};
use crate::lane::{LaneHost, LaneStatus, Qwen4LaneState, Qwen4LaneStore, StepParts};
use hipfire_runtime::slot_batch::{
    BatchStepPlan, RequestAdvance, RequestEpoch, RequestRows, RequestStepKind, StepOutput,
};
use rdna_compute::tensor_ops::{argmax_f32, ArgmaxF32};
use rdna_compute::Gpu;

/// Plan identity recorded by provision/forward and checked by commit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct StepKey {
    rows: Vec<(RequestEpoch, usize, usize, RequestStepKind)>,
    tokens: Vec<u32>,
    positions: Vec<i32>,
}

impl StepKey {
    fn of(plan: &BatchStepPlan) -> Self {
        Self {
            rows: plan
                .requests
                .iter()
                .map(|r| (r.epoch, r.rows.begin, r.rows.len, r.kind))
                .collect(),
            tokens: plan.batch.tokens.clone(),
            positions: plan.batch.positions.clone(),
        }
    }
}

/// Where the store's step is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LanePhase {
    Idle,
    Provisioned(StepKey),
    Forwarded(StepKey, u64),
}

impl LanePhase {
    fn check_idle(&self) -> Result<(), String> {
        match self {
            Self::Idle => Ok(()),
            _ => Err("provision_step: previous step not committed or aborted".to_string()),
        }
    }

    fn check_forward(&self, key: &StepKey) -> Result<(), String> {
        match self {
            Self::Provisioned(provisioned) if provisioned == key => Ok(()),
            _ => Err("forward_step: plan was not provisioned".to_string()),
        }
    }

    fn check_commit(&self, key: &StepKey, step_id: u64) -> Result<(), String> {
        match self {
            Self::Forwarded(forwarded, id) if forwarded == key && *id == step_id => Ok(()),
            _ => Err("commit_step: output does not belong to the forwarded plan".to_string()),
        }
    }
}

/// What plan validation reads of one lane.
struct LaneView<'a> {
    slot: usize,
    host: &'a LaneHost,
    /// `state.position`: the next row's absolute position.
    position: usize,
    /// `state.max_seq_len`.
    max_seq_len: usize,
}

fn view_of(lane: &Qwen4LaneState) -> LaneView<'_> {
    LaneView {
        slot: lane.host.slot,
        host: &lane.host,
        position: lane.state.position,
        max_seq_len: lane.state.max_seq_len,
    }
}

#[derive(Clone, Copy)]
struct PlanLimits {
    max_lanes: usize,
    row_budget: usize,
}

/// Every host-only check of `provision_step`. `view` finds a live lane by
/// epoch; `prefill_len` is the singleton tile length at the lane's
/// `prefilled`. Returns each request's end position (`state.position +
/// rows`), in plan order.
fn validate_plan<'a>(
    plan: &BatchStepPlan,
    limits: PlanLimits,
    view: impl Fn(&RequestEpoch) -> Option<LaneView<'a>>,
    prefill_len: impl Fn(&LaneHost) -> usize,
) -> Result<Vec<usize>, String> {
    let b = &plan.batch;
    let n = b.total_rows();
    if n == 0 || plan.requests.is_empty() {
        return Err("provision_step: empty plan".to_string());
    }
    if b.m_per_slot.len() > limits.max_lanes
        || b.tokens.len() != n
        || b.positions.len() != n
        || b.row_slot.len() != n
    {
        return Err("provision_step: malformed SlotBatch".to_string());
    }
    if !b.pos3.is_empty() || b.ext_emb.iter().any(|&e| e >= 0) {
        return Err("provision_step: VL rows are not admitted on AR lanes".to_string());
    }
    let mut covered = vec![false; n];
    let mut seen_slots = vec![false; limits.max_lanes];
    let mut ar_rows = 0usize;
    let mut completing = 0usize;
    let mut ends = Vec::with_capacity(plan.requests.len());
    for r in &plan.requests {
        if matches!(r.kind, RequestStepKind::Verify { .. } | RequestStepKind::Forced) {
            return Err(format!(
                "provision_step: {:?} rows are not admitted on AR lanes",
                r.kind
            ));
        }
        let lane = view(&r.epoch)
            .ok_or_else(|| format!("provision_step: stale or unknown epoch {:?}", r.epoch))?;
        let host = lane.host;
        if host.status == LaneStatus::Poisoned {
            return Err(format!("provision_step: epoch {:?} is poisoned", r.epoch));
        }
        if lane.slot >= limits.max_lanes {
            return Err(format!("provision_step: slot {} out of range", lane.slot));
        }
        if std::mem::replace(&mut seen_slots[lane.slot], true) {
            return Err(format!("provision_step: slot {} planned twice", lane.slot));
        }
        if r.rows.len == 0 || r.rows.end() > n {
            return Err(format!("provision_step: bad row range {:?}", r.rows));
        }
        if b.m_per_slot.get(lane.slot).copied() != Some(r.rows.len) {
            return Err(format!(
                "provision_step: m_per_slot[{}] disagrees with row range len {}",
                lane.slot, r.rows.len
            ));
        }
        let tokens = &b.tokens[r.rows.begin..r.rows.end()];
        match r.kind {
            RequestStepKind::Ar => {
                if r.rows.len != 1 {
                    return Err("provision_step: AR request must contribute one row".to_string());
                }
                if host.status != LaneStatus::Decoding {
                    return Err(format!(
                        "provision_step: AR row for {:?} lane {:?}",
                        host.status, r.epoch
                    ));
                }
                let seed = host.pending_seed.ok_or_else(|| {
                    format!("provision_step: epoch {:?} has no pending seed", r.epoch)
                })?;
                if tokens[0] != seed {
                    return Err(format!(
                        "provision_step: AR token {} != pending seed {seed} for {:?}",
                        tokens[0], r.epoch
                    ));
                }
                ar_rows += 1;
            }
            RequestStepKind::Prefill => {
                if host.status != LaneStatus::Prefilling {
                    return Err(format!(
                        "provision_step: prefill rows for {:?} lane {:?}",
                        host.status, r.epoch
                    ));
                }
                if lane.position != host.prefilled {
                    return Err(format!(
                        "provision_step: lane {:?} state position {} != prefilled {}",
                        r.epoch, lane.position, host.prefilled
                    ));
                }
                let want = prefill_len(host);
                if want == 0 || host.prefilled + want > host.prompt.len() {
                    return Err(format!(
                        "provision_step: no prefill tile left for {:?} ({} of {})",
                        r.epoch,
                        host.prefilled,
                        host.prompt.len()
                    ));
                }
                if r.rows.len != want {
                    return Err(format!(
                        "provision_step: prefill tile for {:?} is {} rows, the singleton route uses {want}",
                        r.epoch, r.rows.len
                    ));
                }
                if tokens != &host.prompt[host.prefilled..host.prefilled + want] {
                    return Err(format!(
                        "provision_step: prefill tokens for {:?} differ from its prompt",
                        r.epoch
                    ));
                }
                if host.prefilled + want == host.prompt.len() {
                    completing += 1;
                }
            }
            RequestStepKind::Verify { .. } | RequestStepKind::Forced => unreachable!(),
        }
        for (j, row) in (r.rows.begin..r.rows.end()).enumerate() {
            if std::mem::replace(&mut covered[row], true) {
                return Err(format!("provision_step: row {row} in two requests"));
            }
            if b.row_slot[row] != lane.slot as i32 {
                return Err(format!(
                    "provision_step: row {row} row_slot {} != owner slot {}",
                    b.row_slot[row], lane.slot
                ));
            }
            if i64::from(b.positions[row]) != (lane.position + j) as i64 {
                return Err(format!(
                    "provision_step: row {row} position {} != committed frontier {}",
                    b.positions[row],
                    lane.position + j
                ));
            }
        }
        let end = lane.position + r.rows.len;
        if end > lane.max_seq_len {
            return Err(format!(
                "provision_step: epoch {:?} needs {end} positions > context {}",
                r.epoch, lane.max_seq_len
            ));
        }
        ends.push(end);
    }
    if covered.iter().any(|c| !c) {
        return Err("provision_step: rows outside every request range".to_string());
    }
    if ar_rows > limits.row_budget {
        return Err(format!(
            "provision_step: {ar_rows} AR rows exceed row budget {}",
            limits.row_budget
        ));
    }
    if completing > limits.max_lanes {
        return Err(format!(
            "provision_step: {completing} prompt-completing tiles exceed {} lanes",
            limits.max_lanes
        ));
    }
    Ok(ends)
}

/// Whether `r` ends in a greedy pick: an AR row, or the tile that completes
/// the prompt.
fn wants_pick(host: &LaneHost, r: &RequestRows) -> bool {
    match r.kind {
        RequestStepKind::Ar => true,
        RequestStepKind::Prefill => host.prefilled + r.rows.len == host.prompt.len(),
        RequestStepKind::Verify { .. } | RequestStepKind::Forced => false,
    }
}

/// Lane bookkeeping of one committed request; returns the committed id.
fn commit_host(host: &mut LaneHost, r: &RequestRows, pick: u32) -> Option<u32> {
    let picks = wants_pick(host, r);
    if r.kind == RequestStepKind::Prefill {
        host.prefilled += r.rows.len;
    }
    if !picks {
        return None;
    }
    if r.kind == RequestStepKind::Prefill {
        host.status = LaneStatus::Decoding;
    }
    host.generated.push(pick);
    host.pending_seed = Some(pick);
    Some(pick)
}

/// The lane executor: borrows the bundle (weights, forward program) and its
/// staged [`Qwen4LaneStore`].
pub struct Qwen4LaneExecutor<'a> {
    pub(crate) bundle: &'a mut Qwen4Bundle,
}

impl Qwen4LaneExecutor<'_> {
    fn store(&self) -> Result<&Qwen4LaneStore, String> {
        self.bundle
            .lanes
            .as_ref()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())
    }

    /// The singleton tile length the lane's next prefill step must use.
    pub fn exact_prefill_chunk_len(&self, epoch: &RequestEpoch) -> Result<usize, String> {
        let lane = self
            .store()?
            .lane(epoch)
            .ok_or_else(|| format!("exact_prefill_chunk_len: unknown or stale epoch {epoch:?}"))?;
        if lane.host.status != LaneStatus::Prefilling {
            return Err(format!(
                "exact_prefill_chunk_len: lane {epoch:?} is {:?}, not Prefilling",
                lane.host.status
            ));
        }
        Ok(self
            .bundle
            .lane_prefill_tile_len(&lane.host.prompt, lane.host.prefilled))
    }

    /// Validate the plan against the lane table and map every lane's QSA
    /// arenas through the step's end position. Mapping happens here, never
    /// inside a forward (I4).
    pub fn provision_step(&mut self, gpu: &mut Gpu, plan: &BatchStepPlan) -> Result<(), String> {
        let ends = {
            let bundle: &Qwen4Bundle = &*self.bundle;
            let store = bundle
                .lanes
                .as_ref()
                .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
            store.phase.check_idle()?;
            validate_plan(
                plan,
                PlanLimits {
                    max_lanes: store.max_lanes(),
                    row_budget: store.row_budget(),
                },
                |epoch| store.lane(epoch).map(view_of),
                |host| bundle.lane_prefill_tile_len(&host.prompt, host.prefilled),
            )?
        };
        let store = self
            .bundle
            .lanes
            .as_mut()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
        for (r, &end) in plan.requests.iter().zip(&ends) {
            let lane = store
                .lane_mut(&r.epoch)
                .ok_or_else(|| format!("provision_step: epoch {:?} vanished", r.epoch))?;
            lane.state
                .ensure_mapped_capacity(gpu, end)
                .map_err(|e| format!("provision_step: map {end} positions for {:?}: {e}", r.epoch))?;
        }
        store.phase = LanePhase::Provisioned(StepKey::of(plan));
        Ok(())
    }

    /// The step's trunk forward(s) and one pick per AR / prompt-completing
    /// row. AR rows of all lanes run as one exact trunk step, then each
    /// prefill tile runs the singleton chunk program on its own lane state:
    /// a decode row is never co-batched with a prefill tile. Waits for the
    /// picks (the download synchronizes). On error every participant is
    /// poisoned and the executor returns to idle.
    pub fn forward_step(&mut self, gpu: &mut Gpu, plan: &BatchStepPlan) -> Result<StepOutput, String> {
        let key = StepKey::of(plan);
        let store = self
            .bundle
            .lanes
            .as_mut()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
        store.phase.check_forward(&key)?;
        let step_id = store.next_step_id;
        store.phase = LanePhase::Forwarded(key, step_id);
        let mut store = self.bundle.lanes.take().expect("store checked above");
        let result = forward_inner(self.bundle, gpu, &mut store, plan);
        let output = match result {
            Ok(target_picks) => {
                store.next_step_id += 1;
                Ok(StepOutput {
                    step_id,
                    target_picks,
                })
            }
            Err(error) => {
                for r in &plan.requests {
                    store.poison(&r.epoch);
                }
                store.phase = LanePhase::Idle;
                Err(format!("forward_step: {error}"))
            }
        };
        self.bundle.lanes = Some(store);
        output
    }

    /// Apply the forwarded step's lane bookkeeping. Validates the plan
    /// identity, step id, pick count and every participant before mutating
    /// anything.
    pub fn commit_step(
        &mut self,
        _gpu: &mut Gpu,
        plan: &BatchStepPlan,
        output: StepOutput,
    ) -> Result<Vec<RequestAdvance>, String> {
        let store = self
            .bundle
            .lanes
            .as_mut()
            .ok_or_else(|| "Qwen4 lanes are not staged".to_string())?;
        store.phase.check_commit(&StepKey::of(plan), output.step_id)?;
        if output.target_picks.len() != plan.total_rows() {
            return Err("commit_step: pick count != planned rows".to_string());
        }
        for r in &plan.requests {
            let lane = match store.lane(&r.epoch) {
                Some(lane) if lane.host.status != LaneStatus::Poisoned => lane,
                _ => {
                    return Err(format!("commit_step: epoch {:?} no longer owns its lane", r.epoch));
                }
            };
            if r.rows.len == 0 || r.rows.end() > output.target_picks.len() {
                return Err(format!("commit_step: bad row range {:?}", r.rows));
            }
            if wants_pick(&lane.host, r) && output.target_picks[r.rows.end() - 1] == u32::MAX {
                return Err("commit_step: head row without a target pick".to_string());
            }
        }
        let mut advances = Vec::with_capacity(plan.requests.len());
        for r in &plan.requests {
            let pick = output.target_picks[r.rows.end() - 1];
            let lane = store.lane_mut(&r.epoch).expect("validated above");
            let committed = commit_host(&mut lane.host, r, pick);
            let position = lane.state.position;
            advances.push(RequestAdvance {
                epoch: r.epoch,
                committed_ids: committed.into_iter().collect(),
                committed_position: position,
                accepted_drafts: 0,
                verified_rows: 0,
                finish: (position >= lane.state.max_seq_len).then(|| "length".to_string()),
            });
        }
        store.phase = LanePhase::Idle;
        Ok(advances)
    }

    /// Drop the step: every participant is poisoned (its device state may be
    /// half-advanced) and the executor returns to idle.
    pub fn abort_step(&mut self, plan: &BatchStepPlan) {
        if let Some(store) = self.bundle.lanes.as_mut() {
            for r in &plan.requests {
                store.poison(&r.epoch);
            }
            store.phase = LanePhase::Idle;
        }
    }
}

/// The forward of a provisioned plan; the store is out of the bundle so the
/// lanes' states can be borrowed next to it.
fn forward_inner(
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    store: &mut Qwen4LaneStore,
    plan: &BatchStepPlan,
) -> Result<Vec<u32>, String> {
    let vocab = bundle.lane_vocab();
    let max_lanes = store.max_lanes();
    let epochs: Vec<RequestEpoch> = plan.requests.iter().map(|r| r.epoch).collect();
    let StepParts {
        lanes,
        logits,
        top1,
        policy,
        top1_host,
    } = store.step_parts(&epochs)?;

    let mut ar: Vec<LaneRows<'_>> = Vec::new();
    let mut ar_begin: Vec<usize> = Vec::new();
    let mut prefill: Vec<(&RequestRows, &mut Qwen4LaneState)> = Vec::new();
    for (r, lane) in plan.requests.iter().zip(lanes) {
        match r.kind {
            RequestStepKind::Ar => {
                ar_begin.push(r.rows.begin);
                ar.push(LaneRows {
                    state: &mut lane.state,
                    tokens: &plan.batch.tokens[r.rows.begin..r.rows.end()],
                    kind: LaneRowKind::Ar,
                });
            }
            RequestStepKind::Prefill => prefill.push((r, lane)),
            RequestStepKind::Verify { .. } | RequestStepKind::Forced => {
                return Err(format!("{:?} rows are not admitted on AR lanes", r.kind));
            }
        }
    }
    let n_ar = ar.len();
    if n_ar > 0 {
        let out = LaneOutputs {
            logits: &logits.sub_offset(0, n_ar * vocab),
            top1: &top1.sub_offset(0, n_ar * std::mem::size_of::<i32>()),
        };
        forward_lanes(gpu, bundle, &mut ar, &out, &policy).map_err(|e| e.to_string())?;
    }
    drop(ar);

    let n_prefill = prefill.len();
    // (prefill ordinal, plan row of its pick)
    let mut final_tiles: Vec<(usize, usize)> = Vec::new();
    for (i, (r, lane)) in prefill.into_iter().enumerate() {
        let tokens = &plan.batch.tokens[r.rows.begin..r.rows.end()];
        if wants_pick(&lane.host, r) {
            let row = logits.sub_offset((max_lanes + i) * vocab, vocab);
            bundle
                .lane_prefill_tile(gpu, &mut lane.state, tokens, Some(&row))
                .map_err(|e| e.to_string())?;
            let slot = top1.sub_offset(
                (max_lanes + i) * std::mem::size_of::<i32>(),
                std::mem::size_of::<i32>(),
            );
            argmax_f32(
                gpu,
                &ArgmaxF32 {
                    logits: &row,
                    indices: &slot,
                    rows: 1,
                    vocab,
                },
            )
            .map_err(|e| e.to_string())?;
            final_tiles.push((i, r.rows.end() - 1));
        } else {
            bundle
                .lane_prefill_tile(gpu, &mut lane.state, tokens, None)
                .map_err(|e| e.to_string())?;
        }
    }

    let used_rows = if final_tiles.is_empty() {
        n_ar
    } else {
        max_lanes + n_prefill
    };
    let used = used_rows * std::mem::size_of::<i32>();
    if used > 0 {
        gpu.hip
            .memcpy_dtoh(&mut top1_host[..used], &top1.buf)
            .map_err(|e| e.to_string())?;
    } else {
        gpu.hip.device_synchronize().map_err(|e| e.to_string())?;
    }
    let pick_at = |row: usize| {
        let b = &top1_host[row * 4..row * 4 + 4];
        u32::from_ne_bytes([b[0], b[1], b[2], b[3]])
    };
    let mut picks = vec![u32::MAX; plan.total_rows()];
    for (k, &begin) in ar_begin.iter().enumerate() {
        picks[begin] = pick_at(k);
    }
    for &(i, row) in &final_tiles {
        picks[row] = pick_at(max_lanes + i);
    }
    Ok(picks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hipfire_runtime::slot_batch::RowRange;
    use std::collections::HashMap;

    const MAX_LANES: usize = 4;
    const ROW_BUDGET: usize = 2;
    const CONTEXT: usize = 64;
    const TILE: usize = 4;

    fn epoch(tag: u64) -> RequestEpoch {
        RequestEpoch {
            request_tag: tag,
            owner_generation: 1,
        }
    }

    struct Lane {
        host: LaneHost,
        position: usize,
    }

    fn host(tag: u64, slot: usize, status: LaneStatus, prompt_len: usize, prefilled: usize) -> LaneHost {
        LaneHost {
            epoch: epoch(tag),
            slot,
            prompt: (0..prompt_len as u32).map(|t| 100 + t).collect(),
            prefilled,
            generated: Vec::new(),
            pending_seed: (status == LaneStatus::Decoding).then_some(77),
            status,
            stamp: tag,
        }
    }

    /// Lane 1 decodes (seed 77 at position 10), lane 2 decodes (seed 77 at
    /// position 20), lane 3 prefills tile 2 of a 10-token prompt.
    fn lanes() -> HashMap<RequestEpoch, Lane> {
        let mut lanes = HashMap::new();
        for (tag, slot, status, prompt, prefilled, position) in [
            (1, 0, LaneStatus::Decoding, 10, 10, 10),
            (2, 1, LaneStatus::Decoding, 20, 20, 20),
            (3, 2, LaneStatus::Prefilling, 10, 4, 4),
        ] {
            lanes.insert(
                epoch(tag),
                Lane {
                    host: host(tag, slot, status, prompt, prefilled),
                    position,
                },
            );
        }
        lanes
    }

    struct Entry {
        tag: u64,
        slot: usize,
        kind: RequestStepKind,
        tokens: Vec<u32>,
        start: usize,
    }

    fn ar(tag: u64, slot: usize, token: u32, position: usize) -> Entry {
        Entry {
            tag,
            slot,
            kind: RequestStepKind::Ar,
            tokens: vec![token],
            start: position,
        }
    }

    fn tile(tag: u64, slot: usize, tokens: &[u32], start: usize) -> Entry {
        Entry {
            tag,
            slot,
            kind: RequestStepKind::Prefill,
            tokens: tokens.to_vec(),
            start,
        }
    }

    fn plan_of(entries: &[Entry]) -> BatchStepPlan {
        let mut plan = BatchStepPlan::default();
        plan.batch.m_per_slot = vec![0; MAX_LANES];
        for e in entries {
            let begin = plan.batch.tokens.len();
            plan.batch.m_per_slot[e.slot] = e.tokens.len();
            for (j, &t) in e.tokens.iter().enumerate() {
                plan.batch.tokens.push(t);
                plan.batch.positions.push((e.start + j) as i32);
                plan.batch.row_slot.push(e.slot as i32);
            }
            plan.requests.push(RequestRows {
                epoch: epoch(e.tag),
                rows: RowRange {
                    begin,
                    len: e.tokens.len(),
                },
                kind: e.kind,
            });
        }
        plan
    }

    fn validate(plan: &BatchStepPlan, lanes: &HashMap<RequestEpoch, Lane>) -> Result<Vec<usize>, String> {
        validate_plan(
            plan,
            PlanLimits {
                max_lanes: MAX_LANES,
                row_budget: ROW_BUDGET,
            },
            |e| {
                lanes.get(e).map(|l| LaneView {
                    slot: l.host.slot,
                    host: &l.host,
                    position: l.position,
                    max_seq_len: CONTEXT,
                })
            },
            |h| TILE.min(h.prompt.len() - h.prefilled),
        )
    }

    /// Prompt tokens of lane 3's second tile (prompt 100.., prefilled 4).
    fn tile3() -> Vec<u32> {
        vec![104, 105, 106, 107]
    }

    fn valid_entries() -> Vec<Entry> {
        vec![ar(1, 0, 77, 10), ar(2, 1, 77, 20), tile(3, 2, &tile3(), 4)]
    }

    fn refused(entries: &[Entry], needle: &str) {
        let error = validate(&plan_of(entries), &lanes()).unwrap_err();
        assert!(error.contains(needle), "{error:?} lacks {needle:?}");
    }

    #[test]
    fn a_mixed_ar_and_prefill_plan_is_accepted() {
        let plan = plan_of(&valid_entries());
        assert_eq!(validate(&plan, &lanes()), Ok(vec![11, 21, 8]));
    }

    #[test]
    fn ar_rows_are_exactly_one_token_at_the_lane_frontier() {
        let wide = Entry {
            tag: 1,
            slot: 0,
            kind: RequestStepKind::Ar,
            tokens: vec![77, 78],
            start: 10,
        };
        refused(&[wide], "one row");
        refused(&[ar(1, 0, 78, 10)], "pending seed");
        refused(&[ar(1, 0, 77, 11)], "position");
        refused(&[ar(1, 1, 77, 10)], "m_per_slot");
        // Wrong row_slot with a consistent m_per_slot.
        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.batch.row_slot[0] = 1;
        let error = validate(&plan, &lanes()).unwrap_err();
        assert!(error.contains("row_slot"), "{error}");
        // An AR row for a lane still prefilling.
        refused(&[ar(3, 2, 77, 4)], "Prefilling");
    }

    #[test]
    fn prefill_tiles_follow_the_singleton_schedule_and_prompt() {
        refused(&[tile(3, 2, &tile3()[..3], 4)], "singleton route uses 4");
        refused(&[tile(3, 2, &[104, 105, 106, 999], 4)], "differ from its prompt");
        refused(&[tile(3, 2, &tile3(), 5)], "position");
        // A prefill row for a decoding lane.
        refused(&[tile(1, 0, &[1, 2, 3, 4], 10)], "Decoding");
    }

    #[test]
    fn verify_and_forced_rows_are_not_admitted() {
        for kind in [RequestStepKind::Verify { draft_len: 0 }, RequestStepKind::Forced] {
            let mut entry = ar(1, 0, 77, 10);
            entry.kind = kind;
            refused(&[entry], "not admitted on AR lanes");
        }
    }

    #[test]
    fn overlapping_and_uncovered_rows_are_refused() {
        let mut plan = plan_of(&[ar(1, 0, 77, 10), ar(2, 1, 77, 20)]);
        plan.requests[1].rows = RowRange { begin: 0, len: 1 };
        let error = validate(&plan, &lanes()).unwrap_err();
        assert!(error.contains("two requests"), "{error}");

        let mut plan = plan_of(&[ar(1, 0, 77, 10), ar(2, 1, 77, 20)]);
        plan.requests.pop();
        plan.batch.m_per_slot[1] = 0;
        let error = validate(&plan, &lanes()).unwrap_err();
        assert!(error.contains("outside every request"), "{error}");

        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.requests[0].rows = RowRange { begin: 1, len: 1 };
        assert!(validate(&plan, &lanes()).is_err());

        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.requests.push(plan.requests[0]);
        assert!(validate(&plan, &lanes()).unwrap_err().contains("planned twice"));
    }

    #[test]
    fn unknown_stale_and_poisoned_epochs_are_refused() {
        refused(&[ar(9, 0, 77, 10)], "stale or unknown");
        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.requests[0].epoch.owner_generation = 2;
        assert!(validate(&plan, &lanes()).unwrap_err().contains("stale or unknown"));
        let mut table = lanes();
        table.get_mut(&epoch(1)).unwrap().host.status = LaneStatus::Poisoned;
        let error = validate(&plan_of(&[ar(1, 0, 77, 10)]), &table).unwrap_err();
        assert!(error.contains("poisoned"), "{error}");
    }

    #[test]
    fn the_row_budget_limits_ar_rows_only() {
        let mut table = lanes();
        for (tag, slot) in [(4u64, 3usize)] {
            table.insert(
                epoch(tag),
                Lane {
                    host: host(tag, slot, LaneStatus::Decoding, 5, 5),
                    position: 5,
                },
            );
        }
        let three = [ar(1, 0, 77, 10), ar(2, 1, 77, 20), ar(4, 3, 77, 5)];
        let error = validate(&plan_of(&three), &table).unwrap_err();
        assert!(error.contains("row budget"), "{error}");
        // The prefill tile does not count against it.
        assert!(validate(&plan_of(&valid_entries()), &table).is_ok());
    }

    #[test]
    fn rows_past_the_context_are_refused() {
        let mut table = lanes();
        table.get_mut(&epoch(1)).unwrap().position = CONTEXT;
        let error = validate(&plan_of(&[ar(1, 0, 77, CONTEXT)]), &table).unwrap_err();
        assert!(error.contains("context"), "{error}");
    }

    #[test]
    fn malformed_batches_are_refused() {
        assert!(validate(&BatchStepPlan::default(), &lanes())
            .unwrap_err()
            .contains("empty plan"));
        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.batch.positions.clear();
        assert!(validate(&plan, &lanes()).unwrap_err().contains("malformed"));
        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.batch.ext_emb = vec![0];
        assert!(validate(&plan, &lanes()).is_err());
        let mut plan = plan_of(&[ar(1, 0, 77, 10)]);
        plan.batch.pos3 = vec![[10, 10, 10]];
        assert!(validate(&plan, &lanes()).is_err());
    }

    #[test]
    fn the_phase_machine_pins_forward_and_commit_to_the_provisioned_plan() {
        let plan = plan_of(&valid_entries());
        let other = plan_of(&[ar(1, 0, 77, 10)]);
        let (key, other_key) = (StepKey::of(&plan), StepKey::of(&other));
        assert_ne!(key, other_key);

        // Forward before provision is refused.
        assert!(LanePhase::Idle.check_forward(&key).is_err());
        // A provisioned phase forwards only its own plan.
        let provisioned = LanePhase::Provisioned(key.clone());
        assert!(provisioned.check_forward(&key).is_ok());
        assert!(provisioned.check_forward(&other_key).is_err());
        assert!(provisioned.check_idle().is_err());
        assert!(LanePhase::Idle.check_idle().is_ok());
        // Commit needs the forwarded phase, the same plan and the step id.
        assert!(provisioned.check_commit(&key, 0).is_err());
        let forwarded = LanePhase::Forwarded(key.clone(), 7);
        assert!(forwarded.check_commit(&key, 7).is_ok());
        assert!(forwarded.check_commit(&key, 8).is_err());
        assert!(forwarded.check_commit(&other_key, 7).is_err());
        assert!(LanePhase::Idle.check_commit(&key, 7).is_err());
        // A changed token or position changes the key.
        let mut moved = plan.clone();
        moved.batch.positions[0] += 1;
        assert_ne!(StepKey::of(&moved), key);
        let mut retokened = plan.clone();
        retokened.batch.tokens[0] += 1;
        assert_ne!(StepKey::of(&retokened), key);
    }

    #[test]
    fn commit_bookkeeping_advances_decode_and_prefill_lanes() {
        let plan = plan_of(&valid_entries());
        let mut table = lanes();
        // AR: the pick becomes the next seed.
        let h = &mut table.get_mut(&epoch(1)).unwrap().host;
        assert_eq!(commit_host(h, &plan.requests[0], 5), Some(5));
        assert_eq!((h.pending_seed, h.generated.clone()), (Some(5), vec![5]));
        // A non-final tile only advances `prefilled`.
        let h = &mut table.get_mut(&epoch(3)).unwrap().host;
        assert!(!wants_pick(h, &plan.requests[2]));
        assert_eq!(commit_host(h, &plan.requests[2], u32::MAX), None);
        assert_eq!((h.prefilled, h.status, h.pending_seed), (8, LaneStatus::Prefilling, None));
        // The prompt-completing tile flips the lane to Decoding with a seed.
        let last = plan_of(&[tile(3, 2, &[108, 109], 8)]);
        assert!(wants_pick(h, &last.requests[0]));
        assert_eq!(commit_host(h, &last.requests[0], 42), Some(42));
        assert_eq!(
            (h.prefilled, h.status, h.pending_seed, h.generated.clone()),
            (10, LaneStatus::Decoding, Some(42), vec![42])
        );
    }
}
