// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Exact AR lane store of the Flash-Next batch route.
//!
//! A lane is one admitted request with its own [`Qwen4State`]; nothing is
//! shared between lanes but the weights. [`Qwen4LaneStore`] owns the lanes
//! and the step output buffers; [`crate::lane_exec::Qwen4LaneExecutor`] runs
//! plans over it. The host-side lane table ([`LaneTable`]) is generic over
//! its device payload so its admission/release/poison rules are covered by
//! CPU tests.
//!
//! Lane device states are allocated on a slot's first admission and kept
//! across releases: a later admission into that slot resets the state in
//! place instead of allocating again.

use crate::bundle::Qwen4Bundle;
use crate::config::Qwen4Config;
use crate::gpu_forward_lanes::LaneStagePolicy;
use crate::kv_backend::{Qwen4ContextCommit, Qwen4KvBackend};
use crate::lane_exec::LanePhase;
use crate::mtp_gpu::MtpGpuState;
use crate::state::{Qwen4State, Qwen4StateFormat};
use hipfire_runtime::slot_batch::RequestEpoch;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Most lanes one store holds.
const MAX_LANES: usize = 64;

/// What a staged lane store reports (load ack, oracle receipts).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaneStoreReceipt {
    pub kv_backend: &'static str,
    pub qsa_format: &'static str,
    pub gdn_format: &'static str,
    /// GDN capture rows armed per lane (0: AR lanes arm no ring).
    pub ring_rows: usize,
    pub max_lanes: usize,
    pub row_budget: usize,
    /// Device bytes the lanes' QSA context arenas commit right now.
    pub mapped_bytes: u64,
    pub stage_policy: String,
    pub stage_evidence: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneStatus {
    /// The prompt is being forwarded tile by tile.
    Prefilling,
    /// The prompt is done; one AR row per step.
    Decoding,
    /// A step failed with this lane in it: it runs nothing and commits
    /// nothing until released.
    Poisoned,
}

/// Host bookkeeping of one admitted request.
#[derive(Clone, Debug)]
pub struct LaneHost {
    pub epoch: RequestEpoch,
    /// Stable lane index.
    pub slot: usize,
    pub prompt: Vec<u32>,
    /// Prompt tokens forwarded so far.
    pub prefilled: usize,
    /// Picks committed so far (the first is the prompt-final pick).
    pub generated: Vec<u32>,
    /// The next AR row's token.
    pub pending_seed: Option<u32>,
    pub status: LaneStatus,
    /// Admission order across the store's lifetime.
    pub stamp: u64,
}

/// One admitted request's whole device state.
pub struct Qwen4LaneState {
    pub host: LaneHost,
    pub state: Qwen4State,
    /// Always `None` on AR lanes.
    pub mtp: Option<MtpGpuState>,
}

/// A lane-table cell: host bookkeeping plus a device payload that outlives
/// the request (the table keeps it when the lane is released).
trait LaneCell {
    type Payload;
    fn assemble(host: LaneHost, payload: Self::Payload) -> Self;
    fn host(&self) -> &LaneHost;
    fn host_mut(&mut self) -> &mut LaneHost;
}

impl LaneCell for Qwen4LaneState {
    type Payload = (Qwen4State, Option<MtpGpuState>);

    fn assemble(host: LaneHost, (state, mtp): Self::Payload) -> Self {
        Self { host, state, mtp }
    }

    fn host(&self) -> &LaneHost {
        &self.host
    }

    fn host_mut(&mut self) -> &mut LaneHost {
        &mut self.host
    }
}

/// Slot table of the lanes: which slots hold a live request and the cells
/// (host + payload) allocated so far. Epoch lookups match both the request
/// tag and the owner generation, so a stale generation never matches.
struct LaneTable<C: LaneCell> {
    cells: Vec<Option<C>>,
    live: Vec<bool>,
    next_stamp: u64,
}

impl<C: LaneCell> LaneTable<C> {
    fn new(max_lanes: usize) -> Self {
        Self {
            cells: (0..max_lanes).map(|_| None).collect(),
            live: vec![false; max_lanes],
            next_stamp: 0,
        }
    }

    fn max_lanes(&self) -> usize {
        self.cells.len()
    }

    fn slot_of(&self, epoch: &RequestEpoch) -> Option<usize> {
        self.cells.iter().enumerate().find_map(|(slot, cell)| {
            let cell = cell.as_ref()?;
            (self.live[slot] && cell.host().epoch == *epoch).then_some(slot)
        })
    }

    fn get(&self, epoch: &RequestEpoch) -> Option<&C> {
        self.slot_of(epoch).and_then(|slot| self.cells[slot].as_ref())
    }

    fn get_mut(&mut self, epoch: &RequestEpoch) -> Option<&mut C> {
        let slot = self.slot_of(epoch)?;
        self.cells[slot].as_mut()
    }

    fn active(&self) -> usize {
        self.live.iter().filter(|live| **live).count()
    }

    /// Live epochs in slot order.
    fn epochs(&self) -> Vec<RequestEpoch> {
        self.cells
            .iter()
            .enumerate()
            .filter(|(slot, _)| self.live[*slot])
            .filter_map(|(_, cell)| cell.as_ref().map(|c| c.host().epoch))
            .collect()
    }

    /// Admit `epoch` into the lowest free slot. `prepare(slot, existing)` is
    /// called exactly once: with the slot's kept payload cell (it must reset
    /// it in place and return `Ok(None)`), or with `None` when the slot has
    /// never held one (it must return the new payload). The new host entry
    /// is published only after `prepare` succeeded.
    fn admit(
        &mut self,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
        prepare: impl FnOnce(usize, Option<&mut C>) -> Result<Option<C::Payload>, String>,
    ) -> Result<usize, String> {
        if !epoch.is_admitted() {
            return Err(format!("admit: generation 0 is never admitted ({epoch:?})"));
        }
        if self.slot_of(&epoch).is_some() {
            return Err(format!("admit: epoch {epoch:?} is already live"));
        }
        if prompt.is_empty() {
            return Err("admit: empty prompt".to_string());
        }
        let Some(slot) = self.live.iter().position(|live| !*live) else {
            return Err(format!("admit: all {} lanes are busy", self.max_lanes()));
        };
        let host = LaneHost {
            epoch,
            slot,
            prompt,
            prefilled: 0,
            generated: Vec::new(),
            pending_seed: None,
            status: LaneStatus::Prefilling,
            stamp: self.next_stamp,
        };
        match self.cells[slot].as_mut() {
            Some(cell) => {
                if prepare(slot, Some(&mut *cell))?.is_some() {
                    return Err("admit: prepare returned a payload for a kept cell".to_string());
                }
                *cell.host_mut() = host;
            }
            None => {
                let payload = prepare(slot, None)?
                    .ok_or_else(|| "admit: prepare returned no payload for a new cell".to_string())?;
                self.cells[slot] = Some(C::assemble(host, payload));
            }
        }
        self.live[slot] = true;
        self.next_stamp += 1;
        Ok(slot)
    }

    /// Free `epoch`'s slot; its payload stays for the next admission.
    fn release(&mut self, epoch: &RequestEpoch) -> Result<(), String> {
        let slot = self
            .slot_of(epoch)
            .ok_or_else(|| format!("release: unknown or stale epoch {epoch:?}"))?;
        self.live[slot] = false;
        let host = self.cells[slot].as_mut().expect("live slot has a cell").host_mut();
        host.prompt = Vec::new();
        host.generated = Vec::new();
        host.pending_seed = None;
        Ok(())
    }

    /// Mark `epoch` untrusted; unknown epochs are ignored.
    fn poison(&mut self, epoch: &RequestEpoch) {
        if let Some(cell) = self.get_mut(epoch) {
            cell.host_mut().status = LaneStatus::Poisoned;
        }
    }

    /// Replace a Decoding lane's next AR token.
    fn force_seed(&mut self, epoch: &RequestEpoch, token: u32) -> Result<(), String> {
        let host = self
            .get_mut(epoch)
            .ok_or_else(|| format!("force_seed: unknown or stale epoch {epoch:?}"))?
            .host_mut();
        if host.status != LaneStatus::Decoding {
            return Err(format!("force_seed: lane {:?} is {:?}, not Decoding", epoch, host.status));
        }
        host.pending_seed = Some(token);
        Ok(())
    }

    /// Disjoint `&mut` cells of the live lanes `epochs`, in that order.
    fn pick_mut(&mut self, epochs: &[RequestEpoch]) -> Result<Vec<&mut C>, String> {
        let slots = epochs
            .iter()
            .map(|epoch| {
                self.slot_of(epoch)
                    .ok_or_else(|| format!("unknown or stale epoch {epoch:?}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut refs: Vec<Option<&mut C>> = self.cells.iter_mut().map(|c| c.as_mut()).collect();
        slots
            .iter()
            .map(|&slot| {
                refs[slot]
                    .take()
                    .ok_or_else(|| format!("lane slot {slot} named twice"))
            })
            .collect()
    }

    fn into_cells(self) -> Vec<C> {
        self.cells.into_iter().flatten().collect()
    }
}

/// Disjoint borrows one step needs out of the store.
pub(crate) struct StepParts<'a> {
    /// The planned lanes, in plan order.
    pub lanes: Vec<&'a mut Qwen4LaneState>,
    /// F32, `2 * max_lanes * vocab`.
    pub logits: &'a GpuTensor,
    /// Raw, `2 * max_lanes * 4` bytes.
    pub top1: &'a GpuTensor,
    pub policy: LaneStagePolicy,
    /// Host mirror of `top1`.
    pub top1_host: &'a mut Vec<u8>,
}

/// The lane owner: lanes, step output buffers, stage policy and the step
/// phase machine. See the module docs of [`crate::lane_exec`].
pub struct Qwen4LaneStore {
    table: LaneTable<Qwen4LaneState>,
    row_budget: usize,
    ring_rows: usize,
    format: Qwen4StateFormat,
    backend: Qwen4KvBackend,
    policy: LaneStagePolicy,
    /// Rows `[0, max_lanes)`: the AR rows of a step; `[max_lanes,
    /// 2 * max_lanes)`: the prompt-completing prefill rows.
    logits: GpuTensor,
    top1: GpuTensor,
    top1_host: Vec<u8>,
    pub(crate) phase: LanePhase,
    pub(crate) next_step_id: u64,
}

impl Qwen4LaneStore {
    /// An empty store for `max_lanes` lanes (states are allocated on first
    /// admission into a slot) and its step output buffers.
    pub(crate) fn new(
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        max_lanes: usize,
        row_budget: usize,
    ) -> Result<Self, String> {
        if max_lanes == 0 || max_lanes > MAX_LANES {
            return Err(format!("lane store: max_lanes {max_lanes} outside 1..={MAX_LANES}"));
        }
        if row_budget == 0 {
            return Err("lane store: row_budget must be at least 1".to_string());
        }
        let rows = max_lanes * 2;
        let vocab = bundle.lane_vocab();
        let logit_elements = rows
            .checked_mul(vocab)
            .ok_or_else(|| "lane store: logits extent overflow".to_string())?;
        let logits = gpu
            .zeros(&[logit_elements], DType::F32)
            .map_err(|e| format!("lane store: logits: {e}"))?;
        let top1 = match gpu.zeros(&[rows * std::mem::size_of::<i32>()], DType::Raw) {
            Ok(tensor) => tensor,
            Err(error) => {
                let _ = gpu.free_tensor(logits);
                return Err(format!("lane store: top1: {error}"));
            }
        };
        Ok(Self {
            table: LaneTable::new(max_lanes),
            row_budget: row_budget.min(max_lanes),
            ring_rows: 0,
            format: bundle.state_format(),
            backend: bundle.state.qsa_backend(),
            // Shared stages only where G0 evidenced them (gfx1151).
            policy: LaneStagePolicy::for_arch(&gpu.arch),
            logits,
            top1,
            top1_host: vec![0; rows * std::mem::size_of::<i32>()],
            phase: LanePhase::Idle,
            next_step_id: 0,
        })
    }

    /// Device bytes the store commits at load: `max_lanes` lane states
    /// ([`Qwen4State::device_bytes`]) plus the step logits and top-1
    /// buffers. `_row_budget` is accepted for the frozen signature: AR rows
    /// share the `max_lanes` output rows, so the budget adds nothing.
    pub fn device_bytes(
        config: &Qwen4Config,
        format: Qwen4StateFormat,
        context: &Qwen4ContextCommit,
        max_lanes: usize,
        _row_budget: usize,
    ) -> Option<u64> {
        let state = Qwen4State::device_bytes(config, format, context)?;
        let rows = max_lanes.checked_mul(2)?;
        let outputs = rows
            .checked_mul(config.vocab_size)?
            .checked_mul(std::mem::size_of::<f32>())?
            .checked_add(rows.checked_mul(std::mem::size_of::<i32>())?)?;
        u64::try_from(max_lanes.checked_mul(state)?.checked_add(outputs)?).ok()
    }

    /// Admit `epoch` with its canonical `prompt` into the lowest free lane
    /// and return its slot. A slot that held a request before resets its
    /// state in place; a fresh slot allocates one.
    pub fn admit(
        &mut self,
        gpu: &mut Gpu,
        bundle: &Qwen4Bundle,
        epoch: RequestEpoch,
        prompt: Vec<u32>,
    ) -> Result<usize, String> {
        self.table.admit(epoch, prompt, |slot, existing| match existing {
            Some(cell) => {
                cell.state
                    .reset(gpu)
                    .map_err(|e| format!("admit: reset lane {slot}: {e}"))?;
                Ok(None)
            }
            None => {
                let state = bundle
                    .new_lane_state(gpu)
                    .map_err(|e| format!("admit: allocate lane {slot}: {e}"))?;
                Ok(Some((state, None)))
            }
        })
    }

    /// Free `epoch`'s lane (its device state stays allocated for reuse).
    pub fn release(&mut self, epoch: &RequestEpoch) -> Result<(), String> {
        self.table.release(epoch)
    }

    /// Mark `epoch`'s lane untrusted; unknown epochs are ignored.
    pub fn poison(&mut self, epoch: &RequestEpoch) {
        self.table.poison(epoch);
    }

    /// Replace a Decoding lane's next AR token (the singleton's think-budget
    /// forced close tokens); `generated` is untouched.
    pub fn force_seed(&mut self, epoch: &RequestEpoch, token: u32) -> Result<(), String> {
        self.table.force_seed(epoch, token)
    }

    pub fn lane(&self, epoch: &RequestEpoch) -> Option<&Qwen4LaneState> {
        self.table.get(epoch)
    }

    pub(crate) fn lane_mut(&mut self, epoch: &RequestEpoch) -> Option<&mut Qwen4LaneState> {
        self.table.get_mut(epoch)
    }

    /// Live epochs in slot order.
    pub fn epochs(&self) -> Vec<RequestEpoch> {
        self.table.epochs()
    }

    /// The step logits (F32, `2 * max_lanes` rows of `vocab`): the k-th AR
    /// request of the plan, in plan order, is row `k`; the i-th Prefill
    /// request is row `max_lanes + i` (written on a prompt-completing tile).
    /// Valid until the next forward; read it before `commit_step`.
    pub fn logits(&self) -> &GpuTensor {
        &self.logits
    }

    /// Disjoint borrows of the `epochs` lanes (plan order) and the step
    /// output buffers.
    pub(crate) fn step_parts(&mut self, epochs: &[RequestEpoch]) -> Result<StepParts<'_>, String> {
        let Self {
            table,
            policy,
            logits,
            top1,
            top1_host,
            ..
        } = self;
        Ok(StepParts {
            lanes: table.pick_mut(epochs)?,
            logits,
            top1,
            policy: *policy,
            top1_host,
        })
    }

    pub fn receipt(&self, gpu: &Gpu) -> LaneStoreReceipt {
        let mapped: usize = self
            .table
            .cells
            .iter()
            .flatten()
            .map(|lane| lane.state.mapped_context_bytes(gpu).unwrap_or(0))
            .sum();
        LaneStoreReceipt {
            kv_backend: self.backend.name(),
            qsa_format: self.format.qsa.name(),
            gdn_format: self.format.gdn.name(),
            ring_rows: self.ring_rows,
            max_lanes: self.max_lanes(),
            row_budget: self.row_budget,
            mapped_bytes: mapped as u64,
            stage_policy: self.policy.describe(),
            stage_evidence: self.policy.evidence(),
        }
    }

    pub fn stage_policy(&self) -> LaneStagePolicy {
        self.policy
    }

    pub fn set_stage_policy(&mut self, policy: LaneStagePolicy) {
        self.policy = policy;
    }

    pub fn max_lanes(&self) -> usize {
        self.table.max_lanes()
    }

    /// Most AR rows one step may carry.
    pub fn row_budget(&self) -> usize {
        self.row_budget
    }

    /// Live lanes.
    pub fn active(&self) -> usize {
        self.table.active()
    }

    /// Free every lane state and the step buffers; the first error wins.
    pub(crate) fn free_gpu(self, gpu: &mut Gpu) -> Result<(), String> {
        let Self {
            table,
            logits,
            top1,
            ..
        } = self;
        let mut first: Option<String> = None;
        for lane in table.into_cells() {
            let Qwen4LaneState { state, mtp, .. } = lane;
            if let Err(error) = state.free_gpu(gpu) {
                first.get_or_insert(format!("lane state: {error}"));
            }
            if let Some(mtp) = mtp {
                if let Some(error) = mtp.free_gpu(gpu) {
                    first.get_or_insert(format!("lane mtp state: {error}"));
                }
            }
        }
        for (name, tensor) in [("logits", logits), ("top1", top1)] {
            if let Err(error) = gpu.free_tensor(tensor) {
                first.get_or_insert(format!("lane {name}: {error}"));
            }
        }
        first.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestCell {
        host: LaneHost,
        payload: u32,
    }

    impl LaneCell for TestCell {
        type Payload = u32;

        fn assemble(host: LaneHost, payload: u32) -> Self {
            Self { host, payload }
        }

        fn host(&self) -> &LaneHost {
            &self.host
        }

        fn host_mut(&mut self) -> &mut LaneHost {
            &mut self.host
        }
    }

    fn epoch(tag: u64, generation: u64) -> RequestEpoch {
        RequestEpoch {
            request_tag: tag,
            owner_generation: generation,
        }
    }

    /// Admission with a counting payload factory.
    fn admit(
        table: &mut LaneTable<TestCell>,
        epoch: RequestEpoch,
        prompt: &[u32],
        made: &mut usize,
    ) -> Result<usize, String> {
        table.admit(epoch, prompt.to_vec(), |slot, existing| match existing {
            Some(cell) => {
                cell.payload += 100;
                Ok(None)
            }
            None => {
                *made += 1;
                Ok(Some(slot as u32))
            }
        })
    }

    #[test]
    fn admission_takes_the_lowest_free_slot() {
        let mut table = LaneTable::<TestCell>::new(3);
        let mut made = 0;
        assert_eq!(admit(&mut table, epoch(1, 1), &[7], &mut made), Ok(0));
        assert_eq!(admit(&mut table, epoch(2, 1), &[7], &mut made), Ok(1));
        assert_eq!(admit(&mut table, epoch(3, 1), &[7], &mut made), Ok(2));
        table.release(&epoch(2, 1)).unwrap();
        assert_eq!(admit(&mut table, epoch(4, 1), &[7], &mut made), Ok(1));
        assert_eq!(table.active(), 3);
        let host = &table.get(&epoch(4, 1)).unwrap().host;
        assert_eq!(host.status, LaneStatus::Prefilling);
        assert_eq!((host.prefilled, host.generated.len(), host.pending_seed), (0, 0, None));
        assert_eq!(host.stamp, 3);
        assert_eq!(table.epochs(), vec![epoch(1, 1), epoch(4, 1), epoch(3, 1)]);
    }

    #[test]
    fn admission_refusals() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        assert!(admit(&mut table, epoch(1, 0), &[7], &mut made)
            .unwrap_err()
            .contains("generation 0"));
        assert!(admit(&mut table, epoch(1, 1), &[], &mut made)
            .unwrap_err()
            .contains("empty prompt"));
        assert_eq!(made, 0);
        admit(&mut table, epoch(1, 1), &[7], &mut made).unwrap();
        assert!(admit(&mut table, epoch(1, 1), &[7], &mut made)
            .unwrap_err()
            .contains("already live"));
        admit(&mut table, epoch(2, 1), &[7], &mut made).unwrap();
        assert!(admit(&mut table, epoch(3, 1), &[7], &mut made)
            .unwrap_err()
            .contains("busy"));
        assert_eq!(made, 2);
        assert_eq!(table.active(), 2);
    }

    #[test]
    fn a_failed_prepare_leaves_the_table_untouched() {
        let mut table = LaneTable::<TestCell>::new(1);
        let error = table
            .admit(epoch(1, 1), vec![7], |_, _| Err("no memory".to_string()))
            .unwrap_err();
        assert_eq!(error, "no memory");
        assert_eq!(table.active(), 0);
        assert_eq!(table.next_stamp, 0);
        assert!(table.get(&epoch(1, 1)).is_none());
    }

    #[test]
    fn release_keeps_the_payload_and_reuse_does_not_remake_it() {
        let mut table = LaneTable::<TestCell>::new(1);
        let mut made = 0;
        admit(&mut table, epoch(1, 1), &[7, 8], &mut made).unwrap();
        assert_eq!(table.get(&epoch(1, 1)).unwrap().payload, 0);
        table.release(&epoch(1, 1)).unwrap();
        assert_eq!(table.active(), 0);
        assert!(table.get(&epoch(1, 1)).is_none());
        admit(&mut table, epoch(2, 1), &[9], &mut made).unwrap();
        assert_eq!(made, 1);
        let cell = table.get(&epoch(2, 1)).unwrap();
        assert_eq!(cell.payload, 100);
        assert_eq!(cell.host.prompt, vec![9]);
        assert_eq!(cell.host.stamp, 1);
    }

    #[test]
    fn release_of_an_unknown_epoch_fails() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        assert!(table.release(&epoch(9, 1)).is_err());
        admit(&mut table, epoch(1, 1), &[7], &mut made).unwrap();
        table.release(&epoch(1, 1)).unwrap();
        assert!(table.release(&epoch(1, 1)).is_err());
    }

    #[test]
    fn a_new_generation_of_the_same_tag_is_admitted_after_release() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        admit(&mut table, epoch(5, 1), &[7], &mut made).unwrap();
        // Same tag, other generation: a different owner, not a duplicate.
        assert!(admit(&mut table, epoch(5, 2), &[7], &mut made).is_ok());
        table.release(&epoch(5, 1)).unwrap();
        // The stale generation matches nothing, the live one still does.
        assert!(table.get(&epoch(5, 1)).is_none());
        assert!(table.get(&epoch(5, 2)).is_some());
        assert!(table.release(&epoch(5, 1)).is_err());
        admit(&mut table, epoch(5, 3), &[7], &mut made).unwrap();
        assert!(table.get(&epoch(5, 3)).is_some());
        assert!(table.get(&epoch(5, 1)).is_none());
    }

    #[test]
    fn poison_marks_only_live_matching_lanes() {
        let mut table = LaneTable::<TestCell>::new(2);
        let mut made = 0;
        admit(&mut table, epoch(1, 1), &[7], &mut made).unwrap();
        admit(&mut table, epoch(2, 1), &[7], &mut made).unwrap();
        table.poison(&epoch(1, 2));
        table.poison(&epoch(9, 1));
        assert_eq!(table.get(&epoch(1, 1)).unwrap().host.status, LaneStatus::Prefilling);
        table.poison(&epoch(1, 1));
        assert_eq!(table.get(&epoch(1, 1)).unwrap().host.status, LaneStatus::Poisoned);
        assert_eq!(table.get(&epoch(2, 1)).unwrap().host.status, LaneStatus::Prefilling);
    }

    #[test]
    fn force_seed_needs_a_live_decoding_lane() {
        let mut table = LaneTable::<TestCell>::new(3);
        let mut made = 0;
        for tag in 1..=3 {
            admit(&mut table, epoch(tag, 1), &[7], &mut made).unwrap();
        }
        // Prefilling lane: refused.
        assert!(table.force_seed(&epoch(1, 1), 5).is_err());
        table.get_mut(&epoch(1, 1)).unwrap().host.status = LaneStatus::Decoding;
        table.get_mut(&epoch(1, 1)).unwrap().host.generated.push(4);
        table.force_seed(&epoch(1, 1), 5).unwrap();
        let host = &table.get(&epoch(1, 1)).unwrap().host;
        assert_eq!(host.pending_seed, Some(5));
        assert_eq!(host.generated, vec![4]);
        // Poisoned lane and unknown epoch: refused.
        table.get_mut(&epoch(2, 1)).unwrap().host.status = LaneStatus::Decoding;
        table.poison(&epoch(2, 1));
        assert!(table.force_seed(&epoch(2, 1), 5).is_err());
        assert!(table.force_seed(&epoch(9, 1), 5).is_err());
        assert!(table.force_seed(&epoch(1, 2), 5).is_err());
    }

    #[test]
    fn pick_mut_returns_disjoint_lanes_in_request_order() {
        let mut table = LaneTable::<TestCell>::new(3);
        let mut made = 0;
        for tag in 1..=3 {
            admit(&mut table, epoch(tag, 1), &[7], &mut made).unwrap();
        }
        let picked = table
            .pick_mut(&[epoch(3, 1), epoch(1, 1)])
            .unwrap();
        assert_eq!(picked.iter().map(|c| c.host.slot).collect::<Vec<_>>(), vec![2, 0]);
        assert!(table.pick_mut(&[epoch(1, 1), epoch(1, 1)]).is_err());
        assert!(table.pick_mut(&[epoch(1, 9)]).is_err());
    }
}
