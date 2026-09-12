// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Native Qwen4 MTP acceptance and transaction-shape helpers.
//!
//! The runtime owns the speculative loop and its pending-seed contract.  This
//! module only records the native MTP count convention and lowers an already
//! verified greedy result onto the canonical runtime types.  In particular,
//! the seed is never copied into `MtpWindow::committed` or `SpecStep::emit`.
//! The target's rollback count is explicit: native `num_accepted_tokens`
//! includes the seed, so `accept_len = num_accepted_tokens - 1` counts only
//! accepted drafts.

use crate::bundle::Qwen4Bundle;
use crate::mtp::{MtpError, Qwen4MtpState};
use crate::state::Qwen4StateSnapshot;
use hipfire_runtime::spec::{
    accept_greedy_prefix, MtpDrafter, MtpSpeculator, MtpWindow, SpecAdvance, SpecGrammar,
    SpecRequestConfig, SpecScratch, SpecStep, SpecTarget, Speculator,
};
use rdna_compute::{Gpu, GpuTensor};

/// Result of one native greedy MTP target comparison.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeMtpAcceptance {
    /// Accepted drafts followed by the target bonus, excluding the seed.
    /// When an accepted draft is EOS, the runtime's shared rule stops there
    /// and no bonus is appended.
    pub committed: Vec<u32>,
    /// Number of draft candidates that matched the target.
    pub accepted_drafts: usize,
    /// Number of candidates offered to the verifier.
    pub drafts_generated: usize,
    /// Native count including the seed and accepted drafts, but excluding the
    /// bonus.  This is the count whose `- 1` value is passed to rollback.
    pub num_accepted_tokens: usize,
    /// Runtime target rollback count: accepted drafts only.
    pub rollback_accept_len: usize,
    /// Last committed token, which is the next window's pending seed.
    pub next_seed: u32,
    /// Whether EOS terminated this window.
    pub hit_eos: bool,
}
impl NativeMtpAcceptance {
    /// Whether EOS was an accepted draft rather than the verifier's bonus.
    /// Accepted EOS remains a pending seed, so neither target nor MTP state
    /// consumes that final token until the terminal flush.
    pub fn accepted_eos(&self) -> bool {
        self.hit_eos && self.committed.len() == self.accepted_drafts
    }

    /// Number of accepted drafts the target should commit before the pending
    /// EOS seed. A bonus EOS is already predicted after this prefix and uses
    /// the ordinary accepted-draft count.
    pub fn target_commit_accept_len(&self) -> usize {
        if self.accepted_eos() {
            self.rollback_accept_len.saturating_sub(1)
        } else {
            self.rollback_accept_len
        }
    }

    /// Captured target-hidden row that produced the next pending seed.
    pub fn pending_hidden_row(&self) -> usize {
        self.target_commit_accept_len()
    }
}

/// Native MTP uses greedy target picks only.  A sampled request must fail
/// closed until an exact distribution-verification implementation exists.
pub fn require_native_greedy(temp: f32) -> Result<(), String> {
    if !temp.is_finite() || temp.abs() > 1.0e-6 {
        return Err(
            "Qwen4 native MTP supports greedy verification only; sampled MTP requires exact distribution verification"
                .to_string(),
        );
    }
    Ok(())
}

/// Refuse native MTP prefill requests that would reuse a cached suffix.
///
/// Native MTP prefill is currently a cold, position-zero operation.  Silent
/// truncation of a cache-hit suffix would leave target and MTP state at
/// different positions, so callers must reject it before touching either
/// owner.
pub fn validate_native_mtp_prefill_request(
    prompt_tokens: &[u32],
    fill_tokens: &[u32],
    start_pos: usize,
    cache_hit: bool,
) -> Result<(), String> {
    if prompt_tokens.is_empty() {
        return Err("Qwen4 native MTP prefill requires at least one prompt token".to_string());
    }
    if cache_hit {
        return Err("Qwen4 native MTP prefill refuses cache-hit suffix reuse".to_string());
    }
    if start_pos != 0 {
        return Err(format!(
            "Qwen4 native MTP prefill requires position zero, got {start_pos}"
        ));
    }
    if fill_tokens != prompt_tokens {
        return Err("Qwen4 native MTP prefill requires a complete prompt fill".to_string());
    }
    Ok(())
}

/// Convert the native count (which includes the seed) into the runtime's
/// accepted-draft count.  Zero is invalid because the seed itself is always
/// present in a verified block.
pub fn native_rollback_accept_len(num_accepted_tokens: usize) -> Result<usize, String> {
    num_accepted_tokens
        .checked_sub(1)
        .ok_or_else(|| "Qwen4 native MTP accepted-token count cannot be zero".to_string())
}

/// Absolute target positions occupied by the committed verify prefix.
///
/// `position` is the seed's position and `num_accepted_tokens` includes that
/// seed, so the returned range has exactly `num_accepted_tokens` rows.  The
/// bonus is predicted, not consumed, and is therefore intentionally absent.
pub fn committed_target_positions(
    position: usize,
    num_accepted_tokens: usize,
) -> Result<Vec<usize>, String> {
    let end = position
        .checked_add(num_accepted_tokens)
        .ok_or_else(|| "Qwen4 native MTP target position overflow".to_string())?;
    Ok((position..end).collect())
}

/// Compact target-aligned MTP QSA rows after a partial acceptance.
///
/// The caller supplies the seed position and the native count.  Rows at or
/// after the committed end are rejected-tail state and are discarded from the
/// sparse selection.  Full main K/V is restored/replayed by the target
/// transaction; this helper only handles the MTP side index row.
pub fn compact_native_qsa_selection(
    state: &mut Qwen4MtpState,
    position: usize,
    num_accepted_tokens: usize,
) -> Result<(), String> {
    let retained_end = position
        .checked_add(num_accepted_tokens)
        .ok_or_else(|| "Qwen4 native MTP QSA position overflow".to_string())?;
    if retained_end > state.qsa.position {
        return Err(format!(
            "Qwen4 native MTP QSA commit end {retained_end} exceeds active position {}",
            state.qsa.position
        ));
    }
    let retained = state
        .qsa
        .selected_indices
        .iter()
        .copied()
        .filter(|&row| row < retained_end)
        .collect::<Vec<_>>();
    state
        .qsa
        .compact_selection(&retained)
        .map_err(|error| format!("Qwen4 native MTP QSA compaction: {error}"))
}

/// Compute the target's post-commit position without mutating either owner.
pub fn native_commit_position(
    position: usize,
    num_accepted_tokens: usize,
) -> Result<usize, String> {
    position
        .checked_add(num_accepted_tokens)
        .ok_or_else(|| "Qwen4 native MTP commit position overflow".to_string())
}

/// Apply the runtime's one shared greedy acceptance rule to native MTP picks.
///
/// `target_picks` has one extra slot for the bonus.  The returned
/// `num_accepted_tokens` deliberately includes the seed, making the
/// `num_accepted_tokens - 1` rollback offset observable and testable rather
/// than inferred from draft count.  The seed itself is intentionally not an
/// argument: it is already represented by the target block and is excluded
/// from the emitted vector by construction.
pub fn accept_native_greedy(
    drafts: &[u32],
    target_picks: &[u32],
    eos: Option<u32>,
) -> Result<NativeMtpAcceptance, String> {
    if target_picks.len() < drafts.len().saturating_add(1) {
        return Err(format!(
            "Qwen4 native MTP verifier returned {} picks for {} drafts; one bonus pick is required",
            target_picks.len(),
            drafts.len()
        ));
    }
    let accepted = accept_greedy_prefix(drafts, target_picks, eos);
    let next_seed = *accepted
        .committed
        .last()
        .ok_or_else(|| "Qwen4 native MTP verifier committed no token".to_string())?;
    let num_accepted_tokens = accepted
        .accepted
        .checked_add(1)
        .ok_or_else(|| "Qwen4 native MTP accepted-token count overflow".to_string())?;
    let rollback_accept_len = native_rollback_accept_len(num_accepted_tokens)?;
    debug_assert_eq!(rollback_accept_len, accepted.accepted);
    Ok(NativeMtpAcceptance {
        committed: accepted.committed,
        accepted_drafts: accepted.accepted,
        drafts_generated: drafts.len(),
        num_accepted_tokens,
        rollback_accept_len,
        next_seed,
        hit_eos: accepted.hit_eos,
    })
}

/// Lower native acceptance onto the runtime's canonical MTP window.
pub fn acceptance_to_window(acceptance: NativeMtpAcceptance) -> Result<MtpWindow, String> {
    if acceptance.committed.is_empty() {
        return Err("Qwen4 native MTP cannot lower an empty committed window".to_string());
    }
    if acceptance.accepted_drafts > acceptance.drafts_generated {
        return Err(format!(
            "Qwen4 native MTP accepted {} drafts out of {}",
            acceptance.accepted_drafts, acceptance.drafts_generated
        ));
    }
    if acceptance.rollback_accept_len != acceptance.accepted_drafts
        || acceptance.num_accepted_tokens != acceptance.accepted_drafts + 1
    {
        return Err("Qwen4 native MTP accepted-count convention is inconsistent".to_string());
    }
    Ok(MtpWindow {
        committed: acceptance.committed,
        accepted: acceptance.accepted_drafts,
        drafts_generated: acceptance.drafts_generated,
    })
}

/// Lower an MTP window directly to the runtime's pending-seed result.
/// `MtpWindow::committed` already excludes the seed, so this is a 1:1 emit
/// mapping and never performs a DFlash-style seed re-echo transformation.
pub fn window_to_spec_step(window: MtpWindow) -> Result<SpecStep, String> {
    let next_seed = *window
        .committed
        .last()
        .ok_or_else(|| "Qwen4 native MTP committed no token (would stall decode)".to_string())?;
    if window.accepted > window.drafts_generated {
        return Err(format!(
            "Qwen4 native MTP accepted {} drafts out of {}",
            window.accepted, window.drafts_generated
        ));
    }
    Ok(SpecStep::new(
        window.committed,
        next_seed,
        window.drafts_generated,
        window.accepted,
    ))
}
/// Reusable Qwen4 target-side verify scratch.  GPU output buffers and the
/// captured wide hidden rows belong to the bundle; this object owns only the
/// checked-out rollback ticket and its fixed block capacity.
pub struct Qwen4SpecScratch {
    block_size: usize,
    target_snapshot: Option<Qwen4StateSnapshot>,
}

impl SpecScratch for Qwen4SpecScratch {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn free(self: Box<Self>, _gpu: &mut Gpu) {
        // A live ticket is consumed by verify/commit before the speculator is
        // released.  Bundle teardown also owns the arena, so there is no
        // independent GPU allocation to free here.
        debug_assert!(
            self.target_snapshot.is_none(),
            "Qwen4 verify scratch dropped with an active target snapshot"
        );
    }
}

impl SpecTarget for Qwen4Bundle {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn reset_recurrent(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.reset(gpu)
            .map_err(|error| format!("Qwen4 reset_recurrent: {error}"))
    }

    fn retry_reset_eligible(&self) -> bool {
        true
    }

    fn new_spec_scratch(
        &mut self,
        gpu: &mut Gpu,
        block_size: usize,
    ) -> Result<Box<dyn SpecScratch>, String> {
        let block_size = block_size.max(1);
        let max_chunk = self
            .execution
            .as_ref()
            .ok_or_else(|| "Qwen4 spec scratch requires attached forward resources".to_string())?
            .scratch
            .max_chunk;
        if block_size > max_chunk {
            return Err(format!(
                "Qwen4 spec block size {block_size} exceeds forward capacity {max_chunk}"
            ));
        }
        self.ensure_spec_hidden(gpu, block_size)
            .map_err(|error| error.to_string())?;
        Ok(Box::new(Qwen4SpecScratch {
            block_size,
            target_snapshot: None,
        }))
    }

    fn spec_advance(
        &mut self,
        gpu: &mut Gpu,
        tokens: &[u32],
        start_pos: usize,
        reset: bool,
        abort: &dyn Fn() -> bool,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<SpecAdvance, String> {
        if reset {
            self.reset_recurrent(gpu)?;
        }
        if tokens.is_empty() {
            return Err("Qwen4 spec_advance cannot process an empty token slice".to_string());
        }
        if self.state.position != start_pos {
            return Err(format!(
                "Qwen4 spec_advance position mismatch: expected {}, got {start_pos}",
                self.state.position
            ));
        }
        let end_pos = start_pos
            .checked_add(tokens.len())
            .ok_or_else(|| "Qwen4 spec position overflow".to_string())?;
        if end_pos > self.state.max_seq_len {
            return Err(format!(
                "Qwen4 spec advance end {end_pos} exceeds context capacity {}",
                self.state.max_seq_len
            ));
        }
        let max_chunk = self
            .execution
            .as_ref()
            .ok_or_else(|| "Qwen4 forward resources are not attached".to_string())?
            .scratch
            .max_chunk;
        let mut offset = 0usize;
        let mut last_argmax = None;
        while offset < tokens.len() {
            if abort() {
                self.reset_recurrent(gpu)?;
                return Ok(SpecAdvance::Aborted);
            }
            let end = (offset + max_chunk).min(tokens.len());
            let picks = self
                .spec_forward_rows(gpu, &tokens[offset..end], false)
                .map_err(|error| error.to_string())?;
            last_argmax = picks.last().copied();
            offset = end;
        }
        if self.state.position != end_pos {
            return Err(format!(
                "Qwen4 spec advance ended at {}, expected {end_pos}",
                self.state.position
            ));
        }
        Ok(SpecAdvance::Ready {
            last_argmax: last_argmax.expect("non-empty spec advance produced no argmax"),
            last_logits: None,
        })
    }

    fn verify_block(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        position: usize,
        scratch: &mut dyn SpecScratch,
        _hidden_out: Option<&mut Vec<f32>>,
    ) -> Result<Vec<u32>, String> {
        let block_size = {
            let s = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .ok_or("Qwen4 verify_block: scratch is not Qwen4SpecScratch")?;
            if s.target_snapshot.is_some() {
                return Err("Qwen4 verify_block: target snapshot is already active".to_string());
            }
            s.block_size
        };
        if block.is_empty() || block.len() > block_size {
            return Err(format!(
                "Qwen4 verify block length {} is outside scratch capacity {block_size}",
                block.len()
            ));
        }
        if self.state.position != position {
            return Err(format!(
                "Qwen4 verify_block position mismatch: expected {}, got {position}",
                self.state.position
            ));
        }
        let end_pos = position
            .checked_add(block.len())
            .ok_or_else(|| "Qwen4 verify position overflow".to_string())?;
        if end_pos > self.state.max_seq_len {
            return Err(format!(
                "Qwen4 verify end {end_pos} exceeds context capacity {}",
                self.state.max_seq_len
            ));
        }
        let snapshot = self.snapshot(gpu).map_err(|error| error.to_string())?;
        scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 verify_block: scratch is not Qwen4SpecScratch")?
            .target_snapshot = Some(snapshot);
        let result = self
            .spec_forward_rows(gpu, block, true)
            .map_err(|error| error.to_string());
        if let Err(error) = &result {
            let snapshot = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .and_then(|s| s.target_snapshot.take());
            if let Some(snapshot) = snapshot {
                self.restore(gpu, snapshot)
                    .map_err(|restore| format!("{error}; target restore failed: {restore}"))?;
            }
            return Err(error.clone());
        }
        if self.state.position != end_pos {
            let mismatch = format!(
                "Qwen4 verify ended at {}, expected {end_pos}",
                self.state.position
            );
            let snapshot = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .and_then(|s| s.target_snapshot.take());
            if let Some(snapshot) = snapshot {
                self.restore(gpu, snapshot)
                    .map_err(|restore| format!("{mismatch}; target restore failed: {restore}"))?;
            }
            return Err(mismatch);
        }
        result
    }

    fn commit_prefix(
        &mut self,
        gpu: &mut Gpu,
        block: &[u32],
        accept_len: usize,
        position: usize,
        scratch: &mut dyn SpecScratch,
    ) -> Result<(), String> {
        if block.is_empty() {
            return Err("Qwen4 commit_prefix cannot commit an empty block".to_string());
        }
        let draft_len = block.len() - 1;
        if accept_len > draft_len {
            return Err(format!(
                "Qwen4 commit_prefix accepts {accept_len} drafts out of {draft_len}"
            ));
        }
        let verified_end = position
            .checked_add(block.len())
            .ok_or_else(|| "Qwen4 commit position overflow".to_string())?;
        if self.state.position != verified_end {
            return Err(format!(
                "Qwen4 commit_prefix target position mismatch: expected {verified_end}, got {}",
                self.state.position
            ));
        }
        let committed_end = position
            .checked_add(accept_len + 1)
            .ok_or_else(|| "Qwen4 commit position overflow".to_string())?;
        let snapshot = scratch
            .as_any_mut()
            .downcast_mut::<Qwen4SpecScratch>()
            .ok_or("Qwen4 commit_prefix: scratch is not Qwen4SpecScratch")?
            .target_snapshot
            .take()
            .ok_or("Qwen4 commit_prefix: target snapshot is not active")?;
        if accept_len == draft_len {
            return self
                .commit(gpu, snapshot)
                .map_err(|error| error.to_string());
        }
        self.restore(gpu, snapshot)
            .map_err(|error| error.to_string())?;
        self.spec_forward_rows(gpu, &block[..accept_len + 1], true)
            .map(|_| ())
            .map_err(|error| error.to_string())?;
        if self.state.position != committed_end {
            return Err(format!(
                "Qwen4 commit_prefix replay ended at {}, expected {committed_end}",
                self.state.position
            ));
        }
        Ok(())
    }

    fn eos_token(&self) -> u32 {
        self.config.eos_token_id
    }

    fn ctx_capacity(&self) -> usize {
        self.state.max_seq_len
    }
}

/// Native GPU MTP drafter.  The MTP operator/state stay model-owned by the
/// target bundle; this adapter owns only the reusable verifier scratch and one
/// pending target-hidden row needed to seed each MTP window.
pub struct Qwen4MtpDrafter {
    max_k: usize,
    ctx_capacity: usize,
    request: SpecRequestConfig,
    scratch: Option<Box<dyn SpecScratch>>,
    pending_hidden: Option<GpuTensor>,
}

impl Qwen4MtpDrafter {
    pub fn new(max_k: usize, ctx_capacity: usize) -> Self {
        Self {
            max_k: max_k.clamp(1, 10),
            ctx_capacity,
            request: SpecRequestConfig::default(),
            scratch: None,
            pending_hidden: None,
        }
    }

    fn bundle<'a>(target: &'a mut dyn SpecTarget) -> Result<&'a mut Qwen4Bundle, String> {
        target
            .as_any_mut()
            .downcast_mut::<Qwen4Bundle>()
            .ok_or_else(|| "Qwen4MtpDrafter: target is not a Qwen4Bundle".to_string())
    }

    fn ensure_resources(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
    ) -> Result<(), String> {
        let width = {
            let bundle = Self::bundle(target)?;
            bundle.mtp_position().map_err(|error| error.to_string())?;
            bundle
                .config
                .hc_count
                .checked_mul(bundle.config.hidden_size)
                .ok_or_else(|| "Qwen4 MTP hidden width overflow".to_string())?
        };
        if self.scratch.is_none() {
            let scratch = target.new_spec_scratch(gpu, self.max_k + 1)?;
            self.scratch = Some(scratch);
        }
        if self.pending_hidden.is_none() {
            self.pending_hidden = Some(
                gpu.zeros(&[width], rdna_compute::DType::F32)
                    .map_err(|error| format!("Qwen4 MTP pending hidden allocation: {error}"))?,
            );
        }
        Ok(())
    }

    fn pending_hidden(&self) -> Result<&GpuTensor, String> {
        self.pending_hidden
            .as_ref()
            .ok_or_else(|| "Qwen4 MTP pending hidden is not allocated".to_string())
    }
}

impl MtpDrafter for Qwen4MtpDrafter {
    fn mtp_prefill(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        prompt_tokens: &[u32],
        fill_tokens: &[u32],
        start_pos: usize,
        cache_hit: bool,
        abort: &dyn Fn() -> bool,
    ) -> Result<u32, String> {
        require_native_greedy(self.request.temp)?;
        validate_native_mtp_prefill_request(prompt_tokens, fill_tokens, start_pos, cache_hit)?;
        // Native Qwen4 MTP has no exact target+MTP suffix rehydration yet.
        // Always discard any AR or stale MTP prefix and rebuild the complete
        // rendered prompt from position zero.
        target.reset_recurrent(gpu)?;
        self.ensure_resources(gpu, target)?;
        {
            let bundle = Self::bundle(target)?;
            let target_position = bundle.state.position;
            let mtp_position = bundle.mtp_position().map_err(|error| error.to_string())?;
            if target_position != start_pos || mtp_position != start_pos {
                return Err(format!(
                    "Qwen4 MTP prefill position mismatch: target={}, mtp={}, start={start_pos}",
                    target_position, mtp_position
                ));
            }
        }
        let pending = self.pending_hidden()?;
        let mut first_token = None;
        for (index, &token) in fill_tokens.iter().enumerate() {
            if abort() {
                target.reset_recurrent(gpu)?;
                return Err("Qwen4 native MTP prefill aborted".to_string());
            }
            let position = start_pos
                .checked_add(index)
                .ok_or_else(|| "Qwen4 native MTP prefill position overflow".to_string())?;
            let bundle = Self::bundle(target)?;
            let argmax = bundle
                .spec_capture_token(gpu, token)
                .map_err(|error| error.to_string())?;
            bundle
                .copy_spec_hidden_row_to(gpu, 0, pending)
                .map_err(|error| error.to_string())?;
            bundle
                .mtp_forward_token(gpu, token, Some(pending), position)
                .map_err(|error| error.to_string())?;
            first_token = Some(argmax);
        }
        Ok(first_token.expect("non-empty MTP prefill produced no seed"))
    }

    fn mtp_step(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        position: usize,
        seed: u32,
        _emitted: &[u32],
        k: usize,
        eos: u32,
        _grammar: Option<&mut dyn SpecGrammar>,
    ) -> Result<MtpWindow, String> {
        require_native_greedy(self.request.temp)?;
        if k > self.max_k {
            return Err(format!(
                "Qwen4 native MTP draft budget {k} exceeds configured K {}",
                self.max_k
            ));
        }
        self.ensure_resources(gpu, target)?;
        {
            let bundle = Self::bundle(target)?;
            let target_position = bundle.state.position;
            let mtp_position = bundle.mtp_position().map_err(|error| error.to_string())?;
            if target_position != position || mtp_position != position {
                return Err(format!(
                    "Qwen4 MTP step position mismatch: target={}, mtp={}, position={position}",
                    target_position, mtp_position
                ));
            }
        }
        let mut snapshot = {
            let bundle = Self::bundle(target)?;
            Some(
                bundle
                    .mtp_snapshot(gpu)
                    .map_err(|error| error.to_string())?,
            )
        };
        let result = (|| -> Result<MtpWindow, String> {
            let mut drafts = Vec::with_capacity(k);
            let mut input = seed;
            for index in 0..k {
                let hidden = if index == 0 {
                    Some(self.pending_hidden()?)
                } else {
                    None
                };
                let token_position = position
                    .checked_add(index)
                    .ok_or_else(|| "Qwen4 MTP step position overflow".to_string())?;
                input = Self::bundle(target)?
                    .mtp_forward_token(gpu, input, hidden, token_position)
                    .map_err(|error| error.to_string())?;
                drafts.push(input);
            }
            let mut block = Vec::with_capacity(k + 1);
            block.push(seed);
            block.extend_from_slice(&drafts);
            let picks = Self::bundle(target)?;
            let pending_hidden = self
                .pending_hidden
                .as_ref()
                .ok_or_else(|| "Qwen4 native MTP pending hidden is not allocated".to_string())?;
            let scratch = self
                .scratch
                .as_mut()
                .ok_or_else(|| "Qwen4 native MTP verify scratch is not allocated".to_string())?;
            let target_picks = picks
                .verify_block(gpu, &block, position, scratch.as_mut(), None)
                .map_err(|error| error.to_string())?;
            let acceptance = accept_native_greedy(&drafts, &target_picks, Some(eos))?;
            let target_accept_len = acceptance.target_commit_accept_len();
            let full_accept = target_accept_len == k;
            let target_scratch = scratch
                .as_any_mut()
                .downcast_mut::<Qwen4SpecScratch>()
                .ok_or("Qwen4 native MTP target scratch type changed")?;
            let target_snapshot = target_scratch
                .target_snapshot
                .ok_or("Qwen4 native MTP target snapshot disappeared")?;

            // Keep both pre-window tickets active until every replay and hidden
            // copy succeeds. A retained restore lets the outer rollback repair
            // both owners if either side's GPU work fails.
            if !full_accept {
                picks
                    .restore_retain(gpu, target_snapshot)
                    .map_err(|error| error.to_string())?;
                picks
                    .spec_forward_rows(gpu, &block[..target_accept_len + 1], true)
                    .map(|_| ())
                    .map_err(|error| error.to_string())?;
            }
            let mtp_ticket = snapshot
                .as_ref()
                .copied()
                .expect("MTP snapshot remains active until transaction commit");
            if full_accept && k > 0 {
                let last_draft = *drafts
                    .last()
                    .ok_or_else(|| "Qwen4 native MTP full accept has no final draft".to_string())?;
                let last_position = position
                    .checked_add(k)
                    .ok_or_else(|| "Qwen4 MTP step position overflow".to_string())?;
                picks
                    .mtp_forward_token(gpu, last_draft, None, last_position)
                    .map_err(|error| error.to_string())?;
            } else {
                picks
                    .mtp_restore_retain(gpu, mtp_ticket)
                    .map_err(|error| error.to_string())?;
                for (index, &token) in block[..target_accept_len + 1].iter().enumerate() {
                    let hidden = if index == 0 {
                        Some(pending_hidden)
                    } else {
                        None
                    };
                    let token_position = position
                        .checked_add(index)
                        .ok_or_else(|| "Qwen4 MTP step position overflow".to_string())?;
                    picks
                        .mtp_forward_token(gpu, token, hidden, token_position)
                        .map_err(|error| error.to_string())?;
                }
            }
            let committed_end = position
                .checked_add(target_accept_len + 1)
                .ok_or_else(|| "Qwen4 MTP commit position overflow".to_string())?;
            if picks.state.position != committed_end
                || picks.mtp_position().map_err(|error| error.to_string())? != committed_end
            {
                return Err(format!(
                    "Qwen4 native MTP transaction ended at target={} mtp={}, expected {committed_end}",
                    picks.state.position,
                    picks.mtp_position().map_err(|error| error.to_string())?
                ));
            }
            picks
                .copy_spec_hidden_row_to(gpu, acceptance.pending_hidden_row(), pending_hidden)
                .map_err(|error| error.to_string())?;

            let window = acceptance_to_window(acceptance)?;
            // All fallible GPU operations are complete. Validate both tickets
            // before invalidating either arena, then perform the no-copy commit
            // boundary and clear the target scratch ticket.
            picks
                .validate_commit(target_snapshot)
                .map_err(|error| error.to_string())?;
            picks
                .mtp_validate_commit(mtp_ticket)
                .map_err(|error| error.to_string())?;
            picks.commit_validated(target_snapshot);
            picks.mtp_commit_validated(mtp_ticket);
            target_scratch.target_snapshot = None;
            snapshot = None;
            Ok(window)
        })();
        if let Err(error) = &result {
            let mut rollback_errors = Vec::new();
            let target_ticket = match self.scratch.as_mut() {
                Some(scratch) => match scratch.as_any_mut().downcast_mut::<Qwen4SpecScratch>() {
                    Some(scratch) => scratch.target_snapshot.take(),
                    None => {
                        rollback_errors.push("target rollback scratch type changed".to_string());
                        None
                    }
                },
                None => None,
            };
            if let Some(ticket) = target_ticket {
                if let Err(rollback) = Self::bundle(target).and_then(|bundle| {
                    bundle
                        .restore(gpu, ticket)
                        .map_err(|restore| restore.to_string())
                }) {
                    rollback_errors.push(format!("target rollback failed: {rollback}"));
                }
            }
            if let Some(ticket) = snapshot.take() {
                if let Err(rollback) = Self::bundle(target).and_then(|bundle| {
                    bundle
                        .mtp_restore(gpu, ticket)
                        .map_err(|restore| restore.to_string())
                }) {
                    rollback_errors.push(format!("MTP rollback failed: {rollback}"));
                }
            }
            if !rollback_errors.is_empty() {
                return Err(format!(
                    "{error}; rollback failed: {}",
                    rollback_errors.join("; ")
                ));
            }
        }
        result
    }

    fn mtp_forced_advance(
        &mut self,
        gpu: &mut Gpu,
        target: &mut dyn SpecTarget,
        tokens: &[u32],
        start_pos: usize,
        abort: &dyn Fn() -> bool,
    ) -> Result<bool, String> {
        if tokens.is_empty() {
            return Ok(true);
        }
        if abort() {
            return Ok(true);
        }
        self.ensure_resources(gpu, target)?;
        let pending = self.pending_hidden()?;
        for (index, &token) in tokens.iter().enumerate() {
            if abort() {
                return Ok(true);
            }
            let position = start_pos
                .checked_add(index)
                .ok_or_else(|| "Qwen4 native MTP forced position overflow".to_string())?;
            let bundle = Self::bundle(target)?;
            bundle
                .spec_capture_token(gpu, token)
                .map_err(|error| error.to_string())?;
            bundle
                .copy_spec_hidden_row_to(gpu, 0, pending)
                .map_err(|error| error.to_string())?;
            bundle
                .mtp_forward_token(gpu, token, Some(pending), position)
                .map_err(|error| error.to_string())?;
        }
        Ok(true)
    }

    fn mtp_reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        if let Some(scratch) = self.scratch.as_mut() {
            if let Some(scratch) = scratch.as_any_mut().downcast_mut::<Qwen4SpecScratch>() {
                scratch.target_snapshot = None;
            }
        }
        if let Some(hidden) = self.pending_hidden.as_ref() {
            gpu.hip
                .memset(&hidden.buf, 0, hidden.buf.size())
                .map_err(|error| format!("Qwen4 native MTP pending reset: {error}"))?;
        }
        Ok(())
    }

    fn mtp_free(self: Box<Self>, gpu: &mut Gpu) {
        let Self {
            scratch,
            pending_hidden,
            ..
        } = *self;
        if let Some(scratch) = scratch {
            scratch.free(gpu);
        }
        if let Some(hidden) = pending_hidden {
            let _ = gpu.free_tensor(hidden);
        }
    }

    fn k(&self) -> usize {
        self.max_k
    }

    fn proposal_capacity(&self) -> usize {
        self.max_k
    }

    fn ctx_capacity(&self) -> usize {
        self.ctx_capacity
    }

    fn requires_greedy(&self) -> bool {
        true
    }

    fn configure_request(&mut self, cfg: SpecRequestConfig) {
        self.request = cfg;
    }

    fn supports_temp_verify(&self) -> bool {
        false
    }
}

/// Build the generic runtime adapter around the native Qwen4 GPU MTP core.
pub fn build_qwen4_mtp_speculator(max_k: usize, ctx_capacity: usize) -> Box<dyn Speculator> {
    Box::new(MtpSpeculator::new(Qwen4MtpDrafter::new(
        max_k,
        ctx_capacity,
    )))
}

/// Convert an equation-level MTP error into the erased runtime error type.
pub fn mtp_error(error: MtpError) -> String {
    format!("Qwen4 native MTP: {error}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mtp::{MtpQsaGeometry, Qwen4MtpState};

    #[test]
    fn native_count_includes_seed_and_rollback_excludes_it() {
        let result = accept_native_greedy(&[10, 11], &[10, 99, 100], None).unwrap();
        assert_eq!(result.committed, vec![10, 99]);
        assert_eq!(result.accepted_drafts, 1);
        assert_eq!(result.drafts_generated, 2);
        assert_eq!(result.num_accepted_tokens, 2);
        assert_eq!(result.rollback_accept_len, 1);
        assert_eq!(result.next_seed, 99);
        assert!(!result.hit_eos);

        let window = acceptance_to_window(result).unwrap();
        assert_eq!(window.committed, vec![10, 99]);
        assert_eq!(window.accepted, 1);
        assert_eq!(window.drafts_generated, 2);
        let step = window_to_spec_step(window).unwrap();
        assert_eq!(step.emit.as_slice(), &[10, 99]);
        assert_eq!(step.next_seed, 99);
        assert_eq!(step.proposed, 2);
        assert_eq!(step.accepted, 1);
    }

    #[test]
    fn native_zero_partial_full_and_eos_counts_are_explicit() {
        let zero = accept_native_greedy(&[], &[42], None).unwrap();
        assert_eq!(zero.committed, vec![42]);
        assert_eq!(zero.num_accepted_tokens, 1);
        assert_eq!(zero.rollback_accept_len, 0);

        let partial = accept_native_greedy(&[10, 11], &[10, 12, 100], None).unwrap();
        assert_eq!(partial.num_accepted_tokens, 2);
        assert_eq!(partial.rollback_accept_len, 1);

        let full = accept_native_greedy(&[10, 11], &[10, 11, 12], None).unwrap();
        assert_eq!(full.committed, vec![10, 11, 12]);
        assert_eq!(full.num_accepted_tokens, 3);
        assert_eq!(full.rollback_accept_len, 2);

        let eos = accept_native_greedy(&[10, 99], &[10, 99, 12], Some(99)).unwrap();
        assert_eq!(eos.committed, vec![10, 99]);
        assert_eq!(eos.accepted_drafts, 2);
        assert_eq!(eos.num_accepted_tokens, 3);
        assert_eq!(eos.rollback_accept_len, 2);
        assert!(eos.hit_eos);

        assert!(native_rollback_accept_len(0).is_err());
        assert!(require_native_greedy(-0.0).is_ok());
        assert!(require_native_greedy(1.0e-6).is_ok());
        assert!(require_native_greedy(1.0e-5).is_err());
        assert!(require_native_greedy(f32::INFINITY).is_err());
        assert!(require_native_greedy(f32::NEG_INFINITY).is_err());
        assert!(require_native_greedy(f32::NAN).is_err());
    }

    #[test]
    fn accepted_eos_stays_pending_for_terminal_flush() {
        let accepted = accept_native_greedy(&[10, 99], &[10, 99, 12], Some(99)).unwrap();
        assert!(accepted.accepted_eos());
        assert_eq!(accepted.target_commit_accept_len(), 1);
        assert_eq!(accepted.pending_hidden_row(), 1);
        assert_eq!(
            committed_target_positions(7, accepted.target_commit_accept_len()).unwrap(),
            vec![7]
        );
        assert_eq!(
            native_commit_position(7, accepted.target_commit_accept_len()).unwrap(),
            8
        );

        let bonus = accept_native_greedy(&[10, 11], &[10, 99, 12], Some(99)).unwrap();
        assert!(!bonus.accepted_eos());
        assert_eq!(bonus.target_commit_accept_len(), 1);
        assert_eq!(bonus.pending_hidden_row(), 1);
    }

    #[test]
    fn zero_draft_replays_seed_before_terminal_flush() {
        let zero = accept_native_greedy(&[], &[42], Some(99)).unwrap();
        assert_eq!(zero.target_commit_accept_len(), 0);
        assert_eq!(zero.pending_hidden_row(), 0);
        assert_eq!(
            committed_target_positions(7, zero.target_commit_accept_len() + 1).unwrap(),
            vec![7]
        );
        assert_eq!(
            native_commit_position(7, zero.target_commit_accept_len() + 1).unwrap(),
            8
        );
    }

    #[test]
    fn native_zero_one_all_acceptance_advances_mtp_state_in_lockstep() {
        let cases = [
            (vec![], vec![42], 0usize),
            (vec![10], vec![10, 42], 1usize),
            (vec![10, 11], vec![10, 11, 42], 2usize),
        ];
        for (drafts, target_picks, expected_accept_len) in cases {
            let acceptance = accept_native_greedy(&drafts, &target_picks, None).unwrap();
            assert_eq!(acceptance.target_commit_accept_len(), expected_accept_len);
            let consumed = acceptance.target_commit_accept_len() + 1;
            let expected_position = native_commit_position(7, consumed).unwrap();
            let mut state = Qwen4MtpState::new(MtpQsaGeometry {
                q_heads: 2,
                kv_heads: 1,
                head_dim: 2,
                index_heads: 1,
                index_dim: 2,
                compress_ratio: 2,
                budget: 4,
                max_seq_len: 16,
                rotary_dim: 2,
                rope_theta: 10_000,
            })
            .unwrap();
            state.position = 7;
            state.qsa.position = 7;
            let snapshot = state.begin_transaction();
            state.forced_advance(consumed).unwrap();
            state.commit(snapshot).unwrap();
            assert_eq!(state.position, expected_position);
            assert_eq!(state.qsa.position, expected_position);
            assert_eq!(state.step_index, consumed);
        }
    }

    #[test]
    fn compact_native_selection_keeps_only_target_aligned_prefix() {
        let mut state = Qwen4MtpState::new(MtpQsaGeometry {
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            index_heads: 1,
            index_dim: 2,
            compress_ratio: 2,
            budget: 4,
            max_seq_len: 16,
            rotary_dim: 2,
            rope_theta: 10_000,
        })
        .unwrap();
        state.qsa.position = 6;
        state.qsa.selected_indices = vec![0, 1, 3, 5];
        compact_native_qsa_selection(&mut state, 1, 3).unwrap();
        assert_eq!(state.qsa.selected_indices, vec![0, 1, 3]);
        assert_eq!(committed_target_positions(1, 3).unwrap(), vec![1, 2, 3]);
        assert_eq!(native_commit_position(1, 3).unwrap(), 4);
    }

    #[test]
    fn malformed_target_picks_are_rejected_before_acceptance() {
        assert!(accept_native_greedy(&[2], &[], None).is_err());
        assert!(accept_native_greedy(&[2, 3], &[2, 3], None).is_err());
        assert!(acceptance_to_window(NativeMtpAcceptance {
            committed: vec![4],
            accepted_drafts: 2,
            drafts_generated: 1,
            num_accepted_tokens: 3,
            rollback_accept_len: 2,
            next_seed: 4,
            hit_eos: false,
        })
        .is_err());
    }
}
