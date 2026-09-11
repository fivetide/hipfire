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

use crate::mtp::{MtpError, Qwen4MtpState};
use hipfire_runtime::spec::{accept_greedy_prefix, MtpWindow, SpecStep};

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
        assert!(require_native_greedy(0.0).is_ok());
        assert!(require_native_greedy(0.7).is_err());
        assert!(require_native_greedy(f32::NAN).is_err());
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
