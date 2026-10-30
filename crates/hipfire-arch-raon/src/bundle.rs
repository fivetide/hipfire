// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! RaonBundle (config, weights, state, scratch, eos). Owner: slice A/I.

use rdna_compute::Gpu;

use crate::config::RaonConfig;
use crate::state::{RaonPrefillScratch, RaonState};
use crate::weights::RaonWeights;

/// Upper bound of token rows one batched prefill call stages (the scratch is
/// sized for this, so a longer prompt is processed in chunks of at most this
/// many rows).
pub const PREFILL_SCRATCH_ROWS: usize = 512;

/// Everything one loaded Raon-1-mini model owns on the device.
pub struct RaonBundle {
    pub config: RaonConfig,
    /// Immutable weights.
    pub weights: RaonWeights,
    /// Mutable per-session state (RDA recurrent state, conv history, MLA latent caches).
    pub state: RaonState,
    /// Bounded prefill scratch (`PREFILL_SCRATCH_ROWS` rows).
    pub scratch: RaonPrefillScratch,
    /// EOS token ids (`generation_config.json`: `<|im_end|>`-class ids).
    pub eos: Vec<u32>,
}

impl RaonBundle {
    /// Whether `token` ends generation.
    pub fn is_eos(&self, token: u32) -> bool {
        self.eos.contains(&token)
    }

    /// Return every device buffer the bundle owns (scratch, state, weights).
    pub fn free_all(self, gpu: &mut Gpu) {
        let RaonBundle {
            config: _,
            weights,
            state,
            scratch,
            eos: _,
        } = self;
        if let Err(e) = scratch.free_gpu(gpu) {
            eprintln!("raon1_mini: scratch free: {e}");
        }
        if let Err(e) = state.free_gpu(gpu) {
            eprintln!("raon1_mini: state free: {e}");
        }
        weights.free_gpu(gpu);
    }
}
