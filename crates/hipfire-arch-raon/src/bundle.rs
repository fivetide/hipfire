// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! RaonBundle (config, weights, state, scratch, work, eos) and the generation
//! entry points built on the forward. Owner: slice A/I.

use rdna_compute::{Gpu, GpuTensor};

use crate::config::RaonConfig;
use crate::forward::{forward_rows, forward_token, RaonWork};
use crate::state::{RaonPrefillScratch, RaonState, MLA_TILE_ROWS};
use crate::weights::RaonWeights;

/// Upper bound of token rows one batched prefill call stages (the scratch is
/// sized for this, so a longer prompt is processed in chunks of at most this
/// many rows).
pub const PREFILL_SCRATCH_ROWS: usize = 512;

/// Prefill route. `Batched` is the product route (N rows per module call);
/// `Serial` (one `forward_token` per prompt token) exists as the explicit
/// oracle/control and is never a default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMode {
    Serial,
    Batched,
}

impl PrefillMode {
    /// `HIPFIRE_RAON_PREFILL` (`serial` | `batched`); unset means batched.
    /// Anything else is a named error, never a silent fallback.
    pub fn from_env() -> Result<Self, String> {
        match hipfire_config::developer_var("HIPFIRE_RAON_PREFILL") {
            Err(_) => Ok(Self::Batched),
            Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                "" | "batched" => Ok(Self::Batched),
                "serial" => Ok(Self::Serial),
                other => Err(format!(
                    "HIPFIRE_RAON_PREFILL={other:?}: expected `serial` or `batched`"
                )),
            },
        }
    }
}

/// Activation precision of batched qt44 projections (`HIPFIRE_RAON_PREFILL_PRECISION`,
/// PLAN §3.6). `F16` (default) keeps FP16 activations at every qt44 GEMM;
/// `A4` admits the shared symmetric IU4/A8 activation-quantized prefill
/// routes, which are not bit-equivalent and stay opt-in pending review.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillPrecision {
    F16,
    A4,
}

impl PrefillPrecision {
    pub fn from_env() -> Result<Self, String> {
        match hipfire_config::developer_var("HIPFIRE_RAON_PREFILL_PRECISION") {
            Err(_) => Ok(Self::F16),
            Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
                "" | "f16" => Ok(Self::F16),
                "a4" => Ok(Self::A4),
                other => Err(format!(
                    "HIPFIRE_RAON_PREFILL_PRECISION={other:?}: expected `f16` or `a4`"
                )),
            },
        }
    }

    /// Resolve the policy onto the device's feature flags once, at load.
    /// `F16` turns off every activation-quantized prefill route for this
    /// process's device; `A4` leaves the device defaults untouched.
    fn apply(self, gpu: &mut Gpu) {
        if self == Self::F16 {
            let mut flags = (*gpu.flags).clone();
            flags.iu4_prefill = Some(false);
            flags.a8_prefill = false;
            flags.gfx11_a8_prefill = false;
            gpu.flags = std::sync::Arc::new(flags);
        }
    }
}

/// Everything one loaded Raon-1-mini model owns on the device.
pub struct RaonBundle {
    pub config: RaonConfig,
    /// Immutable weights.
    pub weights: RaonWeights,
    /// Mutable per-session state (RDA recurrent state, conv history, MLA latent caches).
    pub state: RaonState,
    /// Bounded prefill scratch (`PREFILL_SCRATCH_ROWS` rows).
    pub scratch: RaonPrefillScratch,
    /// Module work buffers (mHC controls, RDA, MoE, dense FFN) matching `scratch`.
    pub work: RaonWork,
    /// EOS token ids (`generation_config.json`: `<|im_end|>`-class ids).
    pub eos: Vec<u32>,
    prefill_mode: PrefillMode,
}

impl RaonBundle {
    /// Assemble the bundle from its loaded parts, allocating the module work
    /// buffers and resolving `HIPFIRE_RAON_PREFILL` once. On error every part
    /// handed in is released.
    pub fn new(
        gpu: &mut Gpu,
        config: RaonConfig,
        weights: RaonWeights,
        state: RaonState,
        scratch: RaonPrefillScratch,
    ) -> Result<Self, String> {
        let built = PrefillMode::from_env()
            .and_then(|m| PrefillPrecision::from_env().map(|p| (m, p)))
            .and_then(|(m, p)| {
                p.apply(gpu);
                let mut work = RaonWork::new(gpu, &config, scratch.rows)?;
                if let Err(e) = work.mla.pack_absorbed(gpu, &config, &weights) {
                    work.free(gpu);
                    return Err(e);
                }
                Ok((m, work))
            });
        match built {
            Ok((prefill_mode, work)) => {
                let eos = config.eos_token_ids.clone();
                Ok(Self {
                    config,
                    weights,
                    state,
                    scratch,
                    work,
                    eos,
                    prefill_mode,
                })
            }
            Err(e) => {
                if let Err(fe) = scratch.free_gpu(gpu) {
                    eprintln!("raon1_mini: scratch free after assembly failure: {fe}");
                }
                if let Err(fe) = state.free_gpu(gpu) {
                    eprintln!("raon1_mini: state free after assembly failure: {fe}");
                }
                weights.free_gpu(gpu);
                Err(format!("raon1_mini: bundle assembly failed: {e}"))
            }
        }
    }

    /// Whether `token` ends generation.
    pub fn is_eos(&self, token: u32) -> bool {
        self.eos.contains(&token)
    }

    /// Prefill route selected at load (see [`PrefillMode::from_env`]).
    pub fn prefill_mode(&self) -> PrefillMode {
        self.prefill_mode
    }

    /// Override the load-time route (oracle/control use).
    pub fn set_prefill_mode(&mut self, mode: PrefillMode) {
        self.prefill_mode = mode;
    }

    /// Tokens consumed by the slot so far.
    pub fn n_tokens(&self) -> usize {
        self.state.n_tokens()
    }

    /// Context rows this slot reserved.
    pub fn max_seq(&self) -> usize {
        self.state.max_seq()
    }

    /// Logits of the last row of the last `prefill`/`decode_step`: F32
    /// `[vocab]` on the device holding BF16-rounded values.
    pub fn logits(&self) -> &GpuTensor {
        &self.scratch.logits
    }

    /// Append `tokens` to the slot starting at `n_tokens()`, in chunks of at
    /// most `scratch.rows`; the logits of the final token are left in
    /// [`logits`](Self::logits). A failure leaves the slot partially advanced:
    /// the caller resets it.
    pub fn prefill(&mut self, gpu: &mut Gpu, tokens: &[u32]) -> Result<(), String> {
        if tokens.is_empty() {
            return Err("raon1_mini: prefill of zero tokens".into());
        }
        let start = self.state.n_tokens();
        if start + tokens.len() > self.state.max_seq() {
            return Err(format!(
                "raon1_mini: {start} + {} tokens exceed the reserved context of {} rows",
                tokens.len(),
                self.state.max_seq()
            ));
        }
        let chunk = match self.prefill_mode {
            PrefillMode::Serial => 1,
            PrefillMode::Batched => self.scratch.rows.min(self.work.rows()).min(MLA_TILE_ROWS),
        };
        let n_chunks = tokens.len().div_ceil(chunk);
        for (ci, piece) in tokens.chunks(chunk).enumerate() {
            let pos = self.state.n_tokens() as u32;
            let last = ci + 1 == n_chunks;
            if chunk == 1 {
                forward_token(
                    &self.config,
                    &self.weights,
                    &mut self.state,
                    &mut self.scratch,
                    &mut self.work,
                    gpu,
                    piece[0],
                    pos,
                    last,
                )?;
            } else {
                forward_rows(
                    &self.config,
                    &self.weights,
                    &mut self.state,
                    &mut self.scratch,
                    &mut self.work,
                    gpu,
                    piece,
                    pos,
                    last,
                )?;
            }
        }
        Ok(())
    }

    /// One decode step: consume `token`, leave its logits in
    /// [`logits`](Self::logits).
    pub fn decode_step(&mut self, gpu: &mut Gpu, token: u32) -> Result<(), String> {
        let pos = self.state.n_tokens();
        if pos + 1 > self.state.max_seq() {
            return Err(format!(
                "raon1_mini: decode past the reserved context of {} rows",
                self.state.max_seq()
            ));
        }
        forward_token(
            &self.config,
            &self.weights,
            &mut self.state,
            &mut self.scratch,
            &mut self.work,
            gpu,
            token,
            pos as u32,
            true,
        )
    }

    /// Start a cold request (zero recurrent state, conv history, length).
    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), String> {
        self.state.reset(gpu)
    }

    /// Reset, then prefill `n_tokens` deterministic synthetic ids through the
    /// production [`prefill`](Self::prefill) route and wait for the device.
    /// Used by the loader's prefill benchmark.
    pub fn bench_prefill_rows(&mut self, gpu: &mut Gpu, n_tokens: usize) -> Result<(), String> {
        if n_tokens == 0 {
            return Err("raon1_mini: bench prefill of zero tokens".into());
        }
        self.reset(gpu)?;
        let vocab = self.config.vocab as u64;
        // Ids stay clear of the low special-token block and are well spread.
        let lo = 1024u64.min(vocab.saturating_sub(1));
        let span = vocab - lo;
        let tokens: Vec<u32> = (0..n_tokens as u64)
            .map(|i| (lo + (i.wrapping_mul(2654435761).wrapping_add(17)) % span) as u32)
            .collect();
        self.prefill(gpu, &tokens)?;
        gpu.hip
            .device_synchronize()
            .map_err(|e| format!("raon1_mini: bench prefill sync: {e:?}"))
    }

    /// Return every device buffer the bundle owns (work, scratch, state, weights).
    pub fn free_all(self, gpu: &mut Gpu) {
        let RaonBundle {
            config: _,
            weights,
            state,
            scratch,
            work,
            eos: _,
            prefill_mode: _,
        } = self;
        work.free(gpu);
        if let Err(e) = scratch.free_gpu(gpu) {
            eprintln!("raon1_mini: scratch free: {e}");
        }
        if let Err(e) = state.free_gpu(gpu) {
            eprintln!("raon1_mini: state free: {e}");
        }
        weights.free_gpu(gpu);
    }
}
