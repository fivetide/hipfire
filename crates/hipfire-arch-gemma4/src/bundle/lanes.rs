// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Gemma 4's engine lanes ([`MeshLanes`]): one extra KV set per request that
//! decodes beside others.
//!
//! The lanes' K/V live in one lane-major arena per cache slot: lane `i`
//! owns the `max_seq` rows from row `i * max_seq`. That is the layout
//! `KvRoute::Lanes` attends with `lane_capacity == max_seq` (batch row `r`
//! is lane `r`), so one forward over the lanes' rows reads the weights once
//! and reads and writes every lane in place ([`forward::forward_lane_rows`],
//! the mixed-row epoch). Its rows are not bit-identical to the one-row
//! decode (batched GEMMs, grouped MoE, batched head). Rows of the row-exact
//! class (`serve.batch_exact`), epochs of fewer than [`MIXED_MIN_ROWS`]
//! shared rows, and loads without batched rows
//! run the one-row decode instead (whole-step): the eager decode the legacy
//! loop's captured graph replays byte-identically, with the lane's caches
//! (non-owning views of its region) swapped in for the step. The resident
//! caches keep their buffers, so the captured decode graph stays bound to
//! them.
//!
//! A lane is never a conversation record ([`MeshLanes::keeps_records`]):
//! decoded rows are not reused (`bundle.rs`), so a finished lane is freed.

use super::mesh_family::device;
use super::{q8_row_bytes, Gemma4Bundle};
use crate::config::Gemma4Config;
use crate::forward;
use crate::gemma4::{Gemma4State, Gemma4Weights};
use hipfire_config::LaneRowExact;
use hipfire_runtime::llama::KvCache;
use hipfire_runtime::mesh_family::{DecodeOut, MeshLanes};
use hipfire_runtime::multi_gpu::executor::Result;
use hipfire_runtime::multi_gpu::Gpus;
use rdna_compute::{DType, Gpu, GpuTensor};

/// Fewest shared rows one lane forward runs; fewer run one-row decodes. A
/// row routes its own top-8 of 128 experts, so the expert reads grow with
/// the rows and only the dense part is shared. halo gfx1151, 26B-A4B mq4,
/// `hipfire bench` stateless, TPOT p50: 2 rows 42.7 ms in one forward vs
/// 2 x 20 ms one-row; 4 rows 56.7 ms vs 4 x 20 ms (HIPF-545).
const MIXED_MIN_ROWS: usize = 3;

struct Lane {
    /// Views of this lane's region of the arenas; never freed.
    kv_sliding: KvCache,
    kv_full: KvCache,
    n_tokens: usize,
    /// The request holding the lane; `None` = free.
    tag: Option<u64>,
}

/// The staged lanes, their K/V arenas and the device argmax slot their rows
/// share.
pub(super) struct Gemma4Lanes {
    /// Lane-major arenas of `lanes.len() * max_seq` rows per slot.
    arena_sliding: KvCache,
    arena_full: KvCache,
    lanes: Vec<Lane>,
    top1: GpuTensor,
    /// Sampled requests may take a lane (`serve.batch_sampled`).
    sampled: bool,
    /// The load runs batched rows, so an epoch of [`MIXED_MIN_ROWS`]+ shared
    /// rows is one forward ([`forward::forward_lane_rows`]).
    mixed: bool,
    /// `serve.batch_exact`: which rows keep the one-row decode.
    exact: LaneRowExact,
}

impl Gemma4Lanes {
    pub(super) fn free_gpu(self, gpu: &mut Gpu) {
        let _ = self.arena_sliding.free_gpu(gpu);
        let _ = self.arena_full.free_gpu(gpu);
        let _ = gpu.free_tensor(self.top1);
    }
}

/// A lane-major arena for `n` KV sets shaped like `kv` (Q8 caches only).
fn alloc_arena(gpu: &mut Gpu, kv: &KvCache, n: usize) -> Result<KvCache> {
    let rows = n * kv.max_seq;
    KvCache::new_gpu_q8(gpu, kv.k_gpu.len(), kv.n_kv_heads, kv.head_dim, rows)
        .map_err(|e| format!("gemma4 lane KV: {e:?}"))
}

/// Lane `lane`'s region of `arena` as a non-owning KV set shaped like `kv`.
fn lane_view(gpu: &mut Gpu, arena: &KvCache, kv: &KvCache, lane: usize) -> Result<KvCache> {
    let lane_bytes = kv.max_seq * q8_row_bytes(kv);
    // The arena tensors count f32 words.
    if !lane_bytes.is_multiple_of(4) {
        return Err(format!(
            "gemma4 lane KV: {lane_bytes} lane bytes are not whole words"
        ));
    }
    let (offset, len) = (lane * lane_bytes / 4, lane_bytes / 4);
    let views = |tensors: &[GpuTensor]| {
        tensors
            .iter()
            .map(|t| t.try_sub_offset(offset, len))
            .collect::<std::result::Result<Vec<_>, String>>()
    };
    // Zero slots allocate nothing; the views replace them.
    let mut view = KvCache::new_gpu_q8(gpu, 0, kv.n_kv_heads, kv.head_dim, kv.max_seq)
        .map_err(|e| format!("gemma4 lane KV view: {e:?}"))?;
    view.k_gpu = views(&arena.k_gpu)?;
    view.v_gpu = views(&arena.v_gpu)?;
    Ok(view)
}

/// Copy the first `rows` positions of every K/V slot of `src` into `dst`.
fn copy_rows(gpu: &mut Gpu, dst: &KvCache, src: &KvCache, rows: usize) -> Result<()> {
    let bytes = rows * q8_row_bytes(src);
    for (d, s) in dst
        .k_gpu
        .iter()
        .chain(&dst.v_gpu)
        .zip(src.k_gpu.iter().chain(&src.v_gpu))
    {
        gpu.memcpy_dtod_at_auto(&d.buf, 0, &s.buf, 0, bytes)
            .map_err(|e| format!("gemma4 lane KV copy: {e:?}"))?;
    }
    Ok(())
}

impl Gemma4Bundle {
    /// Stage `n` engine lanes, each a KV set shaped like the resident caches
    /// in lane-major arenas; `sampled` admits sampled requests to them. Q8
    /// caches only (a park copies Q8 rows).
    pub fn stage_engine_lanes(&mut self, gpu: &mut Gpu, n: usize, sampled: bool) -> Result<()> {
        if self.lanes.is_some() {
            return Err("gemma4 engine lanes are already staged".into());
        }
        let (sliding, full) = (&self.state.kv_sliding, &self.state.kv_full);
        if !(sliding.quant_q8 && full.quant_q8) {
            return Err("gemma4 engine lanes need Q8 KV caches".into());
        }
        let arena_sliding = alloc_arena(gpu, sliding, n)?;
        let arena_full = match alloc_arena(gpu, full, n) {
            Ok(arena) => arena,
            Err(error) => {
                let _ = arena_sliding.free_gpu(gpu);
                return Err(error);
            }
        };
        let top1 = match gpu.zeros(&[std::mem::size_of::<u32>()], DType::Raw) {
            Ok(top1) => top1,
            Err(error) => {
                let _ = arena_sliding.free_gpu(gpu);
                let _ = arena_full.free_gpu(gpu);
                return Err(format!("gemma4 lane argmax slot: {error:?}"));
            }
        };
        // The lane attention sizes its LDS by one lane's rows.
        let head_dim = sliding.head_dim.max(full.head_dim);
        let lds_ok =
            gpu.attention_q8_0_kv_independent_max_lane_capacity(head_dim) >= sliding.max_seq;
        let mut staged = Gemma4Lanes {
            arena_sliding,
            arena_full,
            lanes: Vec::with_capacity(n),
            top1,
            sampled,
            mixed: self.prefill_rows > 1 && (2..=64).contains(&n) && lds_ok,
            exact: hipfire_config::qwen4_lane_row_exact(),
        };
        for i in 0..n {
            let lane = lane_view(gpu, &staged.arena_sliding, sliding, i).and_then(|kv_sliding| {
                Ok(Lane {
                    kv_sliding,
                    kv_full: lane_view(gpu, &staged.arena_full, full, i)?,
                    n_tokens: 0,
                    tag: None,
                })
            });
            match lane {
                Ok(lane) => staged.lanes.push(lane),
                Err(error) => {
                    staged.free_gpu(gpu);
                    return Err(error);
                }
            }
        }
        self.lanes = Some(staged);
        Ok(())
    }

    /// Whether [`Self::stage_engine_lanes`] staged lanes on this load.
    pub fn engine_lanes_staged(&self) -> bool {
        self.lanes.is_some()
    }

    /// The staged lanes' `sampled` flag; `None` without lanes.
    pub fn engine_lanes_sampled(&self) -> Option<bool> {
        self.lanes.as_ref().map(|staged| staged.sampled)
    }
}

impl MeshLanes for Gemma4Bundle {
    fn lane_capacity(&self) -> usize {
        self.lanes.as_ref().map_or(0, |staged| staged.lanes.len())
    }

    fn free_lanes(&self) -> usize {
        self.lanes.as_ref().map_or(0, |staged| {
            staged
                .lanes
                .iter()
                .filter(|lane| lane.tag.is_none())
                .count()
        })
    }

    /// One epoch steps every decoding lane.
    fn lane_rows(&self) -> usize {
        self.lane_capacity()
    }

    fn keeps_records(&self) -> bool {
        false
    }

    /// Copy the resident rows into a free lane; the prompt's pool snapshots
    /// publish now (as on a client cancel), and the resident state resets.
    fn park(
        &mut self,
        gpus: &mut Gpus,
        tag: u64,
        prompt: &[u32],
        ran: &[u32],
        _seed: u32,
        _remaining: usize,
    ) -> Result<()> {
        let gpu = device(gpus)?;
        let position = prompt.len() + ran.len();
        if self.state.n_tokens != position {
            return Err(format!(
                "park: the resident state sits at {}, not after its {position} tokens",
                self.state.n_tokens
            ));
        }
        let staged = self.lanes.as_mut().ok_or("no gemma4 engine lanes staged")?;
        let lane = staged
            .lanes
            .iter_mut()
            .find(|lane| lane.tag.is_none())
            .ok_or("no free gemma4 engine lane")?;
        copy_rows(gpu, &lane.kv_sliding, &self.state.kv_sliding, position)?;
        copy_rows(gpu, &lane.kv_full, &self.state.kv_full, position)?;
        lane.n_tokens = position;
        lane.tag = Some(tag);
        if let Some(session) = self.session.as_mut() {
            session.publish_captures();
            session.forget_live();
        }
        self.state.reset();
        Ok(())
    }

    fn unpark(&mut self, _gpus: &mut Gpus, tag: u64, _tokens: &[u32]) -> Result<()> {
        Err(format!("gemma4 lane {tag} is not a conversation record"))
    }

    fn step_lanes(&mut self, gpus: &mut Gpus, rows: &[(u64, u32, bool)]) -> Result<Vec<DecodeOut>> {
        let gpu = device(gpus)?;
        let Gemma4Bundle {
            config,
            weights,
            state,
            lanes,
            ..
        } = self;
        let staged = lanes.as_mut().ok_or("no gemma4 engine lanes staged")?;
        let lane_of = rows
            .iter()
            .map(|&(tag, ..)| {
                staged
                    .lanes
                    .iter()
                    .position(|lane| lane.tag == Some(tag))
                    .ok_or_else(|| format!("no gemma4 lane holds request {tag}"))
            })
            .collect::<Result<Vec<_>>>()?;
        // `logits` rows are sampled; the rest are greedy (argmax).
        // ponytail: `seeded` keeps every row exact, unseeded sampled ones
        // too: rows carry no seed flag. Pass it when unseeded rows matter.
        let mode = staged.exact;
        let exact = |logits: bool| match mode {
            LaneRowExact::Off => false,
            LaneRowExact::Greedy => !logits,
            LaneRowExact::GreedySeeded => true,
        };
        let shared: Vec<usize> = (0..rows.len()).filter(|&i| !exact(rows[i].2)).collect();
        let mut outs: Vec<Option<DecodeOut>> = rows.iter().map(|_| None).collect();
        if staged.mixed && shared.len() >= MIXED_MIN_ROWS {
            let lane_rows: Vec<_> = shared
                .iter()
                .map(|&i| (lane_of[i], rows[i].1, rows[i].2))
                .collect();
            let mixed = step_mixed(config, weights, state, gpu, staged, &lane_rows)?;
            for (&i, out) in shared.iter().zip(mixed) {
                outs[i] = Some(out);
            }
        }
        for (i, &(_, token, logits)) in rows.iter().enumerate() {
            if outs[i].is_none() {
                let out = step_one(
                    config, weights, state, gpu, staged, lane_of[i], token, logits,
                )?;
                outs[i] = Some(out);
            }
        }
        Ok(outs.into_iter().flatten().collect())
    }

    fn free_lane(&mut self, tag: u64) {
        if let Some(lane) = self
            .lanes
            .as_mut()
            .and_then(|staged| staged.lanes.iter_mut().find(|lane| lane.tag == Some(tag)))
        {
            lane.tag = None;
            lane.n_tokens = 0;
        }
    }
}

/// One forward over `rows` (`(lane, token, logits)`, each lane once): row
/// `r` of it is lane `r`, lanes without a row are masked out. Returns the
/// outputs in `rows` order and advances each lane.
fn step_mixed(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    staged: &mut Gemma4Lanes,
    rows: &[(usize, u32, bool)],
) -> Result<Vec<DecodeOut>> {
    let n = rows.iter().map(|r| r.0).max().map_or(0, |lane| lane + 1);
    let (mut tokens, mut positions, mut mask) = (vec![0u32; n], vec![0usize; n], 0u64);
    for &(lane, token, _) in rows {
        tokens[lane] = token;
        positions[lane] = staged.lanes[lane].n_tokens;
        mask |= 1 << lane;
    }
    let (mut top1, mut logits) = (Vec::new(), Vec::new());
    let want_top1 = rows.iter().any(|r| !r.2);
    let want_logits = rows.iter().any(|r| r.2);
    forward::forward_lane_rows(
        cfg,
        weights,
        state,
        gpu,
        &tokens,
        &forward::LaneRows {
            kv_sliding: &staged.arena_sliding,
            kv_full: &staged.arena_full,
            positions: &positions,
            active_mask: mask,
        },
        want_top1.then_some(&mut top1),
        want_logits.then_some(&mut logits),
    )?;
    let vocab = cfg.vocab_size;
    Ok(rows
        .iter()
        .map(|&(lane, _, sampled)| {
            staged.lanes[lane].n_tokens += 1;
            if sampled {
                DecodeOut::Logits(logits[lane * vocab..(lane + 1) * vocab].to_vec())
            } else {
                DecodeOut::Top1(top1[lane])
            }
        })
        .collect())
}

/// Lane `lane` runs `token` as its own one-token forward (whole-step), with
/// its caches swapped into the resident state for the step.
#[allow(clippy::too_many_arguments)]
fn step_one(
    cfg: &Gemma4Config,
    weights: &Gemma4Weights,
    state: &mut Gemma4State,
    gpu: &mut Gpu,
    staged: &mut Gemma4Lanes,
    lane: usize,
    token: u32,
    logits: bool,
) -> Result<DecodeOut> {
    let resident = state.n_tokens;
    let lane = &mut staged.lanes[lane];
    std::mem::swap(&mut state.kv_sliding, &mut lane.kv_sliding);
    std::mem::swap(&mut state.kv_full, &mut lane.kv_full);
    let position = lane.n_tokens as u32;
    let out =
        forward::decode_step_device(cfg, weights, state, gpu, token, position).and_then(|()| {
            if logits {
                return gpu
                    .download_f32(&state.logits)
                    .map(DecodeOut::Logits)
                    .map_err(|e| format!("gemma4 lane logits: {e:?}"));
            }
            gpu.argmax_f32_batched(&state.logits, &staged.top1, cfg.vocab_size, 1)
                .and_then(|()| gpu.download_raw_bytes(&staged.top1))
                .map_err(|e| format!("gemma4 lane argmax: {e:?}"))
                .and_then(|bytes| {
                    bytes
                        .get(..4)
                        .map(|b| DecodeOut::Top1(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
                        .ok_or_else(|| "gemma4 lane argmax readback is short".to_string())
                })
        });
    std::mem::swap(&mut state.kv_sliding, &mut lane.kv_sliding);
    std::mem::swap(&mut state.kv_full, &mut lane.kv_full);
    state.n_tokens = resident;
    let out = out?;
    lane.n_tokens += 1;
    Ok(out)
}
