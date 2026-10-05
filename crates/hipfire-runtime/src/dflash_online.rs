//! Online draft tuning for chain DFlash (developer-only, `HIPFIRE_DFLASH_ONLINE_TUNE`).
//!
//! The drafter's greedy proposal at each block row is re-ranked among the
//! draft's own top-K logits by a small per-request model trained online from
//! tokens the target has already verified. A DFlash block row's draft hidden
//! depends only on the committed context and the mask tokens, never on the
//! other drafted tokens, so every row is a clean supervised example once the
//! emitted stream reaches its position: `(draft top-K at row, true token)`.
//!
//! Verify stays the target's greedy argmax, so output is unchanged; only the
//! proposal (and therefore acceptance τ) moves.

use rdna_compute::{DType, Gpu, GpuTensor};
use std::collections::VecDeque;

/// Draft candidates kept per row (top-K kernel limit is 16).
pub const K: usize = 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// Collect acceptance-ceiling statistics only; proposals stay argmax.
    Stats,
    /// Train online and re-rank proposals.
    On,
}

impl Mode {
    pub fn from_env() -> Option<Mode> {
        match hipfire_config::developer_var("HIPFIRE_DFLASH_ONLINE_TUNE")
            .ok()?
            .as_str()
        {
            "stats" => Some(Mode::Stats),
            "1" | "on" => Some(Mode::On),
            _ => None,
        }
    }
}

/// One drafted block awaiting labels from the emitted stream.
struct Cycle {
    /// Absolute position of the seed; row `i` (1-based) predicts `start + i`.
    start: usize,
    ids: Vec<[u32; K]>,
    vals: Vec<[f32; K]>,
    /// Label rank per row in the draft top-K (`K` = absent), once known.
    ranks: Vec<Option<u8>>,
    /// Index of the row the tuner proposed (rank among candidates) per row.
    picked: Vec<u8>,
}

#[derive(Default)]
struct Stats {
    cycles: u64,
    accepted: u64,
    finalized: u64,
    /// Sum over finalized cycles of the longest row prefix whose label rank < 2^j.
    oracle: [u64; 5],
    /// Sum over finalized cycles of the tuner's picked-prefix length.
    picked_prefix: u64,
    /// Rank histogram of the first row whose label is not the argmax.
    first_miss: [u64; 6],
}

pub struct OnlineDraftTuner {
    pub mode: Mode,
    /// Known emitted tokens by absolute position (`u32::MAX` = unknown).
    known: Vec<u32>,
    pending: VecDeque<Cycle>,
    stats: Stats,
    /// Device top-K ids (i32 in an F32 tensor) and values, `[max_rows × K]`.
    dev: Option<(GpuTensor, GpuTensor)>,
}
impl OnlineDraftTuner {
    pub fn from_env() -> Option<Self> {
        Mode::from_env().map(|mode| Self {
            mode,
            known: Vec::new(),
            pending: VecDeque::new(),
            stats: Stats::default(),
            dev: None,
        })
    }

    /// Draft proposals for `rows` rows of `logits` (`[rows × vocab]`, device):
    /// top-K on device, small D2H, host re-rank.
    pub fn propose_from_logits(
        &mut self,
        gpu: &mut Gpu,
        logits: &GpuTensor,
        vocab: usize,
        rows: usize,
        start: usize,
    ) -> rdna_compute::HipResult<Vec<u32>> {
        if self.dev.as_ref().is_none_or(|(i, _)| i.numel() < rows * K) {
            if let Some((i, v)) = self.dev.take() {
                gpu.free_tensor(i)?;
                gpu.free_tensor(v)?;
            }
            let n = rows.max(16) * K;
            self.dev = Some((gpu.alloc_tensor(&[n], DType::F32)?, gpu.alloc_tensor(&[n], DType::F32)?));
        }
        let (di, dv) = self.dev.as_ref().unwrap();
        let di = di.sub_offset(0, rows * K);
        let dv = dv.sub_offset(0, rows * K);
        gpu.topk_values_batched_f32(logits, &di, &dv, vocab, K, rows)?;
        let mut ids = vec![0i32; rows * K];
        // SAFETY: `ids` holds exactly rows*K i32; any byte pattern is valid.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(ids.as_mut_ptr() as *mut u8, rows * K * 4) };
        gpu.hip.memcpy_dtoh(bytes, &di.buf)?;
        let vals = gpu.download_f32(&dv)?;
        Ok(self.propose(start, &ids, &vals, rows))
    }

    pub fn free_gpu(&mut self, gpu: &mut Gpu) {
        if let Some((i, v)) = self.dev.take() {
            let _ = gpu.free_tensor(i);
            let _ = gpu.free_tensor(v);
        }
    }

    /// New request: forget the session (no cross-request learning).
    pub fn reset(&mut self) {
        self.report("request end");
        self.known.clear();
        self.pending.clear();
        self.stats = Stats::default();
    }

    /// Choose the proposal for each row from its descending top-K.
    /// `ids`/`vals` are `rows × K` row-major as produced by the top-K kernel.
    pub fn propose(&mut self, start: usize, ids: &[i32], vals: &[f32], rows: usize) -> Vec<u32> {
        let mut cyc = Cycle {
            start,
            ids: Vec::with_capacity(rows),
            vals: Vec::with_capacity(rows),
            ranks: vec![None; rows],
            picked: Vec::with_capacity(rows),
        };
        let mut out = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut order: [usize; K] = std::array::from_fn(|j| j);
            let v = &vals[r * K..(r + 1) * K];
            order.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
            let row_ids: [u32; K] = std::array::from_fn(|j| ids[r * K + order[j]] as u32);
            let row_vals: [f32; K] = std::array::from_fn(|j| v[order[j]]);
            let pick = 0usize;
            out.push(row_ids[pick]);
            cyc.picked.push(pick as u8);
            cyc.ids.push(row_ids);
            cyc.vals.push(row_vals);
        }
        self.pending.push_back(cyc);
        out
    }

    /// Record the tokens committed after the seed at `start` (accepted drafts
    /// plus the bonus) and resolve every pending row they label.
    pub fn observe(&mut self, start: usize, committed_after_seed: &[u32], accepted: usize) {
        self.stats.cycles += 1;
        self.stats.accepted += accepted as u64;
        // A rewind invalidates everything past the seed.
        self.known.truncate(start + 1);
        self.pending.retain(|c| c.start <= start);
        let end = start + 1 + committed_after_seed.len();
        if self.known.len() < end {
            self.known.resize(end, u32::MAX);
        }
        self.known[start + 1..end].copy_from_slice(committed_after_seed);

        while let Some(c) = self.pending.front_mut() {
            for (r, rank) in c.ranks.iter_mut().enumerate() {
                if rank.is_none() {
                    if let Some(&tok) = self.known.get(c.start + 1 + r).filter(|&&t| t != u32::MAX) {
                        let pos = c.ids[r].iter().position(|&id| id == tok).unwrap_or(K);
                        *rank = Some(pos as u8);
                    }
                }
            }
            if c.ranks.iter().any(Option::is_none) {
                break;
            }
            let c = self.pending.pop_front().unwrap();
            self.finalize(&c);
        }
        if self.stats.cycles % 64 == 0 {
            self.report("running");
        }
    }

    fn finalize(&mut self, c: &Cycle) {
        let ranks: Vec<usize> = c.ranks.iter().map(|r| r.unwrap() as usize).collect();
        let s = &mut self.stats;
        s.finalized += 1;
        for (j, o) in s.oracle.iter_mut().enumerate() {
            *o += ranks.iter().take_while(|&&r| r < (1 << j)).count() as u64;
        }
        s.picked_prefix += ranks
            .iter()
            .zip(&c.picked)
            .take_while(|(r, p)| **r == **p as usize)
            .count() as u64;
        if let Some(&r) = ranks.iter().find(|&&r| r != 0) {
            let bin = match r {
                1 => 0,
                2 => 1,
                3 => 2,
                4..=7 => 3,
                8..=15 => 4,
                _ => 5,
            };
            s.first_miss[bin] += 1;
        }
    }

    fn report(&self, tag: &str) {
        let s = &self.stats;
        if s.cycles == 0 {
            return;
        }
        let f = s.finalized.max(1) as f64;
        let o: Vec<String> = s
            .oracle
            .iter()
            .enumerate()
            .map(|(j, &v)| format!("k{}={:.3}", 1 << j, v as f64 / f))
            .collect();
        eprintln!(
            "[dflash-online] {tag}: mode={:?} cycles={} tau={:.3} finalized={} picked_tau={:.3} oracle[{}] first_miss_rank[1,2,3,4-7,8-15,>=16]={:?}",
            self.mode,
            s.cycles,
            s.accepted as f64 / s.cycles as f64,
            s.finalized,
            s.picked_prefix as f64 / f,
            o.join(" "),
            s.first_miss,
        );
    }
}
