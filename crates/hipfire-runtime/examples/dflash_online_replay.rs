//! Offline replay of `HIPFIRE_DFLASH_ONLINE_DUMP` records through the online
//! draft tuner (developer-only). Block starts and emitted text are the
//! recorded ones, so the reported `picked τ` is the tuner's acceptance at those
//! starts on that text — a deterministic, alignment-free estimate of the
//! online τ (online, a longer accept moves the next block start).
//! Each dump is one session; the summary covers its last request.
//!
//! `--first N`: score only the first N verify cycles of each session (short
//! requests). `--carry-loo`: each session runs after all the others in one
//! tuner with `carry` on (weights kept across requests, session statistics
//! reset) — what a long-running daemon sees. `--no-prompt`: ignore the
//! recorded prompt (session history starts at the first generated token).
//!
//! Usage: dflash_online_replay [--hp key=val,...] [--first N] [--carry-loo] [--no-prompt] DUMP...

use hipfire_runtime::dflash_online::{Hyper, Mode, OnlineDraftTuner, K};

fn u64_at(b: &[u8], o: &mut usize) -> u64 {
    let v = u64::from_le_bytes(b[*o..*o + 8].try_into().unwrap());
    *o += 8;
    v
}

fn u32s(b: &[u8], o: &mut usize, n: usize) -> Vec<u32> {
    let v = b[*o..*o + 4 * n]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    *o += 4 * n;
    v
}

/// Feed the records of one dump into `t` (a request boundary calls
/// `t.reset()`), stopping after `max_obs` verify cycles of a request.
fn feed(path: &str, b: &[u8], t: &mut OnlineDraftTuner, max_obs: usize, prompt: bool) {
    let (mut o, mut obs) = (0usize, 0usize);
    let mut prop: Option<(usize, u32, usize, Vec<u32>)> = None;
    while o < b.len() {
        let tag = b[o];
        o += 1;
        match tag {
            b'P' => {
                let start = u64_at(b, &mut o) as usize;
                let seed = u64_at(b, &mut o) as u32;
                let rows = u64_at(b, &mut o) as usize;
                prop = Some((start, seed, rows, u32s(b, &mut o, rows * K)));
            }
            b'L' => {
                let (start, seed, rows, ids) = prop.take().expect("L record without P");
                let vals: Vec<f32> = u32s(b, &mut o, ids.len())
                    .into_iter()
                    .map(f32::from_bits)
                    .collect();
                t.propose(start, seed, &ids, &vals, rows);
            }
            b'O' => {
                let start = u64_at(b, &mut o) as usize;
                let accepted = u64_at(b, &mut o) as usize;
                let n = u64_at(b, &mut o) as usize;
                t.observe(start, &u32s(b, &mut o, n), accepted);
                obs += 1;
                if obs >= max_obs {
                    return;
                }
            }
            b'R' => {
                t.reset();
                obs = 0;
            }
            b'S' => {
                let n = u64_at(b, &mut o) as usize;
                let toks = u32s(b, &mut o, n);
                if prompt {
                    t.seed_prompt(&toks);
                }
            }
            _ => panic!("{path}: bad record tag {tag} at {}", o - 1),
        }
    }
}

fn arg_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let i = args.iter().position(|a| a == flag)?;
    let v = args.remove(i + 1);
    args.remove(i);
    Some(v)
}

fn take_flag(args: &mut Vec<String>, flag: &str) -> bool {
    args.iter()
        .position(|a| a == flag)
        .map(|i| args.remove(i))
        .is_some()
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let hyper = Hyper::parse(&arg_value(&mut args, "--hp").unwrap_or_default())
        .unwrap_or_else(|e| panic!("{e}"));
    let first = arg_value(&mut args, "--first").map_or(usize::MAX, |v| v.parse().unwrap());
    let carry_loo = take_flag(&mut args, "--carry-loo");
    let prompt = !take_flag(&mut args, "--no-prompt");
    let data: Vec<Vec<u8>> = args
        .iter()
        .map(|p| std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}")))
        .collect();
    let mut ratios = Vec::new();
    for (i, path) in args.iter().enumerate() {
        let mut t = if carry_loo {
            let mut t = OnlineDraftTuner::new(
                Mode::On,
                Hyper {
                    carry: true,
                    ..hyper.clone()
                },
            );
            for (j, p) in args.iter().enumerate().filter(|(j, _)| *j != i) {
                feed(p, &data[j], &mut t, usize::MAX, prompt);
            }
            t.reset();
            t
        } else {
            OnlineDraftTuner::new(Mode::On, hyper.clone())
        };
        feed(path, &data[i], &mut t, first, prompt);
        let (fin, base, pick) = t.summary();
        println!("{path}: cycles={fin} argmax_tau={base:.3} picked_tau={pick:.3}");
        ratios.push(pick / base);
    }
    let geo = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len().max(1) as f64).exp();
    println!(
        "GEOMEAN picked/argmax tau over {} sessions: {geo:.4} ({}hp: {hyper:?})",
        ratios.len(),
        if carry_loo { "carry-loo, " } else { "" }
    );
}
