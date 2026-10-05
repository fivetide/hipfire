//! Offline replay of `HIPFIRE_DFLASH_ONLINE_DUMP` records through the online
//! draft tuner (developer-only). Block starts and emitted text are the
//! recorded ones, so the reported `picked τ` is the tuner's acceptance at those
//! starts on that text — a deterministic, alignment-free estimate of the
//! online τ (online, a longer accept moves the next block start).
//! Each dump is one session (a request boundary resets the tuner); the
//! summary covers the last request.
//!
//! Usage: dflash_online_replay [--hp key=val,...] DUMP...

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

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut hp = String::new();
    if let Some(i) = args.iter().position(|a| a == "--hp") {
        hp = args.remove(i + 1);
        args.remove(i);
    }
    let hyper = Hyper::parse(&hp).unwrap_or_else(|e| panic!("{e}"));
    let mut ratios = Vec::new();
    for path in &args {
        let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut t = OnlineDraftTuner::new(Mode::On, hyper.clone());
        let mut o = 0usize;
        let mut prop: Option<(usize, u32, usize, Vec<u32>)> = None;
        while o < b.len() {
            let tag = b[o];
            o += 1;
            match tag {
                b'P' => {
                    let start = u64_at(&b, &mut o) as usize;
                    let seed = u64_at(&b, &mut o) as u32;
                    let rows = u64_at(&b, &mut o) as usize;
                    prop = Some((start, seed, rows, u32s(&b, &mut o, rows * K)));
                }
                b'L' => {
                    let (start, seed, rows, ids) = prop.take().expect("L record without P");
                    let vals: Vec<f32> = u32s(&b, &mut o, ids.len())
                        .into_iter()
                        .map(f32::from_bits)
                        .collect();
                    t.propose(start, seed, &ids, &vals, rows);
                }
                b'O' => {
                    let start = u64_at(&b, &mut o) as usize;
                    let accepted = u64_at(&b, &mut o) as usize;
                    let n = u64_at(&b, &mut o) as usize;
                    t.observe(start, &u32s(&b, &mut o, n), accepted);
                }
                b'R' => t = OnlineDraftTuner::new(Mode::On, hyper.clone()),
                _ => panic!("{path}: bad record tag {tag} at {}", o - 1),
            }
        }
        let (fin, base, pick) = t.summary();
        println!("{path}: cycles={fin} argmax_tau={base:.3} picked_tau={pick:.3}");
        ratios.push(pick / base);
    }
    let geo = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len().max(1) as f64).exp();
    println!(
        "GEOMEAN picked/argmax tau over {} sessions: {geo:.4} (hp: {hyper:?})",
        ratios.len()
    );
}
