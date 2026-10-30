// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Node bench + accuracy for the gfx12 W8A8 Q8_0 GEMM (`gemm_q8_0_a8_batched`)
//! against the F16-activation incumbent (`gemm_q8_0_batched_chunked`).
//!
//! Per `(M, K)` shape and row count `N`:
//!   * time: median over `ITERS` single calls (GPU events around the whole
//!     launcher call, so the incumbent includes its F32 -> F16 convert and the
//!     candidate its activation quantizer), plus the GEMM-only time of every
//!     tile variant and the quantizer alone;
//!   * accuracy on BF16-valued activations with realistic outliers (persistent
//!     massive channels and a few hot rows): new vs incumbent over the whole
//!     output (max / mean abs, relative = sum|d| / sum|ref|), and both vs an F64
//!     CPU reference on a random subset of outputs.
//!
//! Usage: bench_q8_0_a8_gfx12 [--shapes all|big] [--rows 512,4096] [--iters 21] [--out file.md]
//! Needs a gfx12 GPU (run through fabric-run.sh).

use hip_bridge::Event;
use rdna_compute::{DType, Gpu, Q8A8Tile};
use std::fmt::Write as _;

/// Raon-1-mini (M, K, label) projection sites.
const SHAPES: &[(usize, usize, &str)] = &[
    (5120, 2560, "RDA q/k/v/g, MLA g"),
    (2560, 5120, "o"),
    (576, 2560, "MLA kv_a"),
    (7680, 576, "MLA"),
    (10240, 512, "MLA q_b"),
];

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    /// Approximately N(0, 1) (sum of 4 uniforms, variance 1/3 * 4 / 4 -> scaled).
    fn gauss(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.unit()).sum();
        (s - 2.0) * 1.732_050_8
    }
}

fn bf16_round(x: f32) -> f32 {
    let u = x.to_bits();
    let r = u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(r)
}

/// Q8_0 weights `[m, k]`: bytes (34 B blocks) and the dequantized F64 matrix rows
/// needed by the reference (full matrix; m*k <= 5.3M doubles).
fn make_weights(m: usize, k: usize, rng: &mut Rng) -> (Vec<u8>, Vec<f64>) {
    let nb = k / 32;
    let mut bytes = Vec::with_capacity(m * nb * 34);
    let mut deq = Vec::with_capacity(m * k);
    for _ in 0..m * nb {
        let sc = half::f16::from_f32(0.004 * (1.0 + 0.5 * rng.gauss().abs()));
        bytes.extend_from_slice(&sc.to_bits().to_le_bytes());
        let scf = sc.to_f32() as f64;
        for _ in 0..32 {
            let q = (rng.gauss() * 38.0).round().clamp(-127.0, 127.0) as i8;
            bytes.push(q as u8);
            deq.push(scf * q as f64);
        }
    }
    (bytes, deq)
}

/// BF16-valued activations: unit Gaussian bulk, ~0.8% persistent massive
/// channels (x25..x60) and ~1.5% hot rows (x8).
fn make_acts(n: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    let hot_ch: Vec<f32> = (0..k)
        .map(|_| {
            if rng.unit() < 0.008 {
                25.0 + 35.0 * rng.unit()
            } else {
                1.0
            }
        })
        .collect();
    let mut x = Vec::with_capacity(n * k);
    for _ in 0..n {
        let row = if rng.unit() < 0.015 { 8.0 } else { 1.0 };
        for c in 0..k {
            x.push(bf16_round(rng.gauss() * row * hot_ch[c]));
        }
    }
    x
}

struct Timer {
    e0: Event,
    e1: Event,
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_ms(gpu: &mut Gpu, t: &Timer, iters: usize, mut f: impl FnMut(&mut Gpu)) -> f32 {
    let mut samples = Vec::with_capacity(iters);
    for i in 0..iters + 3 {
        gpu.hip.event_record(&t.e0, None).unwrap();
        f(gpu);
        gpu.hip.event_record(&t.e1, None).unwrap();
        gpu.hip.event_synchronize(&t.e1).unwrap();
        let ms = gpu.hip.event_elapsed_ms(&t.e0, &t.e1).unwrap();
        if i >= 3 {
            samples.push(ms);
        }
    }
    median(samples)
}

struct ErrStat {
    max_abs: f64,
    mean_abs: f64,
    rel: f64,
}

fn diff(a: &[f32], b: &[f32]) -> ErrStat {
    let (mut mx, mut s, mut sr) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let d = (x as f64 - y as f64).abs();
        mx = mx.max(d);
        s += d;
        sr += (y as f64).abs();
    }
    ErrStat {
        max_abs: mx,
        mean_abs: s / a.len() as f64,
        rel: s / sr.max(1e-30),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let iters: usize = arg("--iters").map_or(21, |s| s.parse().unwrap());
    let rows: Vec<usize> = arg("--rows").map_or(vec![512, 4096], |s| {
        s.split(',').map(|v| v.parse().unwrap()).collect()
    });
    let big_only = arg("--shapes").as_deref() == Some("big");
    let mut gpu = Gpu::init().expect("gpu init");
    println!("arch {}", gpu.arch);
    assert!(gpu.q8_a8_supported(32), "needs gfx12");
    let timer = Timer {
        e0: gpu.hip.event_create().unwrap(),
        e1: gpu.hip.event_create().unwrap(),
    };
    let mut md = String::new();
    let _ = writeln!(
        md,
        "| M | K | N | incumbent F16 ms (TFLOP/s) | W8A8 ms (TFLOP/s) | speedup | quant ms | gemm 128x128 | gemm 128x64 | gemm 64x64 |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    );
    let mut acc_md = String::new();
    let _ = writeln!(
        acc_md,
        "| M | K | N | new-vs-inc max abs | mean abs | rel | new-vs-F64 max abs | mean abs | rel | inc-vs-F64 max abs | mean abs | rel |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    );
    let mut clean_md = String::new();
    let _ = writeln!(
        clean_md,
        "| M | K | N | new-vs-inc max abs | mean abs | rel | new-vs-F64 max abs | mean abs | rel | inc-vs-F64 max abs | mean abs | rel |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    );
    let mut rng = Rng(0x5eed_0a8a_0001);
    for &(m, k, label) in SHAPES {
        if big_only && m * k < 5120 * 2560 {
            continue;
        }
        let (wbytes, wdeq) = make_weights(m, k, &mut rng);
        let w = gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w");
        for &n in &rows {
            println!("shape M={m} K={k} N={n} ({label})");
            let x_host = make_acts(n, k, &mut rng);
            let x = gpu.upload_f32(&x_host, &[n * k]).expect("x");
            let y_inc = gpu.alloc_tensor(&[n * m], DType::F32).expect("y");
            let y_new = gpu.alloc_tensor(&[n * m], DType::F32).expect("y2");

            gpu.gemm_q8_0_batched_chunked(&w, &x, &y_inc, m, k, n)
                .unwrap();
            gpu.gemm_q8_0_a8_batched(&w, &x, &y_new, m, k, n).unwrap();
            gpu.hip.device_synchronize().unwrap();

            let t_inc = time_ms(&mut gpu, &timer, iters, |g| {
                g.gemm_q8_0_batched_chunked(&w, &x, &y_inc, m, k, n)
                    .unwrap()
            });
            let t_new = time_ms(&mut gpu, &timer, iters, |g| {
                g.gemm_q8_0_a8_batched(&w, &x, &y_new, m, k, n).unwrap()
            });
            let acts = gpu.q8_a8_quantize_x(&x, n, k).unwrap();
            let t_q = time_ms(&mut gpu, &timer, iters, |g| {
                g.q8_a8_quantize_x(&x, n, k).unwrap();
            });
            let mut t_tiles = Vec::new();
            for tile in Q8A8Tile::ALL {
                let t = time_ms(&mut gpu, &timer, iters, |g| {
                    g.gemm_q8_0_a8_gemm_tile(&w, &acts, &y_new, m, tile)
                        .unwrap()
                });
                t_tiles.push(t);
            }
            // The tile loop left y_new from the last tile; recompute with the
            // production pick for the accuracy comparison.
            gpu.gemm_q8_0_a8_batched(&w, &x, &y_new, m, k, n).unwrap();
            gpu.hip.device_synchronize().unwrap();

            let tf = 2.0 * m as f64 * k as f64 * n as f64 / 1e12;
            let _ = writeln!(
                md,
                "| {m} | {k} | {n} | {t_inc:.4} ({:.1}) | {t_new:.4} ({:.1}) | {:.2}x | {t_q:.4} | {:.4} ({:.2}x) | {:.4} ({:.2}x) | {:.4} ({:.2}x) |",
                tf / (t_inc as f64 / 1e3),
                tf / (t_new as f64 / 1e3),
                t_inc / t_new,
                t_tiles[0],
                t_inc / t_tiles[0],
                t_tiles[1],
                t_inc / t_tiles[1],
                t_tiles[2],
                t_inc / t_tiles[2],
            );

            let inc = gpu.download_f32(&y_inc).unwrap();
            let new = gpu.download_f32(&y_new).unwrap();
            let d_ni = diff(&new, &inc);
            // F64 reference on a random subset of (token, row) outputs.
            let mut r2 = Rng(0xf64 + (m * 31 + n) as u64);
            let (mut ref_v, mut new_v, mut inc_v) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..384 {
                let tok = r2.next_u32() as usize % n;
                let row = r2.next_u32() as usize % m;
                let mut s = 0f64;
                for c in 0..k {
                    s += wdeq[row * k + c] * x_host[tok * k + c] as f64;
                }
                ref_v.push(s as f32);
                new_v.push(new[tok * m + row]);
                inc_v.push(inc[tok * m + row]);
            }
            // diff() takes (a, b) with b the reference; refs here are F32-rounded
            // F64 sums (|ref| >> f32 eps relative to the errors measured).
            let d_nr = diff(&new_v, &ref_v);
            let d_ir = diff(&inc_v, &ref_v);
            let _ = writeln!(
                acc_md,
                "| {m} | {k} | {n} | {:.4e} | {:.4e} | {:.3e} | {:.4e} | {:.4e} | {:.3e} | {:.4e} | {:.4e} | {:.3e} |",
                d_ni.max_abs, d_ni.mean_abs, d_ni.rel,
                d_nr.max_abs, d_nr.mean_abs, d_nr.rel,
                d_ir.max_abs, d_ir.mean_abs, d_ir.rel,
            );

            // Same comparison on plain BF16-rounded Gaussian activations (no outlier
            // channels / hot rows) for the smallest row count.
            if n == rows[0] {
                let xc: Vec<f32> = (0..n * k).map(|_| bf16_round(rng.gauss())).collect();
                let xct = gpu.upload_f32(&xc, &[n * k]).expect("xc");
                gpu.gemm_q8_0_batched_chunked(&w, &xct, &y_inc, m, k, n)
                    .unwrap();
                gpu.gemm_q8_0_a8_batched(&w, &xct, &y_new, m, k, n).unwrap();
                gpu.hip.device_synchronize().unwrap();
                let inc = gpu.download_f32(&y_inc).unwrap();
                let new = gpu.download_f32(&y_new).unwrap();
                let d_ni = diff(&new, &inc);
                let mut r3 = Rng(0xc1ea + (m * 31 + n) as u64);
                let (mut ref_v, mut new_v, mut inc_v) = (Vec::new(), Vec::new(), Vec::new());
                for _ in 0..384 {
                    let tok = r3.next_u32() as usize % n;
                    let row = r3.next_u32() as usize % m;
                    let mut s = 0f64;
                    for c in 0..k {
                        s += wdeq[row * k + c] * xc[tok * k + c] as f64;
                    }
                    ref_v.push(s as f32);
                    new_v.push(new[tok * m + row]);
                    inc_v.push(inc[tok * m + row]);
                }
                let d_nr = diff(&new_v, &ref_v);
                let d_ir = diff(&inc_v, &ref_v);
                let _ = writeln!(
                    clean_md,
                    "| {m} | {k} | {n} | {:.4e} | {:.4e} | {:.3e} | {:.4e} | {:.4e} | {:.3e} | {:.4e} | {:.4e} | {:.3e} |",
                    d_ni.max_abs, d_ni.mean_abs, d_ni.rel,
                    d_nr.max_abs, d_nr.mean_abs, d_nr.rel,
                    d_ir.max_abs, d_ir.mean_abs, d_ir.rel,
                );
            }
        }
    }
    let out = format!(
        "## Node bench (median of {iters} calls, GPU events)\n\n{md}\n## Accuracy (BF16-valued X with massive channels + hot rows)\n\n{acc_md}\n## Accuracy (plain BF16-rounded Gaussian X)\n\n{clean_md}"
    );
    println!("\n{out}");
    if let Some(path) = arg("--out") {
        std::fs::write(&path, &out).expect("write --out");
    }
}
