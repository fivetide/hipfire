// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx12 W8A8 Q8_0 GEMM (`gemm_q8_0_a8_batched`, `kernels/src/gemm_q8_0_mmq_i8.gfx12.hip`)
//! against a CPU emulation of its exact contract:
//!
//!   amax = max |x| over a 32-K block; d = RN(amax / 127); q = clamp(rint(x / d), +-127)
//!   (zero block: d = 0, q = 0); per (row, token) acc = 0; for blocks b ascending:
//!   c = sum_i W8 * X8 (exact int); acc = fma(RN(ws * xs), float(c), acc).
//!
//! The integer dot is exact, so GPU and emulation must agree bit for bit for every
//! tile variant, both weight read paths (4-byte aligned K % 64 == 0 `FAST`, and the
//! 2-byte aligned `_slow` path used for K % 64 == 32 or a misaligned weight base)
//! and all M / N tails. An F64 reference bounds the quantization error.
//!
//! `#[ignore]`d: needs a gfx12 GPU. Run through fabric-run.sh:
//!
//!   cargo test -p rdna-compute --release --test gemm_q8_0_a8_gfx12 -- --ignored

use rdna_compute::{DType, Gpu, GpuTensor, Q8A8Tile};

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
    fn gauss(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.unit()).sum();
        (s - 2.0) * 1.732_050_8
    }
}

fn bf16_round(x: f32) -> f32 {
    let u = x.to_bits();
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}

/// Returns (Q8_0 bytes, int8 quants [m*k], f32 scales [m * k/32]).
fn make_weights(m: usize, k: usize, rng: &mut Rng) -> (Vec<u8>, Vec<i8>, Vec<f32>) {
    let nb = k / 32;
    let mut bytes = Vec::new();
    let (mut qs, mut ss) = (Vec::new(), Vec::new());
    for _ in 0..m * nb {
        let sc = half::f16::from_f32(0.004 * (1.0 + 0.5 * rng.gauss().abs()));
        bytes.extend_from_slice(&sc.to_bits().to_le_bytes());
        ss.push(sc.to_f32());
        for _ in 0..32 {
            let q = (rng.gauss() * 40.0).round().clamp(-128.0, 127.0) as i8;
            bytes.push(q as u8);
            qs.push(q);
        }
    }
    (bytes, qs, ss)
}

fn emulate(wq: &[i8], ws: &[f32], x: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let nb = k / 32;
    let mut xq = vec![0i8; n * k];
    let mut xs = vec![0f32; n * nb];
    for t in 0..n {
        for b in 0..nb {
            let blk = &x[t * k + b * 32..t * k + b * 32 + 32];
            let amax = blk.iter().fold(0f32, |a, v| a.max(v.abs()));
            if amax == 0.0 {
                continue;
            }
            let d = amax / 127.0;
            xs[t * nb + b] = d;
            for (i, &v) in blk.iter().enumerate() {
                xq[t * k + b * 32 + i] = (v / d).round_ties_even().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    let mut y = vec![0f32; n * m];
    for t in 0..n {
        for r in 0..m {
            let mut acc = 0f32;
            for b in 0..nb {
                let c: i32 = (0..32)
                    .map(|i| wq[r * k + b * 32 + i] as i32 * xq[t * k + b * 32 + i] as i32)
                    .sum();
                let scale = ws[r * nb + b] * xs[t * nb + b];
                acc = scale.mul_add(c as f32, acc);
            }
            y[t * m + r] = acc;
        }
    }
    y
}

fn run_case(gpu: &mut Gpu, m: usize, k: usize, n: usize, misalign_base: bool, seed: u64) {
    let mut rng = Rng(seed);
    let (wbytes, wq, ws) = make_weights(m, k, &mut rng);
    // Massive channels + hot rows exercise the per-block scale range.
    let hot: Vec<f32> = (0..k)
        .map(|_| if rng.unit() < 0.02 { 30.0 } else { 1.0 })
        .collect();
    let x: Vec<f32> = (0..n)
        .flat_map(|t| {
            let row = if t % 7 == 3 { 9.0 } else { 1.0 };
            let g: Vec<f32> = (0..k).map(|_| rng.gauss()).collect();
            g.into_iter()
                .zip(hot.iter())
                .map(move |(v, h)| bf16_round(v * h * row))
                .collect::<Vec<_>>()
        })
        .collect();
    // An all-zero token block exercises the d = 0 path.
    let mut x = x;
    for v in &mut x[..32.min(k)] {
        *v = 0.0;
    }

    // Weight upload: optionally shifted by 2 bytes so the base is only 2-aligned.
    let (w, _owner): (GpuTensor, Option<GpuTensor>) = if misalign_base {
        let mut padded = vec![0u8; 2];
        padded.extend_from_slice(&wbytes);
        let owner = gpu.upload_raw(&padded, &[padded.len()]).expect("w");
        let view = owner.sub_offset(2, wbytes.len());
        (view, Some(owner))
    } else {
        (gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w"), None)
    };
    let xt = gpu.upload_f32(&x, &[n * k]).expect("x");
    let want = emulate(&wq, &ws, &x, m, k, n);

    let check = |gpu: &Gpu, y: &GpuTensor, what: &str| {
        let got = gpu.download_f32(y).expect("download");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what} m={m} k={k} n={n} misalign={misalign_base}: Y[{}][{}] gpu {g} emu {w}",
                i / m,
                i % m
            );
        }
    };

    let acts = gpu.q8_a8_quantize_x(&xt, n, k).expect("quantize");
    for tile in Q8A8Tile::ALL {
        // NaN-poisoned output so a missed cell fails the bit compare.
        let y = gpu.upload_f32(&vec![f32::NAN; n * m], &[n * m]).expect("y");
        gpu.gemm_q8_0_a8_gemm_tile(&w, &acts, &y, m, tile)
            .unwrap_or_else(|e| panic!("{}: {e:?}", tile.label()));
        check(gpu, &y, tile.label());
    }
    let y = gpu.upload_f32(&vec![f32::NAN; n * m], &[n * m]).expect("y");
    gpu.gemm_q8_0_a8_batched(&w, &xt, &y, m, k, n)
        .expect("batched");
    check(gpu, &y, "batched");
}

#[test]
#[ignore = "needs a gfx12 GPU"]
fn w8a8_q8_0_matches_cpu_emulation_bitwise() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(gpu.q8_a8_supported(32), "needs gfx12, got {}", gpu.arch);
    // (m, k, n): tails in M and N; K = 576 / 512 (Raon MLA), K % 64 == 32 (slow
    // path), single-slab K, a wider multi-tile case.
    for (i, &(m, k, n)) in [
        (130usize, 640usize, 70usize),
        (97, 96, 33),
        (200, 576, 130),
        (64, 512, 64),
        (129, 32, 5),
        (260, 1280, 257),
    ]
    .iter()
    .enumerate()
    {
        run_case(&mut gpu, m, k, n, false, 0xa8a8_0000 + i as u64);
    }
    // Misaligned weight base takes the 2-byte `_slow` kernels even at K % 64 == 0.
    run_case(&mut gpu, 130, 640, 70, true, 0xa8a8_1000);
}

#[test]
#[ignore = "needs a gfx12 GPU"]
fn w8a8_q8_0_error_vs_f64_is_activation_quantization_sized() {
    let mut gpu = Gpu::init().expect("gpu init");
    assert!(gpu.q8_a8_supported(32), "needs gfx12, got {}", gpu.arch);
    let (m, k, n) = (192usize, 2560usize, 96usize);
    let mut rng = Rng(0xf64f_a8a8);
    let (wbytes, wq, ws) = make_weights(m, k, &mut rng);
    let x: Vec<f32> = (0..n * k).map(|_| bf16_round(rng.gauss())).collect();
    let w = gpu.upload_raw(&wbytes, &[wbytes.len()]).expect("w");
    let xt = gpu.upload_f32(&x, &[n * k]).expect("x");
    let y = gpu.alloc_tensor(&[n * m], DType::F32).expect("y");
    gpu.gemm_q8_0_a8_batched(&w, &xt, &y, m, k, n).expect("a8");
    let got = gpu.download_f32(&y).expect("download");
    let nb = k / 32;
    let (mut se, mut sr) = (0f64, 0f64);
    for t in 0..n {
        for r in 0..m {
            let mut s = 0f64;
            for c in 0..k {
                s += ws[r * nb + c / 32] as f64 * wq[r * k + c] as f64 * x[t * k + c] as f64;
            }
            se += (got[t * m + r] as f64 - s).abs();
            sr += s.abs();
        }
    }
    let rel = se / sr;
    // int8 activations at amax/127 on Gaussian blocks: ~0.5% mean relative error
    // (measured 5.1e-3). A structural error shows up as >1e-2.
    assert!(rel < 1.5e-2, "mean relative error vs F64 {rel:.3e}");
}
