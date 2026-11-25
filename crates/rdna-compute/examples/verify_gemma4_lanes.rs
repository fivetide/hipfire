// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.
//
// Gate for the lane kernels of `KvRoute::Lanes` at the Gemma 4 attention
// shapes (HIPF-545): sliding (head_dim 256, window 1024) and full (head_dim
// 512, K=V), 4 lanes of 8192 rows in one lane-major arena, at positions
// 0, 1023, 1500 and 8191 (window edge and cache end).
//   - The masked lane K/V write is byte-equal to the one-row decode write
//     (`kv_cache_write_q8_0`) into a separate per-request cache.
//   - The lane attention matches a CPU reference over the dequantized cache.
//   - An inactive lane neither writes its region nor its output row.
//
// Exit 0 = all pass, 1 = mismatch. Skips without a GPU.

use rdna_compute::{Gpu, GpuTensor};

const NH: usize = 16;
const CAP: usize = 8192;
const POS: [usize; 4] = [0, 1023, 1500, CAP - 1];

struct Shape {
    name: &'static str,
    n_kv: usize,
    hd: usize,
    window: usize,
}

const SHAPES: [Shape; 2] = [
    Shape {
        name: "sliding hd256 w1024",
        n_kv: 8,
        hd: 256,
        window: 1024,
    },
    Shape {
        name: "full hd512 K=V",
        n_kv: 2,
        hd: 512,
        window: 0,
    },
];

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        self.0
    }
    fn f(&mut self) -> f32 {
        ((self.next() >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

fn f16_val(lo: u8, hi: u8) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([lo, hi])).to_f32()
}

/// `rows` random Q8_0 cache rows of `row_bytes`.
fn q8_rows(rng: &mut Rng, rows: usize, row_bytes: usize) -> Vec<u8> {
    let mut kv = vec![0u8; rows * row_bytes];
    for blk in kv.chunks_exact_mut(34) {
        let [lo, hi] = f16_bits(0.005 + 0.02 * rng.f().abs()).to_le_bytes();
        blk[0] = lo;
        blk[1] = hi;
        for b in &mut blk[2..] {
            *b = ((rng.next() % 255) as i32 - 127) as i8 as u8;
        }
    }
    kv
}

fn dequant(row: &[u8], kv_h: usize, hd: usize) -> Vec<f32> {
    (0..hd)
        .map(|d| {
            let blk = &row[(kv_h * hd / 32 + d / 32) * 34..];
            f16_val(blk[0], blk[1]) * blk[2 + d % 32] as i8 as f32
        })
        .collect()
}

/// CPU attention of one lane row over its `[lo, pos]` cache rows.
fn reference(s: &Shape, q: &[f32], k: &[u8], v: &[u8], pos: usize) -> Vec<f32> {
    let row_bytes = s.n_kv * s.hd / 32 * 34;
    let lo = if s.window > 0 && pos + 1 > s.window {
        pos + 1 - s.window
    } else {
        0
    };
    let scale = 1.0 / (s.hd as f32).sqrt();
    let mut out = vec![0f32; NH * s.hd];
    for h in 0..NH {
        let kv_h = h / (NH / s.n_kv);
        let qh = &q[h * s.hd..(h + 1) * s.hd];
        let scores: Vec<f64> = (lo..=pos)
            .map(|t| {
                let kt = dequant(&k[t * row_bytes..], kv_h, s.hd);
                qh.iter().zip(&kt).map(|(a, b)| (a * b) as f64).sum::<f64>() * scale as f64
            })
            .collect();
        let max = scores.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = scores.iter().map(|x| (x - max).exp()).collect();
        let sum: f64 = e.iter().sum();
        for (i, t) in (lo..=pos).enumerate() {
            let vt = dequant(&v[t * row_bytes..], kv_h, s.hd);
            for d in 0..s.hd {
                out[h * s.hd + d] += (e[i] / sum * vt[d] as f64) as f32;
            }
        }
    }
    out
}

fn upload_i32(gpu: &mut Gpu, data: &[i32]) -> GpuTensor {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.upload_raw(&bytes, &[data.len()]).expect("upload i32")
}

fn check(gpu: &mut Gpu, s: &Shape, mask: u64) -> bool {
    let lanes = POS.len();
    let row_bytes = s.n_kv * s.hd / 32 * 34;
    let lane_bytes = CAP * row_bytes;
    let kv_dim = s.n_kv * s.hd;
    let mut rng = Rng(0x5eed ^ s.hd as u32 ^ mask as u32);
    let mut ok = true;

    // History rows (as a park copies them) and this step's F32 K/V rows.
    let k_hist = q8_rows(&mut rng, lanes * CAP, row_bytes);
    let v_hist = if s.window == 0 {
        k_hist.clone()
    } else {
        q8_rows(&mut rng, lanes * CAP, row_bytes)
    };
    let k_new: Vec<f32> = (0..lanes * kv_dim).map(|_| rng.f()).collect();
    let v_new: Vec<f32> = if s.window == 0 {
        k_new.clone()
    } else {
        (0..lanes * kv_dim).map(|_| rng.f()).collect()
    };
    let q: Vec<f32> = (0..lanes * NH * s.hd).map(|_| rng.f() * 4.0).collect();
    let canary = -7.25f32;

    // Lane path: one arena, one masked write per K/V, one attention launch.
    let k_arena = gpu.upload_raw(&k_hist, &[k_hist.len()]).expect("k arena");
    let v_arena = gpu.upload_raw(&v_hist, &[v_hist.len()]).expect("v arena");
    let k_src = gpu.upload_f32(&k_new, &[k_new.len()]).expect("k src");
    let v_src = gpu.upload_f32(&v_new, &[v_new.len()]).expect("v src");
    let q_t = gpu.upload_f32(&q, &[q.len()]).expect("q");
    let out_t = gpu
        .upload_f32(&vec![canary; q.len()], &[q.len()])
        .expect("out");
    let pos: Vec<i32> = POS.iter().map(|&p| p as i32).collect();
    let pos_t = upload_i32(gpu, &pos);
    for (cache, src) in [(&k_arena, &k_src), (&v_arena, &v_src)] {
        gpu.kv_cache_write_q8_0_independent_masked(
            cache, src, &pos_t, s.n_kv, s.hd, lanes, CAP, mask,
        )
        .expect("lane write");
    }
    gpu.attention_q8_0_kv_independent_masked_windowed(
        &q_t, &k_arena, &v_arena, &out_t, &pos_t, NH, s.n_kv, s.hd, CAP, CAP, lanes, mask, s.window,
    )
    .expect("lane attention");
    let k_lanes = gpu.download_raw_bytes(&k_arena).expect("dl k");
    let v_lanes = gpu.download_raw_bytes(&v_arena).expect("dl v");
    let out = gpu.download_f32(&out_t).expect("dl out");

    for (lane, &p) in POS.iter().enumerate() {
        let active = mask >> lane & 1 == 1;
        let region = lane * lane_bytes..(lane + 1) * lane_bytes;
        let row = lane * NH * s.hd..(lane + 1) * NH * s.hd;
        if !active {
            let same = k_lanes[region.clone()] == k_hist[region.clone()]
                && v_lanes[region.clone()] == v_hist[region.clone()]
                && out[row].iter().all(|&x| x == canary);
            println!(
                "  {} {} lane {lane} inactive: untouched",
                if same { "PASS" } else { "FAIL" },
                s.name
            );
            ok &= same;
            continue;
        }
        // One-row decode write into this request's own cache.
        let mut write_one = |hist: &[u8], new: &[f32]| {
            let cache = gpu
                .upload_raw(&hist[region.clone()], &[lane_bytes])
                .unwrap();
            let src = gpu
                .upload_f32(&new[lane * kv_dim..(lane + 1) * kv_dim], &[kv_dim])
                .unwrap();
            let p_t = upload_i32(gpu, &[p as i32]);
            gpu.kv_cache_write_q8_0(&cache, &src, &p_t.buf, s.n_kv, s.hd)
                .expect("one-row write");
            let bytes = gpu.download_raw_bytes(&cache).unwrap();
            for t in [cache, src, p_t] {
                let _ = gpu.free_tensor(t);
            }
            bytes
        };
        let k_one = write_one(&k_hist, &k_new);
        let v_one = write_one(&v_hist, &v_new);
        let write_eq = k_lanes[region.clone()] == k_one[..] && v_lanes[region.clone()] == v_one[..];
        let want = reference(s, &q[row.clone()], &k_one, &v_one, p);
        let got = &out[row];
        let mad = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let attn_ok = mad < 1e-4;
        println!(
            "  {} {} lane {lane} pos {p}: write byte-equal={write_eq} attention max_abs={mad:.3e}",
            if write_eq && attn_ok { "PASS" } else { "FAIL" },
            s.name
        );
        ok &= write_eq && attn_ok;
    }
    for t in [k_arena, v_arena, k_src, v_src, q_t, out_t, pos_t] {
        let _ = gpu.free_tensor(t);
    }
    ok
}

fn main() {
    let mut gpu = match Gpu::init() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP: no GPU ({e})");
            return;
        }
    };
    let mut ok = true;
    for s in &SHAPES {
        let max_cap = gpu.attention_q8_0_kv_independent_max_lane_capacity(s.hd);
        println!("=== {} (max lane capacity {max_cap}) ===", s.name);
        if max_cap < CAP {
            eprintln!("  FAIL: lane capacity {CAP} > device limit {max_cap}");
            ok = false;
            continue;
        }
        ok &= check(&mut gpu, s, 0b1111);
        ok &= check(&mut gpu, s, 0b1011);
    }
    if !ok {
        eprintln!("FAIL");
        std::process::exit(1);
    }
    println!("PASS: Gemma 4 lane write + attention");
}
