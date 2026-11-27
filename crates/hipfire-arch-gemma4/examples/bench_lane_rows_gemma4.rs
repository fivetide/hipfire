// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Björn Bösel
// hipfire — see LICENSE and NOTICE in the project root.

//! Step time of the Gemma 4 engine-lane forward (`forward_lane_rows`) at 2..=N
//! rows vs the one-row decode (eager and captured graph), for HIPF-545.
//!
//! Usage: bench_lane_rows_gemma4 --model M.mq4 [--rows 2,3,4] [--pos 800] [--iters 20]
//! Lane `r` decodes at `pos + 37 * r`; the KV rows are whatever the arena holds
//! (zeros), which does not change the work.

use hipfire_arch_gemma4::config::Gemma4Config;
use hipfire_arch_gemma4::forward::{
    decode_step_device, decode_step_with_graph_device, forward_lane_rows, LaneRows,
};
use hipfire_arch_gemma4::gemma4::{Gemma4State, Gemma4Weights};
use hipfire_runtime::hfq::HfqFile;
use hipfire_runtime::llama::KvCache;
use std::time::Instant;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let arg = |name: &str| {
        argv.iter()
            .position(|a| a == name)
            .map(|i| argv[i + 1].clone())
    };
    let model = arg("--model").expect("--model");
    let rows: Vec<usize> = arg("--rows")
        .unwrap_or_else(|| "2,3,4".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let pos: usize = arg("--pos").map_or(800, |s| s.parse().unwrap());
    let iters: usize = arg("--iters").map_or(20, |s| s.parse().unwrap());
    let max_seq = 2048;
    let lanes = *rows.iter().max().unwrap();

    let mut gpu = rdna_compute::Gpu::init().expect("gpu");
    let hfq = HfqFile::open(std::path::Path::new(&model)).expect("open");
    let cfg = Gemma4Config::from_hfq(&hfq).expect("config");
    let weights = Gemma4Weights::load(&hfq, &cfg, &mut gpu).expect("weights");
    let mut state = Gemma4State::new_with_max_seq(&mut gpu, &cfg, max_seq).expect("state");
    let arena = |gpu: &mut rdna_compute::Gpu, kv: &KvCache| {
        KvCache::new_gpu_q8(
            gpu,
            kv.k_gpu.len(),
            kv.n_kv_heads,
            kv.head_dim,
            lanes * max_seq,
        )
        .expect("arena")
    };
    let sliding = arena(&mut gpu, &state.kv_sliding);
    let full = arena(&mut gpu, &state.kv_full);
    let token = 1000u32;

    let time = |gpu: &mut rdna_compute::Gpu, f: &mut dyn FnMut(&mut rdna_compute::Gpu)| {
        for _ in 0..3 {
            f(gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            f(gpu);
        }
        gpu.hip.device_synchronize().unwrap();
        t.elapsed().as_secs_f64() * 1e3 / iters as f64
    };

    let eager = time(&mut gpu, &mut |gpu| {
        decode_step_device(&cfg, &weights, &mut state, gpu, token, pos as u32).unwrap();
    });
    // A capture cannot synchronize, so a step profile (HIPF545_PROF) skips it.
    let graph = if std::env::var_os("HIPF545_PROF").is_some() {
        eager
    } else {
        time(&mut gpu, &mut |gpu| {
            decode_step_with_graph_device(&cfg, &weights, &mut state, gpu, token, pos as u32)
                .unwrap();
        })
    };
    println!("one-row decode: eager {eager:.2} ms, graph {graph:.2} ms");
    for &n in &rows {
        let positions: Vec<usize> = (0..n).map(|r| pos + 37 * r).collect();
        let tokens: Vec<u32> = (0..n as u32).map(|r| token + r).collect();
        let mut top1 = Vec::new();
        let ms = time(&mut gpu, &mut |gpu| {
            forward_lane_rows(
                &cfg,
                &weights,
                &mut state,
                gpu,
                &tokens,
                &LaneRows {
                    kv_sliding: &sliding,
                    kv_full: &full,
                    positions: &positions,
                    active_mask: (1u64 << n) - 1,
                },
                Some(&mut top1),
                None,
            )
            .unwrap();
        });
        println!(
            "lane forward {n} rows: {ms:.2} ms ({:.2} ms/row; graph one-row x{n} = {:.2} ms)",
            ms / n as f64,
            graph * n as f64
        );
    }
}
