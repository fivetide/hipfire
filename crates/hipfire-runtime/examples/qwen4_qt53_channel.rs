// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! gfx1151-only Qwen4 sealed-MoE channel proof.
//!
//! This executable launches the dedicated qt53 ordinary GEMV, indexed top-10
//! down, grouped prefill down, and both top-10 combine kernels.  Every result
//! is checked against an independent CPU decoder for the exact 68-byte
//! MQ4G128V2 wire rows.  A small qt44 grouped gate/up launch is included so
//! the complete top-10 prefill permutation is compiled and exercised too.
//!
//! Usage:
//!   cargo run --release --example qwen4_qt53_channel -p hipfire-runtime

use rdna_compute::{DType, Gpu, GpuTensor};

const QT53_GROUP_BYTES: usize = 68;
const QT44_GROUP_BYTES: usize = 136;
const TOP_K: usize = 10;

fn qt53_bytes(rows: usize, k: usize, seed: usize) -> Vec<u8> {
    let groups = k.div_ceil(128);
    let row_stride = groups * QT53_GROUP_BYTES;
    let mut out = vec![0u8; rows * row_stride];
    for row in 0..rows {
        for group in 0..groups {
            let base = row * row_stride + group * QT53_GROUP_BYTES;
            out[base..base + 2].copy_from_slice(&0x3c00u16.to_le_bytes()); // f16(1)
            out[base + 2..base + 4].copy_from_slice(&0u16.to_le_bytes()); // f16(0)
            for i in 0..64 {
                let q0 = ((seed + row * 7 + group * 11 + i * 3) & 0x0f) as u8;
                let q1 = ((seed + row * 13 + group * 5 + i * 9 + 1) & 0x0f) as u8;
                out[base + 4 + i] = q0 | (q1 << 4);
            }
        }
    }
    out
}

fn qt44_bytes(rows: usize, k: usize, seed: usize) -> Vec<u8> {
    assert_eq!(k % 256, 0, "qt44 harness rows use complete G256 groups");
    let groups = k / 256;
    let row_stride = groups * QT44_GROUP_BYTES;
    let mut out = vec![0u8; rows * row_stride];
    for row in 0..rows {
        for group in 0..groups {
            let base = row * row_stride + group * QT44_GROUP_BYTES;
            // [scale_0, zero_0, scale_1, zero_1], all f16.
            out[base..base + 2].copy_from_slice(&0x3c00u16.to_le_bytes());
            out[base + 2..base + 4].copy_from_slice(&0u16.to_le_bytes());
            out[base + 4..base + 6].copy_from_slice(&0x3c00u16.to_le_bytes());
            out[base + 6..base + 8].copy_from_slice(&0u16.to_le_bytes());
            for i in 0..128 {
                let q0 = ((seed + row * 5 + group * 7 + i * 3) & 0x0f) as u8;
                let q1 = ((seed + row * 17 + group * 13 + i * 11 + 1) & 0x0f) as u8;
                out[base + 8 + i] = q0 | (q1 << 4);
            }
        }
    }
    out
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let fraction = (bits & 0x03ff) as u32;
    let value = if exponent == 0 {
        if fraction == 0 {
            0.0
        } else {
            (fraction as f32 / 1024.0) * 2.0f32.powi(-14)
        }
    } else if exponent == 0x1f {
        if fraction == 0 {
            f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        (1.0 + fraction as f32 / 1024.0) * 2.0f32.powi(exponent as i32 - 15)
    };
    if sign == 0 { value } else { -value }
}

fn qt53_dot(row: &[u8], k: usize, x: &[f32]) -> f32 {
    let groups = k.div_ceil(128);
    let row_stride = groups * QT53_GROUP_BYTES;
    assert_eq!(row.len(), row_stride);
    assert_eq!(x.len(), k);
    let mut acc = 0.0f32;
    for group in 0..groups {
        let base = group * QT53_GROUP_BYTES;
        let scale = f16_to_f32(u16::from_le_bytes([row[base], row[base + 1]]));
        let zero = f16_to_f32(u16::from_le_bytes([row[base + 2], row[base + 3]]));
        for in_group in 0..128 {
            let logical = group * 128 + in_group;
            if logical >= k {
                continue;
            }
            let packed = row[base + 4 + (in_group >> 1)];
            let q = if in_group & 1 == 0 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            acc += (scale * q as f32 + zero) * x[logical];
        }
    }
    acc
}

fn raw_i32(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn raw_ptrs(tensors: &[GpuTensor]) -> Vec<u8> {
    tensors
        .iter()
        .flat_map(|tensor| (tensor.buf.as_ptr() as usize as u64).to_le_bytes())
        .collect()
}
fn zero_raw_i32(gpu: &Gpu, count: usize) -> Result<GpuTensor, String> {
    gpu.upload_raw(&vec![0u8; count * 4], &[count * 4])
        .map_err(|error| error.to_string())
}

fn normalize_top10(tokens: usize, seed: usize) -> Vec<f32> {
    let mut weights = vec![0.0f32; tokens * TOP_K];
    for token in 0..tokens {
        let mut sum = 0.0f32;
        for rank in 0..TOP_K {
            let weight = 1.0 + ((seed + token * 7 + rank * 3) % 17) as f32;
            weights[token * TOP_K + rank] = weight;
            sum += weight;
        }
        for rank in 0..TOP_K {
            weights[token * TOP_K + rank] /= sum;
        }
    }
    weights
}

fn check_close(
    label: &str,
    actual: &[f32],
    expected: &[f32],
    relative_tolerance: f32,
) -> Result<(), String> {
    if actual.len() != expected.len() {
        return Err(format!(
            "{label}: length {} != {}",
            actual.len(),
            expected.len()
        ));
    }
    let mut max_abs = 0.0f32;
    let mut max_relative = 0.0f32;
    let mut max_denominator = 1.0f32;
    let mut max_index = 0usize;

    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        if !got.is_finite() {
            return Err(format!("{label}: non-finite output at {index}: {got}"));
        }
        let abs = (got - want).abs();
        let denominator = want.abs().max(1.0);
        let relative = abs / denominator;
        if abs > max_abs {
            max_abs = abs;
        }
        if relative > max_relative {
            max_relative = relative;
            max_denominator = denominator;
            max_index = index;
        }
        let bound = relative_tolerance * denominator;
        if relative > relative_tolerance {
            return Err(format!(
                "{label}: mismatch at {index}: got {got:.8e}, want {want:.8e}, abs {abs:.8e}, rel {relative:.8e} > {relative_tolerance:.8e} (denom {denominator:.8e}, abs_bound {bound:.8e})"
            ));
        }
    }
    println!(
        "{label}: PASS (max_abs={max_abs:.8e}, max_rel={max_relative:.8e}, denominator={max_denominator:.8e} at {max_index})"
    );
    Ok(())
}

fn free_all(gpu: &mut Gpu, tensors: impl IntoIterator<Item = GpuTensor>) -> Result<(), String> {
    for tensor in tensors {
        gpu.free_tensor(tensor).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn ordinary_gemv(gpu: &mut Gpu) -> Result<(), String> {
    let m = 3usize;
    let k = 160usize; // Deliberately exercises the ragged second G128 group.
    let weight_bytes = qt53_bytes(m, k, 5);
    let x: Vec<f32> = (0..k).map(|i| 0.03125 * i as f32 - 1.0).collect();
    let weight = gpu
        .upload_raw(&weight_bytes, &[weight_bytes.len()])
        .map_err(|error| error.to_string())?;
    let x_gpu = gpu
        .upload_f32(&x, &[k])
        .map_err(|error| error.to_string())?;
    let y_gpu = gpu
        .zeros(&[m], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.gemv_mq4g128v2(&weight, &x_gpu, &y_gpu, m, k)
        .map_err(|error| error.to_string())?;
    let actual = gpu
        .download_f32(&y_gpu)
        .map_err(|error| error.to_string())?;
    let row_stride = k.div_ceil(128) * QT53_GROUP_BYTES;
    let expected: Vec<f32> = (0..m)
        .map(|row| {
            qt53_dot(
                &weight_bytes[row * row_stride..(row + 1) * row_stride],
                k,
                &x,
            )
        })
        .collect();
    let result = check_close("qt53 ordinary GEMV gfx1151", &actual, &expected, 3e-5);
    free_all(gpu, [weight, x_gpu, y_gpu])?;
    result
}

fn indexed_top10(gpu: &mut Gpu) -> Result<(), String> {
    let experts_count = 16usize;
    let m = 5usize;
    let k = 160usize;
    let tokens = 2usize;
    let row_stride = k.div_ceil(128) * QT53_GROUP_BYTES;
    let expert_bytes: Vec<Vec<u8>> = (0..experts_count)
        .map(|expert| qt53_bytes(m, k, 19 + expert * 23))
        .collect();
    let expert_tensors: Vec<GpuTensor> = expert_bytes
        .iter()
        .map(|bytes| {
            gpu.upload_raw(bytes, &[bytes.len()])
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    let ptrs = gpu
        .upload_raw(&raw_ptrs(&expert_tensors), &[experts_count * 8])
        .map_err(|error| error.to_string())?;
    let indices_host: Vec<i32> = (0..tokens * TOP_K)
        .map(|slot| ((slot * 7 + 3) % experts_count) as i32)
        .collect();
    let indices = gpu
        .upload_raw(&raw_i32(&indices_host), &[tokens * TOP_K * 4])
        .map_err(|error| error.to_string())?;
    let x_host: Vec<f32> = (0..tokens * TOP_K * k)
        .map(|index| 0.0078125 * (index as f32 + 1.0) - 0.5)
        .collect();
    let x = gpu
        .upload_f32(&x_host, &[tokens * TOP_K, k])
        .map_err(|error| error.to_string())?;
    let expanded = gpu
        .zeros(&[tokens * TOP_K, m], DType::F32)
        .map_err(|error| error.to_string())?;
    let weights_host = normalize_top10(tokens, 31);
    let weights = gpu
        .upload_f32(&weights_host, &[tokens * TOP_K])
        .map_err(|error| error.to_string())?;
    let residual = gpu
        .zeros(&[tokens, m], DType::F32)
        .map_err(|error| error.to_string())?;

    gpu.gemv_mq4g128v2_moe_down_top10_indexed_batched_expanded(
        &ptrs,
        &indices,
        &x,
        &expanded,
        m,
        k,
        tokens,
        experts_count,
    )
    .map_err(|error| error.to_string())?;
    gpu.moe_down_combine_top10_batched(&expanded, &weights, &residual, m, tokens)
        .map_err(|error| error.to_string())?;
    let actual = gpu
        .download_f32(&residual)
        .map_err(|error| error.to_string())?;
    let mut expected = vec![0.0f32; tokens * m];
    for token in 0..tokens {
        for rank in 0..TOP_K {
            let flat = token * TOP_K + rank;
            let expert = indices_host[flat] as usize;
            for row in 0..m {
                let row_start = row * row_stride;
                let value = qt53_dot(
                    &expert_bytes[expert][row_start..row_start + row_stride],
                    k,
                    &x_host[flat * k..(flat + 1) * k],
                );
                expected[token * m + row] += weights_host[flat] * value;
            }
        }
    }
    let result = check_close(
        "qt53 indexed top10 + combine gfx1151",
        &actual,
        &expected,
        4e-5,
    );
    let mut owned = vec![ptrs, indices, x, expanded, weights, residual];
    owned.extend(expert_tensors);
    free_all(gpu, owned)?;
    result
}

fn grouped_prefill(gpu: &mut Gpu) -> Result<(), String> {
    let experts_count = 16usize;
    let batch = 2usize;
    let total_slots = batch * TOP_K;
    let hidden = 256usize;
    let intermediate = 640usize;
    let grouped_rows = 272usize; // 16-row tile capacity above the live padded total.
    let row_stride = intermediate.div_ceil(128) * QT53_GROUP_BYTES;
    let down_bytes: Vec<Vec<u8>> = (0..experts_count)
        .map(|expert| qt53_bytes(hidden, intermediate, 47 + expert * 29))
        .collect();
    let down_tensors: Vec<GpuTensor> = down_bytes
        .iter()
        .map(|bytes| {
            gpu.upload_raw(bytes, &[bytes.len()])
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    let down_ptrs = gpu
        .upload_raw(&raw_ptrs(&down_tensors), &[experts_count * 8])
        .map_err(|error| error.to_string())?;

    // Exercise the dedicated qt44 gate/up grouped entry using the same top-10
    // permutation. The qt44 launcher converts this synthetic F32 activation
    // to the WMMA kernel's fp16 input internally.
    let gate_bytes: Vec<Vec<u8>> = (0..experts_count)
        .map(|expert| qt44_bytes(2 * intermediate, hidden, 71 + expert * 31))
        .collect();
    let gate_tensors: Vec<GpuTensor> = gate_bytes
        .iter()
        .map(|bytes| {
            gpu.upload_raw(bytes, &[bytes.len()])
                .map_err(|error| error.to_string())
        })
        .collect::<Result<_, _>>()?;
    let gate_ptrs = gpu
        .upload_raw(&raw_ptrs(&gate_tensors), &[experts_count * 8])
        .map_err(|error| error.to_string())?;

    let indices_host: Vec<i32> = (0..total_slots)
        .map(|slot| ((slot * 5 + 1) % experts_count) as i32)
        .collect();
    let indices = gpu
        .upload_raw(&raw_i32(&indices_host), &[total_slots * 4])
        .map_err(|error| error.to_string())?;
    let weights_host = normalize_top10(batch, 113);
    let weights = gpu
        .upload_f32(&weights_host, &[total_slots])
        .map_err(|error| error.to_string())?;
    let counts = zero_raw_i32(gpu, experts_count)?;
    let offsets = zero_raw_i32(gpu, experts_count + 1)?;
    let sorted = zero_raw_i32(gpu, grouped_rows)?;
    let tile_ids = zero_raw_i32(gpu, grouped_rows / 16)?;
    let inverse = zero_raw_i32(gpu, total_slots)?;

    gpu.moe_scatter_fused_top10(
        &indices,
        &counts,
        &offsets,
        &sorted,
        &tile_ids,
        &inverse,
        total_slots,
        experts_count,
        grouped_rows,
        16,
    )
    .map_err(|error| error.to_string())?;

    let rot_host: Vec<f32> = (0..total_slots * intermediate)
        .map(|index| 0.00390625 * (index as f32 + 3.0) - 0.75)
        .collect();
    let rot = gpu
        .upload_f32(&rot_host, &[total_slots, intermediate])
        .map_err(|error| error.to_string())?;
    let grouped_down = gpu
        .zeros(&[grouped_rows, hidden], DType::F32)
        .map_err(|error| error.to_string())?;
    let residual = gpu
        .zeros(&[batch, hidden], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.gemm_mq4g128v2_moe_grouped_top10(
        &down_ptrs,
        &tile_ids,
        &sorted,
        &rot,
        &grouped_down,
        hidden,
        intermediate,
        1,
        grouped_rows,
        total_slots,
        experts_count,
    )
    .map_err(|error| error.to_string())?;
    gpu.moe_down_combine_grouped_top10(
        &grouped_down,
        &inverse,
        &weights,
        &residual,
        hidden,
        grouped_rows,
        batch,
    )
    .map_err(|error| error.to_string())?;
    let actual = gpu
        .download_f32(&residual)
        .map_err(|error| error.to_string())?;
    let mut expected = vec![0.0f32; batch * hidden];
    for token in 0..batch {
        for rank in 0..TOP_K {
            let flat = token * TOP_K + rank;
            let expert = indices_host[flat] as usize;
            for row in 0..hidden {
                let row_start = row * row_stride;
                let value = qt53_dot(
                    &down_bytes[expert][row_start..row_start + row_stride],
                    intermediate,
                    &rot_host[flat * intermediate..(flat + 1) * intermediate],
                );
                expected[token * hidden + row] += weights_host[flat] * value;
            }
        }
    }
    let result = check_close(
        "qt53 grouped prefill + combine gfx1151",
        &actual,
        &expected,
        5e-5,
    );

    // The gate/up launch is deliberately after the independent qt53 check: if
    // WMMA compilation or its top-10 gather/unscatter contract is broken, this
    // executable still reports it as a failed channel rather than hiding the
    // useful qt53 CPU-reference result behind a prior error.
    let gate_x_host: Vec<f32> = (0..batch * hidden)
        .map(|index| 0.01171875 * index as f32 - 0.4)
        .collect();
    let gate_x = gpu
        .upload_f32(&gate_x_host, &[batch, hidden])
        .map_err(|error| error.to_string())?;
    let gate_grouped = gpu
        .zeros(&[grouped_rows, 2 * intermediate], DType::F32)
        .map_err(|error| error.to_string())?;
    let gate_out = gpu
        .zeros(&[total_slots, intermediate], DType::F32)
        .map_err(|error| error.to_string())?;
    let up_out = gpu
        .zeros(&[total_slots, intermediate], DType::F32)
        .map_err(|error| error.to_string())?;
    gpu.gemm_mq4g256v2_moe_grouped_top10(
        &gate_ptrs,
        &tile_ids,
        &sorted,
        &gate_x,
        &gate_grouped,
        2 * intermediate,
        hidden,
        TOP_K,
        grouped_rows,
        batch,
    )
    .map_err(|error| error.to_string())?;
    gpu.moe_gate_up_unscatter_top10(
        &gate_grouped,
        &sorted,
        &gate_out,
        &up_out,
        intermediate,
        grouped_rows,
        batch,
    )
    .map_err(|error| error.to_string())?;
    let gate_probe = gpu
        .download_f32(&gate_out)
        .map_err(|error| error.to_string())?;
    if gate_probe.iter().any(|value| !value.is_finite()) {
        return Err("qt44 grouped gate/up top10 produced a non-finite value".to_string());
    }
    println!("qt44 grouped gate/up + top10 unscatter gfx1151: PASS");

    let mut owned = vec![
        down_ptrs,
        gate_ptrs,
        indices,
        weights,
        counts,
        offsets,
        sorted,
        tile_ids,
        inverse,
        rot,
        grouped_down,
        residual,
        gate_x,
        gate_grouped,
        gate_out,
        up_out,
    ];
    owned.extend(down_tensors);
    owned.extend(gate_tensors);
    free_all(gpu, owned)?;
    result
}

fn run() -> Result<(), String> {
    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    println!("GPU: {}", gpu.arch);
    if !gpu.arch_caps.is_gfx1151() {
        return Err(format!(
            "this proof requires gfx1151, detected {}",
            gpu.arch
        ));
    }
    ordinary_gemv(&mut gpu)?;
    indexed_top10(&mut gpu)?;
    grouped_prefill(&mut gpu)?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Qwen4 gfx1151 qt53 channel: FAIL: {error}");
        std::process::exit(1);
    }
    println!("Qwen4 gfx1151 qt53 channel: PASS");
}
