// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4-only ordinary-HIP helpers.
//!
//! These operations retain fixed Qwen4 policy/geometry (HC prepare/mix and
//! the legacy PLE depthwise path). Parameterized tensor operations live in
//! [`rdna_compute::tensor_ops`].

use hip_bridge::{HipError, HipResult, KernargBlob};
use rdna_compute::{DType, Gpu, GpuTensor};

const QWEN4_SPECIFIC_SRC: &str = include_str!("qwen4_specific.hip");

fn shape_error() -> HipError {
    HipError::new(0, &"tensor shape mismatch".to_string())
}

fn dtype_error() -> HipError {
    HipError::new(0, &"Qwen4 GPU operation expects F32 tensors".to_string())
}

fn arch_error() -> HipError {
    HipError::new(0, &"Qwen4 GPU operation requires gfx1151".to_string())
}

fn ensure_f32(tensor: &GpuTensor) -> HipResult<()> {
    if tensor.dtype == DType::F32 {
        Ok(())
    } else {
        Err(dtype_error())
    }
}

fn ensure_gfx1151(gpu: &Gpu) -> HipResult<()> {
    if gpu.arch_caps.is_gfx1151() {
        Ok(())
    } else {
        Err(arch_error())
    }
}

fn checked_i32(value: usize) -> HipResult<i32> {
    i32::try_from(value).map_err(|_| shape_error())
}

fn blocks(elements: usize) -> HipResult<u32> {
    let rounded = elements.checked_add(255).ok_or_else(shape_error)?;
    u32::try_from(rounded / 256).map_err(|_| shape_error())
}

pub struct Qwen4PleDepthwise<'a> {
    pub input: &'a GpuTensor,
    pub history: &'a GpuTensor,
    pub kernel: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub next_history: &'a GpuTensor,
    pub channels: usize,
    pub history_rows: usize,
    pub kernel_size: usize,
    pub dilation: usize,
    pub cursor: usize,
}

pub fn qwen4_ple_depthwise(gpu: &mut Gpu, p: &Qwen4PleDepthwise<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.input, p.history, p.kernel, p.output, p.next_history] {
        ensure_f32(tensor)?;
    }
    let kernel_elements = p
        .channels
        .checked_mul(p.kernel_size)
        .ok_or_else(shape_error)?;
    let history_elements = p
        .channels
        .checked_mul(p.history_rows)
        .ok_or_else(shape_error)?;
    if p.channels == 0
        || p.history_rows != 9
        || p.kernel_size != 4
        || p.dilation != 3
        || p.cursor >= p.history_rows
        || p.input.numel() != p.channels
        || p.output.numel() != p.channels
        || p.kernel.numel() != kernel_elements
        || p.history.numel() != history_elements
        || p.next_history.numel() != history_elements
    {
        return Err(shape_error());
    }
    let channels = checked_i32(p.channels)?;
    let history_rows = checked_i32(p.history_rows)?;
    let kernel_size = checked_i32(p.kernel_size)?;
    let dilation = checked_i32(p.dilation)?;
    let cursor = checked_i32(p.cursor)?;
    let grid = blocks(p.channels)?;
    gpu.ensure_kernel_public(
        "qwen4_specific",
        QWEN4_SPECIFIC_SRC,
        "qwen4_ple_depthwise_f32",
    )?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.history, p.kernel, p.output, p.next_history] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(channels);
    args.push_i32(history_rows);
    args.push_i32(kernel_size);
    args.push_i32(dilation);
    args.push_i32(cursor);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_ple_depthwise_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct Qwen4HcPrepare<'a> {
    pub input: &'a GpuTensor,
    pub branch_weight: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub branch: &'a GpuTensor,
    pub hidden: usize,
    pub inv_rms: f32,
}

pub fn qwen4_hc_prepare(gpu: &mut Gpu, p: &Qwen4HcPrepare<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.input, p.branch_weight, p.normalized, p.branch] {
        ensure_f32(tensor)?;
    }
    if p.hidden == 0
        || [p.input, p.branch_weight, p.normalized, p.branch]
            .iter()
            .any(|tensor| tensor.numel() != p.hidden)
    {
        return Err(shape_error());
    }
    let hidden = checked_i32(p.hidden)?;
    let grid = blocks(p.hidden)?;
    gpu.ensure_kernel_public("qwen4_specific", QWEN4_SPECIFIC_SRC, "qwen4_hc_prepare_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.branch_weight, p.normalized, p.branch] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(hidden);
    args.push_f32(p.inv_rms);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_hc_prepare_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct Qwen4HcMix<'a> {
    pub branches: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub branch_count: usize,
    pub hidden: usize,
}

pub fn qwen4_hc_mix(gpu: &mut Gpu, p: &Qwen4HcMix<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.branches)?;
    ensure_f32(p.output)?;
    let branch_elements = p
        .branch_count
        .checked_mul(p.hidden)
        .ok_or_else(shape_error)?;
    if p.branch_count != 4
        || p.hidden == 0
        || p.branches.numel() != branch_elements
        || p.output.numel() != p.hidden
    {
        return Err(shape_error());
    }
    let branch_count = checked_i32(p.branch_count)?;
    let hidden = checked_i32(p.hidden)?;
    let grid = blocks(p.hidden)?;
    gpu.ensure_kernel_public("qwen4_specific", QWEN4_SPECIFIC_SRC, "qwen4_hc_mix_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.branches.buf.as_ptr());
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(branch_count);
    args.push_i32(hidden);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_hc_mix_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_gfx1151_gpu() -> Option<Gpu> {
        let gpu = Gpu::init().ok()?;
        if !gpu.arch_caps.is_gfx1151() {
            eprintln!("skip: Qwen4-only ordinary-HIP validation requires gfx1151");
            return None;
        }
        Some(gpu)
    }

    fn null_tensor(shape: &[usize], dtype: DType) -> GpuTensor {
        let mut tensor = GpuTensor::null_for_test();
        tensor.shape = shape.to_vec();
        tensor.dtype = dtype;
        tensor
    }

    #[test]
    fn ple_depthwise_rejects_wrong_dtype_before_kernel_launch() {
        let Some(mut gpu) = try_gfx1151_gpu() else {
            eprintln!("skip: no gfx1151 GPU");
            return;
        };
        let input = null_tensor(&[1], DType::F16);
        let history = null_tensor(&[9], DType::F32);
        let kernel = null_tensor(&[4], DType::F32);
        let output = null_tensor(&[1], DType::F32);
        let next_history = null_tensor(&[9], DType::F32);
        let params = Qwen4PleDepthwise {
            input: &input,
            history: &history,
            kernel: &kernel,
            output: &output,
            next_history: &next_history,
            channels: 1,
            history_rows: 9,
            kernel_size: 4,
            dilation: 3,
            cursor: 0,
        };

        let error = qwen4_ple_depthwise(&mut gpu, &params).expect_err("wrong dtype must fail");
        assert!(error.to_string().contains("expects F32 tensors"));
        assert_eq!(gpu.last_launched_kernel(), None);
    }

    #[test]
    fn hc_mix_rejects_noncanonical_branch_count() {
        let Some(mut gpu) = try_gfx1151_gpu() else {
            eprintln!("skip: no gfx1151 GPU");
            return;
        };
        let branches = null_tensor(&[3], DType::F32);
        let output = null_tensor(&[1], DType::F32);
        let params = Qwen4HcMix {
            branches: &branches,
            output: &output,
            branch_count: 3,
            hidden: 1,
        };

        let error = qwen4_hc_mix(&mut gpu, &params).expect_err("branch count must be four");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);
    }
}
