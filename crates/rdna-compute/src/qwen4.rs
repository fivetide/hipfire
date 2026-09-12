// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Ordinary-HIP Qwen4 operation wrappers.
//!
//! These functions are explicit F32 contracts for the native trunk.  They do
//! not perform source I/O, allocate scratch, or silently choose a different
//! family implementation.  The parent carrier selects the gfx1151 Single
//! route before calling them.

use hip_bridge::{HipError, HipResult, KernargBlob};

use crate::{DType, Gpu, GpuTensor};

const QWEN4_OPS_SRC: &str = include_str!("../../../kernels/src/qwen4_ops.hip");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen4ComputeError {
    WrongDtype,
    WrongShape,
    UnsupportedArch,
}

impl std::fmt::Display for Qwen4ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongDtype => write!(f, "Qwen4 compute expects F32 tensors"),
            Self::WrongShape => write!(f, "Qwen4 compute tensor shape mismatch"),
            Self::UnsupportedArch => write!(f, "Qwen4 ordinary HIP path requires gfx1151"),
        }
    }
}

impl std::error::Error for Qwen4ComputeError {}

fn ensure_f32(tensor: &GpuTensor) -> HipResult<()> {
    if tensor.dtype != DType::F32 {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongDtype.to_string()));
    }
    Ok(())
}

fn ensure_gfx1151(gpu: &Gpu) -> HipResult<()> {
    if !gpu.arch_caps.is_gfx1151() {
        return Err(HipError::new(
            0,
            &Qwen4ComputeError::UnsupportedArch.to_string(),
        ));
    }
    Ok(())
}

fn blocks(elements: usize) -> u32 {
    elements.div_ceil(256) as u32
}

pub struct Qwen4GdnStep<'a> {
    pub q: &'a GpuTensor,
    pub k: &'a GpuTensor,
    pub v: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
    pub state: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

pub fn qwen4_gdn_step(gpu: &mut Gpu, p: &Qwen4GdnStep<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.q, p.k, p.v, p.gate, p.beta, p.state, p.output] {
        ensure_f32(tensor)?;
    }
    if p.key_heads == 0
        || p.value_heads == 0
        || p.value_heads % p.key_heads != 0
        || p.key_dim == 0
        || p.value_dim == 0
        || p.q.numel() != p.key_heads * p.key_dim
        || p.k.numel() != p.key_heads * p.key_dim
        || p.v.numel() != p.value_heads * p.value_dim
        || p.gate.numel() != p.value_heads
        || p.beta.numel() != p.value_heads
        || p.state.numel() != p.value_heads * p.value_dim * p.key_dim
        || p.output.numel() != p.value_heads * p.value_dim
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_gdn_step_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.q.buf.as_ptr());
    args.push_ptr(p.k.buf.as_ptr());
    args.push_ptr(p.v.buf.as_ptr());
    args.push_ptr(p.gate.buf.as_ptr());
    args.push_ptr(p.beta.buf.as_ptr());
    args.push_ptr(p.state.buf.as_ptr());
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(p.key_heads as i32);
    args.push_i32(p.value_heads as i32);
    args.push_i32(p.key_dim as i32);
    args.push_i32(p.value_dim as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_gdn_step_f32",
        [p.value_heads as u32, p.value_dim.div_ceil(256) as u32, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
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
    if p.channels == 0
        || p.history_rows != 9
        || p.kernel_size != 4
        || p.dilation != 3
        || p.cursor >= p.history_rows
        || p.input.numel() != p.channels
        || p.output.numel() != p.channels
        || p.kernel.numel() != p.channels * p.kernel_size
        || p.history.numel() != p.channels * p.history_rows
        || p.next_history.numel() != p.history.numel()
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_ple_depthwise_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.history, p.kernel, p.output, p.next_history] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(p.channels as i32);
    args.push_i32(p.history_rows as i32);
    args.push_i32(p.kernel_size as i32);
    args.push_i32(p.dilation as i32);
    args.push_i32(p.cursor as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_ple_depthwise_f32",
        [blocks(p.channels), 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct Qwen4QsaPool<'a> {
    pub raw_keys: &'a GpuTensor,
    pub pooled: &'a GpuTensor,
    pub block_count: usize,
    pub head_dim: usize,
}

pub fn qwen4_qsa_pool(gpu: &mut Gpu, p: &Qwen4QsaPool<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.raw_keys)?;
    ensure_f32(p.pooled)?;
    if p.block_count == 0
        || p.head_dim == 0
        || p.raw_keys.numel() < p.block_count * 4 * p.head_dim
        || p.pooled.numel() < p.block_count * p.head_dim
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_qsa_pool_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.raw_keys.buf.as_ptr());
    args.push_ptr(p.pooled.buf.as_ptr());
    args.push_i32(p.block_count as i32);
    args.push_i32(p.head_dim as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_qsa_pool_f32",
        [p.block_count as u32, blocks(p.head_dim), 1],
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
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_hc_prepare_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.branch_weight, p.normalized, p.branch] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(p.hidden as i32);
    args.push_f32(p.inv_rms);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_hc_prepare_f32",
        [blocks(p.hidden), 1, 1],
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
    if p.branch_count != 4
        || p.hidden == 0
        || p.branches.numel() != p.branch_count * p.hidden
        || p.output.numel() != p.hidden
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_hc_mix_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.branches.buf.as_ptr());
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(p.branch_count as i32);
    args.push_i32(p.hidden as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_hc_mix_f32",
        [blocks(p.hidden), 1, 1],
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
            eprintln!("skip: Qwen4 ordinary-HIP validation requires gfx1151");
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
    fn gdn_rejects_shape_before_kernel_launch() {
        let Some(mut gpu) = try_gfx1151_gpu() else {
            eprintln!("skip: no gfx1151 GPU");
            return;
        };
        let q = null_tensor(&[1], DType::F32);
        let k = null_tensor(&[2], DType::F32);
        let v = null_tensor(&[12], DType::F32);
        let gate = null_tensor(&[3], DType::F32);
        let beta = null_tensor(&[3], DType::F32);
        let state = null_tensor(&[24], DType::F32);
        let output = null_tensor(&[12], DType::F32);
        let params = Qwen4GdnStep {
            q: &q,
            k: &k,
            v: &v,
            gate: &gate,
            beta: &beta,
            state: &state,
            output: &output,
            key_heads: 1,
            value_heads: 3,
            key_dim: 2,
            value_dim: 4,
        };

        let error = qwen4_gdn_step(&mut gpu, &params).expect_err("invalid shape must fail");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);
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
