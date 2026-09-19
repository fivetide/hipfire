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
const QSA_SELECT_PARALLEL_THREADS: u32 = 256;
// gfx1151's ordinary-HIP dynamic LDS ceiling. Oversized score arrays retain
// the serial kernel so this tuning never changes the existing large-shape path.
const QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES: usize = 64 * 1024;
const QSA_ATTENTION_PARALLEL_THREADS: u32 = 256;
const QSA_ATTENTION_LDS_BYTES_PER_ROW: usize = 8; // F32 score + i32 token.
                                                  // gfx1151's ordinary-HIP dynamic LDS ceiling; oversized attention rows keep
                                                  // the serial kernel and unchanged launch contract.
const QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES: usize = 64 * 1024;

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
    let kernel = if p.key_dim == 128 && p.value_dim == 128 {
        "qwen4_gdn_step_shared_norm128_gfx1151"
    } else {
        "qwen4_gdn_step_f32"
    };
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, kernel)?;
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
    args.push_f32((p.key_dim as f32).sqrt().recip());
    args.pad_to(16);
    gpu.launch_kernel_blob(
        kernel,
        [p.value_heads as u32, p.value_dim.div_ceil(256) as u32, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

/// Device-side F32 -> BF16 storage -> F32 conversion at a source activation
/// boundary.  `scratch` is caller-owned and is allocated with the forward
/// arena; no host transfer or per-token allocation occurs.
pub struct Qwen4GdnBf16Roundtrip<'a> {
    pub input: &'a GpuTensor,
    pub scratch: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub elements: usize,
}

pub fn qwen4_gdn_bf16_roundtrip(gpu: &mut Gpu, p: &Qwen4GdnBf16Roundtrip<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.input)?;
    ensure_f32(p.output)?;
    if p.scratch.dtype != DType::BF16
        || p.elements == 0
        || p.input.numel() < p.elements
        || p.scratch.numel() < p.elements
        || p.output.numel() < p.elements
        || p.elements > i32::MAX as usize
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_gdn_bf16_roundtrip_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.input.buf.as_ptr());
    args.push_ptr(p.scratch.buf.as_ptr());
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(p.elements as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_gdn_bf16_roundtrip_f32",
        [blocks(p.elements), 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}
/// Source-exact BF16 product and residual add for the Qwen4 shared expert.
pub struct Qwen4Bf16ScaledAdd<'a> {
    pub residual: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub scalar: &'a GpuTensor,
    pub elements: usize,
}

pub fn qwen4_bf16_scaled_add(gpu: &mut Gpu, p: &Qwen4Bf16ScaledAdd<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.residual)?;
    ensure_f32(p.value)?;
    ensure_f32(p.scalar)?;
    if p.elements == 0
        || p.elements > i32::MAX as usize
        || p.residual.numel() < p.elements
        || p.value.numel() < p.elements
        || p.scalar.numel() == 0
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_bf16_scaled_add_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.residual.buf.as_ptr());
    args.push_ptr(p.value.buf.as_ptr());
    args.push_ptr(p.scalar.buf.as_ptr());
    args.push_i32(p.elements as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_bf16_scaled_add_f32",
        [blocks(p.elements), 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}
/// Source-exact BF16 product/residual add for a row-batched Qwen4 shared expert.
pub struct Qwen4Bf16ScaledAddBatched<'a> {
    pub residual: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub scalar: &'a GpuTensor,
    pub rows: usize,
    pub elements: usize,
}

pub fn qwen4_bf16_scaled_add_batched(
    gpu: &mut Gpu,
    p: &Qwen4Bf16ScaledAddBatched<'_>,
) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.residual)?;
    ensure_f32(p.value)?;
    ensure_f32(p.scalar)?;
    if p.rows == 0
        || p.elements == 0
        || p.rows.checked_mul(p.elements).is_none()
        || p.residual.numel() < p.rows * p.elements
        || p.value.numel() < p.rows * p.elements
        || p.scalar.numel() < p.rows
        || p.rows > i32::MAX as usize
        || p.elements > i32::MAX as usize
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public(
        "qwen4_ops",
        QWEN4_OPS_SRC,
        "qwen4_bf16_scaled_add_batched_f32",
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.residual.buf.as_ptr());
    args.push_ptr(p.value.buf.as_ptr());
    args.push_ptr(p.scalar.buf.as_ptr());
    args.push_i32(p.rows as i32);
    args.push_i32(p.elements as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_bf16_scaled_add_batched_f32",
        [blocks(p.rows * p.elements), 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct Qwen4HcRead<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub low: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
    pub rank: usize,
}

pub fn qwen4_hc_read(gpu: &mut Gpu, p: &Qwen4HcRead<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.input, p.low, p.up, p.normalized, p.mixed] {
        ensure_f32(tensor)?;
    }
    if p.norm_weight.dtype != DType::BF16
        || p.branches == 0
        || p.hidden == 0
        || p.rank == 0
        || p.input.numel() != p.branches * p.hidden
        || p.norm_weight.numel() != p.input.numel()
        || p.low.numel() != p.rank
        || p.up.numel() != p.input.numel() * p.rank
        || p.normalized.numel() != p.input.numel()
        || p.mixed.numel() != p.hidden
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_hc_read_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.norm_weight, p.low, p.up, p.normalized, p.mixed] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(p.branches as i32);
    args.push_i32(p.hidden as i32);
    args.push_i32(p.rank as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_hc_read_f32",
        [1, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

/// HC read where the up projection has already been evaluated to one F32
/// gate logit per branch/hidden column.
pub struct Qwen4HcReadProjected<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
}

pub fn qwen4_hc_read_projected(gpu: &mut Gpu, p: &Qwen4HcReadProjected<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.input, p.up, p.normalized, p.mixed] {
        ensure_f32(tensor)?;
    }
    let wide = p.branches * p.hidden;
    if p.norm_weight.dtype != DType::BF16
        || p.branches == 0
        || p.hidden == 0
        || wide == 0
        || p.input.numel() == 0
        || p.input.numel() % wide != 0
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    let rows = p.input.numel() / wide;
    if p.norm_weight.numel() != wide
        || p.up.numel() != p.input.numel()
        || p.normalized.numel() != p.input.numel()
        || p.mixed.numel() != rows * p.hidden
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_hc_read_projected_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.norm_weight, p.up, p.normalized, p.mixed] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(p.branches as i32);
    args.push_i32(p.hidden as i32);
    args.push_i32(rows as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_hc_read_projected_f32",
        [blocks(wide), rows as u32, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct Qwen4HcWrite<'a> {
    pub input: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub gates: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
}

pub fn qwen4_hc_write(gpu: &mut Gpu, p: &Qwen4HcWrite<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.input, p.normalized, p.mixed, p.gates, p.output] {
        ensure_f32(tensor)?;
    }
    let wide = p.branches * p.hidden;
    if p.branches == 0
        || p.hidden == 0
        || wide == 0
        || p.input.numel() == 0
        || p.input.numel() % wide != 0
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    let rows = p.input.numel() / wide;
    if p.normalized.numel() != p.input.numel()
        || p.mixed.numel() != rows * p.hidden
        || p.gates.numel() != rows * p.branches
        || p.output.numel() != p.input.numel()
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_hc_write_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.normalized, p.mixed, p.gates, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(p.branches as i32);
    args.push_i32(p.hidden as i32);
    args.push_i32(rows as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_hc_write_f32",
        [blocks(wide), rows as u32, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct Qwen4HcFinal<'a> {
    pub normalized: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
}

pub fn qwen4_hc_final(gpu: &mut Gpu, p: &Qwen4HcFinal<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.normalized, p.output] {
        ensure_f32(tensor)?;
    }
    if p.branches == 0
        || p.hidden == 0
        || p.normalized.numel() != p.branches * p.hidden
        || p.output.numel() != p.hidden
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_hc_final_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.normalized.buf.as_ptr());
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(p.branches as i32);
    args.push_i32(p.hidden as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_hc_final_f32",
        [blocks(p.hidden), 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}
pub struct Qwen4HcNorm<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
}

pub fn qwen4_hc_norm(gpu: &mut Gpu, p: &Qwen4HcNorm<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.input)?;
    ensure_f32(p.normalized)?;
    let wide = p.branches * p.hidden;
    if p.norm_weight.dtype != DType::BF16
        || p.branches == 0
        || p.hidden == 0
        || wide == 0
        || p.input.numel() == 0
        || p.input.numel() % wide != 0
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    let rows = p.input.numel() / wide;
    if p.norm_weight.numel() != wide || p.normalized.numel() != p.input.numel() {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_hc_norm_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.input.buf.as_ptr());
    args.push_ptr(p.norm_weight.buf.as_ptr());
    args.push_ptr(p.normalized.buf.as_ptr());
    args.push_i32(p.branches as i32);
    args.push_i32(p.hidden as i32);
    args.push_i32(rows as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_hc_norm_f32",
        [p.branches as u32, rows as u32, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}
pub struct Qwen4GdnConv<'a> {
    pub input: &'a GpuTensor,
    pub kernel: &'a GpuTensor,
    pub history: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub next_history: &'a GpuTensor,
    pub channels: usize,
    pub history_rows: usize,
    pub kernel_size: usize,
    pub cursor: usize,
}

pub fn qwen4_gdn_conv(gpu: &mut Gpu, p: &Qwen4GdnConv<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.input, p.history, p.output, p.next_history] {
        ensure_f32(tensor)?;
    }
    if p.kernel.dtype != DType::BF16
        || p.channels == 0
        || p.history_rows != p.kernel_size.saturating_sub(1)
        || p.cursor >= p.history_rows.max(1)
        || p.input.numel() != p.channels
        || p.kernel.numel() != p.channels * p.kernel_size
        || p.history.numel() != p.channels * p.history_rows
        || p.output.numel() != p.channels
        || p.next_history.numel() != p.history.numel()
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_gdn_conv_bf16_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.kernel, p.history, p.output, p.next_history] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(p.channels as i32);
    args.push_i32(p.history_rows as i32);
    args.push_i32(p.kernel_size as i32);
    args.push_i32(p.cursor as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_gdn_conv_bf16_f32",
        [blocks(p.channels), 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct Qwen4GdnParams<'a> {
    pub a: &'a GpuTensor,
    pub b: &'a GpuTensor,
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
}

pub fn qwen4_gdn_params(gpu: &mut Gpu, p: &Qwen4GdnParams<'_>, heads: usize) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.a, p.b, p.gate, p.beta] {
        ensure_f32(tensor)?;
    }
    if p.a_log.dtype != DType::BF16
        || p.dt_bias.dtype != DType::BF16
        || p.a.numel() != heads
        || p.b.numel() != heads
        || p.a_log.numel() != heads
        || p.dt_bias.numel() != heads
        || p.gate.numel() != heads
        || p.beta.numel() != heads
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_gdn_params_bf16_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(heads as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_gdn_params_bf16_f32",
        [blocks(heads), 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub fn qwen4_gdn_params_f32(gpu: &mut Gpu, p: &Qwen4GdnParams<'_>, heads: usize) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        ensure_f32(tensor)?;
    }
    if p.a.numel() != heads
        || p.b.numel() != heads
        || p.a_log.numel() != heads
        || p.dt_bias.numel() != heads
        || p.gate.numel() != heads
        || p.beta.numel() != heads
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_gdn_params_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(heads as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_gdn_params_f32",
        [blocks(heads), 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct Qwen4GdnGate<'a> {
    pub recurrent_output: &'a GpuTensor,
    pub z: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub value_heads: usize,
    pub value_dim: usize,
}

pub fn qwen4_gdn_gate(gpu: &mut Gpu, p: &Qwen4GdnGate<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.recurrent_output, p.z, p.output] {
        ensure_f32(tensor)?;
    }
    if p.norm.dtype != DType::BF16
        || p.value_heads == 0
        || p.value_dim == 0
        || p.recurrent_output.numel() != p.value_heads * p.value_dim
        || p.z.numel() != p.recurrent_output.numel()
        || p.norm.numel() != p.value_dim
        || p.output.numel() != p.recurrent_output.numel()
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_gdn_gate_bf16_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.recurrent_output, p.z, p.norm, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(p.value_heads as i32);
    args.push_i32(p.value_dim as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_gdn_gate_bf16_f32",
        [p.value_heads as u32, blocks(p.value_dim), 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}
pub struct Qwen4QsaNormRope<'a> {
    pub values: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub heads: usize,
    pub head_dim: usize,
    /// Distance, in elements, between consecutive query/key heads.
    pub head_stride: usize,
    pub position: usize,
    pub rotary_dim: usize,
}

pub fn qwen4_qsa_norm_rope(gpu: &mut Gpu, p: &Qwen4QsaNormRope<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.values)?;
    if p.norm.dtype != DType::BF16
        || p.heads == 0
        || p.head_dim == 0
        || p.head_dim > 256
        || p.head_stride < p.head_dim
        || p.rotary_dim == 0
        || p.rotary_dim % 2 != 0
        || p.rotary_dim > p.head_dim
        || p.values.numel() < (p.heads - 1) * p.head_stride + p.head_dim
        || p.norm.numel() != p.head_dim
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_qsa_norm_rope_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.values.buf.as_ptr());
    args.push_ptr(p.norm.buf.as_ptr());
    args.push_i32(p.heads as i32);
    args.push_i32(p.head_dim as i32);
    args.push_i32(p.head_stride as i32);
    args.push_i32(p.position as i32);
    args.push_i32(p.rotary_dim as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_qsa_norm_rope_f32",
        [p.heads as u32, blocks(p.head_dim), 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct Qwen4QsaCacheAppend<'a> {
    pub key: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub position: usize,
    pub kv_width: usize,
}

pub fn qwen4_qsa_cache_append(gpu: &mut Gpu, p: &Qwen4QsaCacheAppend<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.key, p.value, p.full_keys, p.full_values] {
        ensure_f32(tensor)?;
    }
    if p.kv_width == 0
        || p.key.numel() != p.kv_width
        || p.value.numel() != p.kv_width
        || p.position * p.kv_width + p.kv_width > p.full_keys.numel()
        || p.full_keys.numel() != p.full_values.numel()
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_qsa_cache_append_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.key, p.value, p.full_keys, p.full_values] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(p.position as i32);
    args.push_i32(p.kv_width as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_qsa_cache_append_f32",
        [blocks(p.kv_width), 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct Qwen4QsaSelect<'a> {
    pub query: &'a GpuTensor,
    pub pooled: &'a GpuTensor,
    pub selected: &'a GpuTensor,
    pub block_count: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub budget_blocks: usize,
    pub compress: usize,
    pub visible: usize,
    pub capacity: usize,
}

pub fn qwen4_qsa_select(gpu: &mut Gpu, p: &Qwen4QsaSelect<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.query, p.pooled] {
        ensure_f32(tensor)?;
    }
    if p.selected.dtype != DType::Raw
        || p.index_heads == 0
        || p.index_dim == 0
        || p.compress == 0
        || p.capacity == 0
        || p.query.numel() != p.index_heads * p.index_dim
        || p.pooled.numel() < p.block_count * p.index_dim
        || p.selected.numel() < p.capacity
        || p.visible == 0
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    let (kernel_name, block, shared_mem) =
        match p.block_count.checked_mul(std::mem::size_of::<f32>()) {
            Some(bytes)
                if bytes <= QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES && bytes <= u32::MAX as usize =>
            {
                (
                    "qwen4_qsa_select_f32",
                    [QSA_SELECT_PARALLEL_THREADS, 1, 1],
                    bytes as u32,
                )
            }
            _ => ("qwen4_qsa_select_f32_serial", [1, 1, 1], 0),
        };
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, kernel_name)?;
    let mut args = KernargBlob::new();
    for tensor in [p.query, p.pooled, p.selected] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for value in [
        p.block_count,
        p.index_heads,
        p.index_dim,
        p.budget_blocks,
        p.compress,
        p.visible,
        p.capacity,
    ] {
        args.push_i32(value as i32);
    }
    args.pad_to(16);
    gpu.launch_kernel_blob(
        kernel_name,
        [1, 1, 1],
        block,
        shared_mem,
        args.as_mut_slice(),
    )
}
/// Device-side stable reuse of a prior MTP QSA selection row.
///
/// `selected` is a byte-addressed [`DType::Raw`] allocation containing i32
/// indices. `selected_len_out` is a persistent four-byte Raw scalar written by
/// the kernel; callers can read only that scalar when the logical span changes,
/// never the selected row itself.
pub struct Qwen4QsaReuseSelection<'a> {
    pub selected: &'a GpuTensor,
    pub selected_len: usize,
    pub position: usize,
    pub capacity: usize,
    pub selected_len_out: &'a GpuTensor,
}

pub fn qwen4_qsa_reuse_selection(gpu: &mut Gpu, p: &Qwen4QsaReuseSelection<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    let selected_bytes = p
        .capacity
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()))?;
    if p.selected.dtype != DType::Raw
        || p.selected_len > p.capacity
        || p.capacity == 0
        || p.selected.numel() < selected_bytes
        || p.selected_len_out.dtype != DType::Raw
        || p.selected_len_out.numel() < std::mem::size_of::<i32>()
        || p.selected_len > i32::MAX as usize
        || p.position > i32::MAX as usize
        || p.capacity > i32::MAX as usize
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_qsa_reuse_selection")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.selected.buf.as_ptr());
    args.push_ptr(p.selected_len_out.buf.as_ptr());
    args.push_i32(p.selected_len as i32);
    args.push_i32(p.position as i32);
    args.push_i32(p.capacity as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_qsa_reuse_selection",
        [1, 1, 1],
        [1, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct Qwen4QsaPoolRope<'a> {
    pub raw_keys: &'a GpuTensor,
    pub pooled: &'a GpuTensor,
    /// Learned index-key RMSNorm. `None` is reserved for synthetic parity
    /// buffers, which intentionally retain the plain-RMS behavior.
    pub norm: Option<&'a GpuTensor>,
    pub block_count: usize,
    pub compress: usize,
    pub index_dim: usize,
}

pub fn qwen4_qsa_pool_rope(gpu: &mut Gpu, p: &Qwen4QsaPoolRope<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.raw_keys, p.pooled] {
        ensure_f32(tensor)?;
    }
    if let Some(norm) = p.norm {
        if norm.dtype != DType::BF16 || norm.numel() != p.index_dim {
            return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
        }
    }
    if p.block_count == 0
        || p.compress == 0
        || p.index_dim == 0
        || p.index_dim % 2 != 0
        || p.raw_keys.numel() < p.block_count * p.compress * p.index_dim
        || p.pooled.numel() < p.block_count * p.index_dim
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_qsa_pool_rope_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.raw_keys.buf.as_ptr());
    args.push_ptr(p.pooled.buf.as_ptr());
    args.push_ptr(
        p.norm
            .map(|norm| norm.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut()),
    );
    args.push_i32(p.block_count as i32);
    args.push_i32(p.compress as i32);
    args.push_i32(p.index_dim as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_qsa_pool_rope_f32",
        [p.block_count as u32, p.index_dim.div_ceil(256) as u32, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct Qwen4QsaAttention<'a> {
    pub q_with_gate: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub selected: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub selected_len: usize,
    pub full_capacity: usize,
}

pub fn qwen4_qsa_attention(gpu: &mut Gpu, p: &Qwen4QsaAttention<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.q_with_gate, p.full_keys, p.full_values, p.output] {
        ensure_f32(tensor)?;
    }
    if p.selected.dtype != DType::Raw
        || p.n_heads == 0
        || p.n_kv_heads == 0
        || p.n_heads % p.n_kv_heads != 0
        || p.q_with_gate.numel() != 2 * p.n_heads * p.head_dim
        || p.output.numel() != p.n_heads * p.head_dim
        || p.full_keys.numel() < p.full_capacity * p.n_kv_heads * p.head_dim
        || p.full_values.numel() < p.full_keys.numel()
        || p.selected.numel() < p.selected_len
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    let (kernel_name, shared_mem) =
        match p.selected_len.checked_mul(QSA_ATTENTION_LDS_BYTES_PER_ROW) {
            Some(bytes)
                if bytes <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES && bytes <= u32::MAX as usize =>
            {
                ("qwen4_qsa_attention_f32", bytes as u32)
            }
            _ => ("qwen4_qsa_attention_f32_serial", 0),
        };
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, kernel_name)?;
    let mut args = KernargBlob::new();
    for tensor in [
        p.q_with_gate,
        p.full_keys,
        p.full_values,
        p.selected,
        p.output,
    ] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for value in [
        p.n_heads,
        p.n_kv_heads,
        p.head_dim,
        p.selected_len,
        p.full_capacity,
    ] {
        args.push_i32(value as i32);
    }
    args.pad_to(16);
    gpu.launch_kernel_blob(
        kernel_name,
        [p.n_heads as u32, blocks(p.head_dim), 1],
        [QSA_ATTENTION_PARALLEL_THREADS, 1, 1],
        shared_mem,
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
/// In-place scalar scaling for learned HC branch projections.
pub struct Qwen4Scale<'a> {
    pub values: &'a GpuTensor,
    pub scale: f32,
}

pub fn qwen4_scale(gpu: &mut Gpu, p: &Qwen4Scale<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.values)?;
    if p.values.numel() == 0 {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_scale_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.values.buf.as_ptr());
    args.push_i32(p.values.numel() as i32);
    args.push_f32(p.scale);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_scale_f32",
        [blocks(p.values.numel()), 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

/// Device-side greedy top-1 over contiguous F32 logits rows.
pub struct Qwen4Argmax<'a> {
    pub logits: &'a GpuTensor,
    pub indices: &'a GpuTensor,
    pub rows: usize,
    pub vocab: usize,
}

pub fn qwen4_argmax(gpu: &mut Gpu, p: &Qwen4Argmax<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.logits)?;
    if p.rows == 0
        || p.vocab == 0
        || p.logits.numel() != p.rows * p.vocab
        || p.indices.dtype != DType::Raw
        || p.indices.numel() < p.rows
    {
        return Err(HipError::new(0, &Qwen4ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("qwen4_ops", QWEN4_OPS_SRC, "qwen4_argmax_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.logits.buf.as_ptr());
    args.push_ptr(p.indices.buf.as_ptr());
    args.push_i32(p.rows as i32);
    args.push_i32(p.vocab as i32);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "qwen4_argmax_f32",
        [p.rows as u32, 1, 1],
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
    #[test]
    fn qsa_reuse_selection_rejects_byte_capacity_mismatch() {
        let Some(mut gpu) = try_gfx1151_gpu() else {
            eprintln!("skip: no gfx1151 GPU");
            return;
        };
        let selected = null_tensor(&[3], DType::Raw);
        let selected_len_out = null_tensor(&[4], DType::Raw);
        let params = Qwen4QsaReuseSelection {
            selected: &selected,
            selected_len: 1,
            position: 0,
            capacity: 1,
            selected_len_out: &selected_len_out,
        };

        let error = qwen4_qsa_reuse_selection(&mut gpu, &params)
            .expect_err("byte capacity must be checked");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);
    }

    #[test]
    fn qsa_reuse_selection_preserves_order_and_boundaries() {
        let Some(mut gpu) = try_gfx1151_gpu() else {
            eprintln!("skip: no gfx1151 GPU");
            return;
        };
        let capacity = 4usize;
        let mut selected = gpu
            .zeros(&[capacity * std::mem::size_of::<i32>()], DType::Raw)
            .expect("selected allocation");
        let selected_len_out = gpu
            .zeros(&[std::mem::size_of::<i32>()], DType::Raw)
            .expect("length allocation");
        let cases: [([i32; 4], usize, usize, [i32; 4], i32); 4] = [
            ([-1, -1, -1, -1], 0usize, 7usize, [7, -1, -1, -1], 1),
            ([0, 3, 1, 99], 4, 2, [0, 1, 2, -1], 3),
            ([0, 2, 1, -1], 3, 2, [0, 2, 1, -1], 3),
            ([0, 1, 2, 3], 4, 4, [0, 1, 2, 3], 4),
        ];
        for (input, selected_len, position, expected, expected_len) in cases {
            let input_bytes = input
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>();
            gpu.hip
                .memcpy_htod(&selected.buf, &input_bytes)
                .expect("upload selected row");
            gpu.hip
                .memset(&selected_len_out.buf, 0, selected_len_out.byte_size())
                .expect("clear selected length");
            qwen4_qsa_reuse_selection(
                &mut gpu,
                &Qwen4QsaReuseSelection {
                    selected: &selected,
                    selected_len,
                    position,
                    capacity,
                    selected_len_out: &selected_len_out,
                },
            )
            .expect("reuse selection");

            let mut output_bytes = vec![0u8; capacity * std::mem::size_of::<i32>()];
            gpu.hip
                .memcpy_dtoh(&mut output_bytes, &selected.buf)
                .expect("download selected row");
            let output = output_bytes
                .chunks_exact(std::mem::size_of::<i32>())
                .map(|bytes| i32::from_ne_bytes(bytes.try_into().expect("i32 bytes")))
                .collect::<Vec<_>>();
            assert_eq!(output, expected);
            let mut length_bytes = [0u8; std::mem::size_of::<i32>()];
            gpu.hip
                .memcpy_dtoh(&mut length_bytes, &selected_len_out.buf)
                .expect("download selected length");
            assert_eq!(i32::from_ne_bytes(length_bytes), expected_len);
        }
        gpu.free_tensor(selected).expect("free selected");
        gpu.free_tensor(selected_len_out)
            .expect("free selected length");
    }
    #[test]
    fn qsa_norm_rope_matches_production_shape_and_preserves_gates() {
        let Some(mut gpu) = try_gfx1151_gpu() else {
            eprintln!("skip: no gfx1151 GPU");
            return;
        };
        const HEADS: usize = 2;
        const HEAD_DIM: usize = 256;
        const HEAD_STRIDE: usize = 2 * HEAD_DIM;
        const ROTARY_DIM: usize = 64;
        const POSITION: usize = 11;
        let mut input = vec![0.0f32; HEADS * HEAD_STRIDE];
        for head in 0..HEADS {
            for channel in 0..HEAD_DIM {
                input[head * HEAD_STRIDE + channel] =
                    (((head * HEAD_DIM + channel) * 37 % 251) as f32 - 125.0) / 37.0;
                input[head * HEAD_STRIDE + HEAD_DIM + channel] =
                    1000.0 + (head * HEAD_DIM + channel) as f32;
            }
        }
        let values = gpu
            .upload_f32(&input, &[input.len()])
            .expect("query/gate upload");
        let norm = gpu
            .zeros(&[HEAD_DIM], DType::BF16)
            .expect("zero norm allocation");
        qwen4_qsa_norm_rope(
            &mut gpu,
            &Qwen4QsaNormRope {
                values: &values,
                norm: &norm,
                heads: HEADS,
                head_dim: HEAD_DIM,
                head_stride: HEAD_STRIDE,
                position: POSITION,
                rotary_dim: ROTARY_DIM,
            },
        )
        .expect("QSA norm/RoPE");
        let actual = gpu.download_f32(&values).expect("query/gate download");
        let mut expected = input.clone();
        for head in 0..HEADS {
            let start = head * HEAD_STRIDE;
            let inv = (input[start..start + HEAD_DIM]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / HEAD_DIM as f32
                + 1.0e-6)
                .sqrt()
                .recip();
            for channel in 0..HEAD_DIM {
                expected[start + channel] = input[start + channel] * inv;
            }
            for channel in 0..ROTARY_DIM / 2 {
                let angle = POSITION as f32
                    / 10_000_000.0f32.powf(2.0 * channel as f32 / ROTARY_DIM as f32);
                let (sine, cosine) = angle.sin_cos();
                let first = input[start + channel] * inv;
                let second = input[start + ROTARY_DIM / 2 + channel] * inv;
                expected[start + channel] = first * cosine - second * sine;
                expected[start + ROTARY_DIM / 2 + channel] = first * sine + second * cosine;
            }
        }
        for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() <= 1.0e-4,
                "QSA norm/RoPE mismatch at {index}: got {got}, expected {want}"
            );
        }
        gpu.free_tensor(values).expect("free query/gate");
        gpu.free_tensor(norm).expect("free norm");
    }
}
