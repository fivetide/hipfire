// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Ordinary-HIP tensor operation wrappers.
//!
//! These functions are explicit F32 contracts for the supported trunk.  They
//! do not perform source I/O, allocate scratch, or silently choose a different
//! family implementation.  The caller selects the supported gfx1151 route
//! before calling them.

use hip_bridge::{HipError, HipResult, KernargBlob};

use crate::{DType, Gpu, GpuTensor};

pub(crate) const TENSOR_OPS_SRC: &str = include_str!("../../../kernels/src/tensor_ops.hip");
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
pub enum ComputeError {
    WrongDtype,
    WrongShape,
    UnsupportedArch,
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongDtype => write!(f, "tensor operation expects F32 tensors"),
            Self::WrongShape => write!(f, "tensor operation tensor shape mismatch"),
            Self::UnsupportedArch => write!(f, "ordinary HIP tensor operation requires gfx1151"),
        }
    }
}

impl std::error::Error for ComputeError {}

pub(crate) fn ensure_f32(tensor: &GpuTensor) -> HipResult<()> {
    if tensor.dtype != DType::F32 {
        return Err(HipError::new(0, &ComputeError::WrongDtype.to_string()));
    }
    Ok(())
}

pub(crate) fn ensure_gfx1151(gpu: &Gpu) -> HipResult<()> {
    if !gpu.arch_caps.is_gfx1151() {
        return Err(HipError::new(0, &ComputeError::UnsupportedArch.to_string()));
    }
    Ok(())
}

const KERNEL_BLOCK: usize = 256;
const MAX_KERNEL_EXTENT: usize = i32::MAX as usize;

fn checked_extent(value: usize, what: &'static str) -> HipResult<usize> {
    if value <= MAX_KERNEL_EXTENT {
        Ok(value)
    } else {
        Err(HipError::new(0, &format!("{what} exceeds i32")))
    }
}

fn checked_product(left: usize, right: usize, what: &'static str) -> HipResult<usize> {
    let value = left
        .checked_mul(right)
        .ok_or_else(|| HipError::new(0, &format!("{what} size overflow")))?;
    checked_extent(value, what)
}

fn checked_product3(
    first: usize,
    second: usize,
    third: usize,
    what: &'static str,
) -> HipResult<usize> {
    checked_product(first, second, what).and_then(|value| checked_product(value, third, what))
}

fn checked_add(left: usize, right: usize, what: &'static str) -> HipResult<usize> {
    let value = left
        .checked_add(right)
        .ok_or_else(|| HipError::new(0, &format!("{what} size overflow")))?;
    checked_extent(value, what)
}

fn checked_i32(value: usize, what: &'static str) -> HipResult<i32> {
    checked_extent(value, what).map(|value| value as i32)
}

fn checked_u32(value: usize, what: &'static str) -> HipResult<u32> {
    u32::try_from(value).map_err(|_| HipError::new(0, &format!("{what} exceeds u32")))
}

pub(crate) fn blocks(elements: usize) -> HipResult<u32> {
    let elements = checked_extent(elements, "tensor operation flattened extent")?;
    checked_u32(
        elements.div_ceil(KERNEL_BLOCK),
        "tensor operation block grid",
    )
}

pub struct GatedDeltaStep<'a> {
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

pub fn gated_delta_step(gpu: &mut Gpu, p: &GatedDeltaStep<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.q, p.k, p.v, p.gate, p.beta, p.state, p.output] {
        ensure_f32(tensor)?;
    }
    if p.key_heads == 0
        || p.value_heads == 0
        || p.value_heads % p.key_heads != 0
        || p.key_dim == 0
        || p.value_dim == 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let key_elements = checked_product(p.key_heads, p.key_dim, "GDN key extent")?;
    let value_elements = checked_product(p.value_heads, p.value_dim, "GDN value extent")?;
    let state_elements = checked_product(value_elements, p.key_dim, "GDN state extent")?;
    if p.q.numel() != key_elements
        || p.k.numel() != key_elements
        || p.v.numel() != value_elements
        || p.gate.numel() != p.value_heads
        || p.beta.numel() != p.value_heads
        || p.state.numel() != state_elements
        || p.output.numel() != value_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let key_heads = checked_i32(p.key_heads, "GDN key heads")?;
    let value_heads = checked_i32(p.value_heads, "GDN value heads")?;
    let key_dim = checked_i32(p.key_dim, "GDN key width")?;
    let value_dim = checked_i32(p.value_dim, "GDN value width")?;
    let value_heads_grid = checked_u32(p.value_heads, "GDN value-head grid")?;
    let value_dim_grid = blocks(p.value_dim)?;
    let kernel = if p.key_dim == 128 && p.value_dim == 128 {
        "gated_delta_step_shared_norm128_gfx1151"
    } else {
        "gated_delta_step_f32"
    };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel)?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.q.buf.as_ptr());
    args.push_ptr(p.k.buf.as_ptr());
    args.push_ptr(p.v.buf.as_ptr());
    args.push_ptr(p.gate.buf.as_ptr());
    args.push_ptr(p.beta.buf.as_ptr());
    args.push_ptr(p.state.buf.as_ptr());
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(key_heads);
    args.push_i32(value_heads);
    args.push_i32(key_dim);
    args.push_i32(value_dim);
    args.push_f32((p.key_dim as f32).sqrt().recip());
    args.pad_to(16);
    gpu.launch_kernel_blob(
        kernel,
        [value_heads_grid, value_dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

/// Device-side F32 -> BF16 storage -> F32 conversion at a source activation
/// boundary.  `scratch` is caller-owned and is allocated with the forward
/// arena; no host transfer or per-token allocation occurs.
pub struct Bf16Roundtrip<'a> {
    pub input: &'a GpuTensor,
    pub scratch: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub elements: usize,
}

pub fn bf16_roundtrip_f32(gpu: &mut Gpu, p: &Bf16Roundtrip<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.input)?;
    ensure_f32(p.output)?;
    let elements = checked_extent(p.elements, "BF16 roundtrip extent")?;
    if elements == 0
        || p.scratch.dtype != DType::BF16
        || p.input.numel() < elements
        || p.scratch.numel() < elements
        || p.output.numel() < elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements_i = checked_i32(elements, "BF16 roundtrip extent")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "bf16_roundtrip_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.input.buf.as_ptr());
    args.push_ptr(p.scratch.buf.as_ptr());
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(elements_i);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "bf16_roundtrip_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}
/// Source-exact BF16 product and residual add for the shared expert.
pub struct Bf16ScaledAdd<'a> {
    pub residual: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub scalar: &'a GpuTensor,
    pub elements: usize,
}

pub fn bf16_scaled_add(gpu: &mut Gpu, p: &Bf16ScaledAdd<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.residual)?;
    ensure_f32(p.value)?;
    ensure_f32(p.scalar)?;
    let elements = checked_extent(p.elements, "BF16 scaled-add extent")?;
    if elements == 0
        || p.residual.numel() < elements
        || p.value.numel() < elements
        || p.scalar.numel() == 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements_i = checked_i32(elements, "BF16 scaled-add extent")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "bf16_scaled_add_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.residual.buf.as_ptr());
    args.push_ptr(p.value.buf.as_ptr());
    args.push_ptr(p.scalar.buf.as_ptr());
    args.push_i32(elements_i);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "bf16_scaled_add_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}
/// Source-exact BF16 product/residual add for a row-batched shared expert.
pub struct Bf16ScaledAddBatched<'a> {
    pub residual: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub scalar: &'a GpuTensor,
    pub rows: usize,
    pub elements: usize,
}

pub fn bf16_scaled_add_batched(gpu: &mut Gpu, p: &Bf16ScaledAddBatched<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.residual)?;
    ensure_f32(p.value)?;
    ensure_f32(p.scalar)?;
    let flat_elements = checked_product(p.rows, p.elements, "BF16 batched scaled-add extent")?;
    if p.rows == 0
        || p.elements == 0
        || p.residual.numel() < flat_elements
        || p.value.numel() < flat_elements
        || p.scalar.numel() < p.rows
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = checked_i32(p.rows, "BF16 batched scaled-add rows")?;
    let elements = checked_i32(p.elements, "BF16 batched scaled-add width")?;
    let grid = blocks(flat_elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "bf16_scaled_add_batched_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.residual.buf.as_ptr());
    args.push_ptr(p.value.buf.as_ptr());
    args.push_ptr(p.scalar.buf.as_ptr());
    args.push_i32(rows);
    args.push_i32(elements);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "bf16_scaled_add_batched_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct HyperRead<'a> {
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

pub fn hyper_read(gpu: &mut Gpu, p: &HyperRead<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.input, p.low, p.up, p.normalized, p.mixed] {
        ensure_f32(tensor)?;
    }
    if p.branches == 0 || p.hidden == 0 || p.rank == 0 || p.norm_weight.dtype != DType::BF16 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let input_elements = checked_product(p.branches, p.hidden, "HC read input extent")?;
    let up_elements = checked_product(input_elements, p.rank, "HC read projection extent")?;
    let branches = checked_i32(p.branches, "HC read branch count")?;
    let hidden = checked_i32(p.hidden, "HC read hidden width")?;
    let rank = checked_i32(p.rank, "HC read rank")?;
    if p.input.numel() != input_elements
        || p.norm_weight.numel() != input_elements
        || p.low.numel() != p.rank
        || p.up.numel() != up_elements
        || p.normalized.numel() != input_elements
        || p.mixed.numel() != p.hidden
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_read_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.norm_weight, p.low, p.up, p.normalized, p.mixed] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(branches);
    args.push_i32(hidden);
    args.push_i32(rank);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "hyper_read_f32",
        [1, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

/// HC read where the up projection has already been evaluated to one F32
/// gate logit per branch/hidden column.
pub struct HyperReadProjected<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub up: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
}

pub fn hyper_read_projected(gpu: &mut Gpu, p: &HyperReadProjected<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.input, p.up, p.normalized, p.mixed] {
        ensure_f32(tensor)?;
    }
    if p.norm_weight.dtype != DType::BF16 || p.branches == 0 || p.hidden == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let wide = checked_product(p.branches, p.hidden, "projected HC width")?;
    let input_elements = checked_extent(p.input.numel(), "projected HC input extent")?;
    if input_elements == 0 || input_elements % wide != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = input_elements / wide;
    let mixed_elements = checked_product(rows, p.hidden, "projected HC mixed extent")?;
    let branches = checked_i32(p.branches, "projected HC branch count")?;
    let hidden = checked_i32(p.hidden, "projected HC hidden width")?;
    let rows_i = checked_i32(rows, "projected HC row count")?;
    let grid_x = blocks(wide)?;
    let grid_y = checked_u32(rows, "projected HC row grid")?;
    if p.norm_weight.numel() != wide
        || p.up.numel() != input_elements
        || p.normalized.numel() != input_elements
        || p.mixed.numel() != mixed_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_read_projected_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.norm_weight, p.up, p.normalized, p.mixed] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(branches);
    args.push_i32(hidden);
    args.push_i32(rows_i);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "hyper_read_projected_f32",
        [grid_x, grid_y, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct HyperWrite<'a> {
    pub input: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub mixed: &'a GpuTensor,
    pub gates: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
}

pub fn hyper_write(gpu: &mut Gpu, p: &HyperWrite<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.input, p.normalized, p.mixed, p.gates, p.output] {
        ensure_f32(tensor)?;
    }
    if p.branches == 0 || p.hidden == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let wide = checked_product(p.branches, p.hidden, "HC write width")?;
    let input_elements = checked_extent(p.input.numel(), "HC write input extent")?;
    if input_elements == 0 || input_elements % wide != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = input_elements / wide;
    let mixed_elements = checked_product(rows, p.hidden, "HC write mixed extent")?;
    let gate_elements = checked_product(rows, p.branches, "HC write gate extent")?;
    let branches = checked_i32(p.branches, "HC write branch count")?;
    let hidden = checked_i32(p.hidden, "HC write hidden width")?;
    let rows_i = checked_i32(rows, "HC write row count")?;
    let grid_x = blocks(wide)?;
    let grid_y = checked_u32(rows, "HC write row grid")?;
    if p.normalized.numel() != input_elements
        || p.mixed.numel() != mixed_elements
        || p.gates.numel() != gate_elements
        || p.output.numel() != input_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_write_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.normalized, p.mixed, p.gates, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(branches);
    args.push_i32(hidden);
    args.push_i32(rows_i);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "hyper_write_f32",
        [grid_x, grid_y, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct HyperFinal<'a> {
    pub normalized: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
}

pub fn hyper_final(gpu: &mut Gpu, p: &HyperFinal<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.normalized, p.output] {
        ensure_f32(tensor)?;
    }
    if p.branches == 0 || p.hidden == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let normalized_elements = checked_product(p.branches, p.hidden, "HC final extent")?;
    let branches = checked_i32(p.branches, "HC final branch count")?;
    let hidden = checked_i32(p.hidden, "HC final hidden width")?;
    let grid = blocks(p.hidden)?;
    if p.normalized.numel() != normalized_elements || p.output.numel() != p.hidden {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_final_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.normalized.buf.as_ptr());
    args.push_ptr(p.output.buf.as_ptr());
    args.push_i32(branches);
    args.push_i32(hidden);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "hyper_final_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}
pub struct HyperNorm<'a> {
    pub input: &'a GpuTensor,
    pub norm_weight: &'a GpuTensor,
    pub normalized: &'a GpuTensor,
    pub branches: usize,
    pub hidden: usize,
}

pub fn hyper_norm(gpu: &mut Gpu, p: &HyperNorm<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.input)?;
    ensure_f32(p.normalized)?;
    if p.norm_weight.dtype != DType::BF16 || p.branches == 0 || p.hidden == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let wide = checked_product(p.branches, p.hidden, "HC norm width")?;
    let input_elements = checked_extent(p.input.numel(), "HC norm input extent")?;
    if input_elements == 0 || input_elements % wide != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let rows = input_elements / wide;
    let branches = checked_i32(p.branches, "HC norm branch count")?;
    let hidden = checked_i32(p.hidden, "HC norm hidden width")?;
    let rows_i = checked_i32(rows, "HC norm row count")?;
    let branch_grid = checked_u32(p.branches, "HC norm branch grid")?;
    let row_grid = checked_u32(rows, "HC norm row grid")?;
    if p.norm_weight.numel() != wide || p.normalized.numel() != input_elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "hyper_norm_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.input.buf.as_ptr());
    args.push_ptr(p.norm_weight.buf.as_ptr());
    args.push_ptr(p.normalized.buf.as_ptr());
    args.push_i32(branches);
    args.push_i32(hidden);
    args.push_i32(rows_i);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "hyper_norm_f32",
        [branch_grid, row_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}
pub struct GatedDeltaConv<'a> {
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

pub fn gated_delta_conv(gpu: &mut Gpu, p: &GatedDeltaConv<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.input, p.history, p.output, p.next_history] {
        ensure_f32(tensor)?;
    }
    if p.kernel.dtype != DType::BF16 || p.channels == 0 || p.kernel_size == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let expected_history_rows = p.kernel_size - 1;
    let kernel_elements =
        checked_product(p.channels, p.kernel_size, "GDN convolution kernel extent")?;
    let history_elements = checked_product(
        p.channels,
        expected_history_rows,
        "GDN convolution history extent",
    )?;
    if p.history_rows != expected_history_rows
        || p.cursor >= expected_history_rows.max(1)
        || p.input.numel() != p.channels
        || p.kernel.numel() != kernel_elements
        || p.history.numel() != history_elements
        || p.output.numel() != p.channels
        || p.next_history.numel() != history_elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let channels = checked_i32(p.channels, "GDN convolution channels")?;
    let history_rows = checked_i32(expected_history_rows, "GDN convolution history rows")?;
    let kernel_size = checked_i32(p.kernel_size, "GDN convolution kernel width")?;
    let cursor = checked_i32(p.cursor, "GDN convolution cursor")?;
    let grid = blocks(p.channels)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "gated_delta_conv_bf16_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.input, p.kernel, p.history, p.output, p.next_history] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(channels);
    args.push_i32(history_rows);
    args.push_i32(kernel_size);
    args.push_i32(cursor);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "gated_delta_conv_bf16_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct GatedDeltaParams<'a> {
    pub a: &'a GpuTensor,
    pub b: &'a GpuTensor,
    pub a_log: &'a GpuTensor,
    pub dt_bias: &'a GpuTensor,
    pub gate: &'a GpuTensor,
    pub beta: &'a GpuTensor,
}

pub fn gated_delta_params(gpu: &mut Gpu, p: &GatedDeltaParams<'_>, heads: usize) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.a, p.b, p.gate, p.beta] {
        ensure_f32(tensor)?;
    }
    if heads == 0
        || p.a_log.dtype != DType::BF16
        || p.dt_bias.dtype != DType::BF16
        || p.a.numel() != heads
        || p.b.numel() != heads
        || p.a_log.numel() != heads
        || p.dt_bias.numel() != heads
        || p.gate.numel() != heads
        || p.beta.numel() != heads
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let heads_i = checked_i32(heads, "GDN parameter head count")?;
    let grid = blocks(heads)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "gated_delta_params_bf16_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(heads_i);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "gated_delta_params_bf16_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub fn gated_delta_params_f32(
    gpu: &mut Gpu,
    p: &GatedDeltaParams<'_>,
    heads: usize,
) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        ensure_f32(tensor)?;
    }
    if heads == 0
        || p.a.numel() != heads
        || p.b.numel() != heads
        || p.a_log.numel() != heads
        || p.dt_bias.numel() != heads
        || p.gate.numel() != heads
        || p.beta.numel() != heads
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let heads_i = checked_i32(heads, "GDN parameter head count")?;
    let grid = blocks(heads)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "gated_delta_params_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.a, p.b, p.a_log, p.dt_bias, p.gate, p.beta] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(heads_i);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "gated_delta_params_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct GatedDeltaGate<'a> {
    pub recurrent_output: &'a GpuTensor,
    pub z: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub output: &'a GpuTensor,
    pub value_heads: usize,
    pub value_dim: usize,
}

pub fn gated_delta_gate(gpu: &mut Gpu, p: &GatedDeltaGate<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.recurrent_output, p.z, p.output] {
        ensure_f32(tensor)?;
    }
    if p.norm.dtype != DType::BF16 || p.value_heads == 0 || p.value_dim == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements = checked_product(p.value_heads, p.value_dim, "GDN gate extent")?;
    if p.recurrent_output.numel() != elements
        || p.z.numel() != elements
        || p.norm.numel() != p.value_dim
        || p.output.numel() != elements
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let value_heads = checked_i32(p.value_heads, "GDN gate head count")?;
    let value_dim = checked_i32(p.value_dim, "GDN gate width")?;
    let value_heads_grid = checked_u32(p.value_heads, "GDN gate head grid")?;
    let value_dim_grid = blocks(p.value_dim)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "gated_delta_gate_bf16_f32")?;
    let mut args = KernargBlob::new();
    for tensor in [p.recurrent_output, p.z, p.norm, p.output] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(value_heads);
    args.push_i32(value_dim);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "gated_delta_gate_bf16_f32",
        [value_heads_grid, value_dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}
pub struct IndexedAttentionNormRope<'a> {
    pub values: &'a GpuTensor,
    pub norm: &'a GpuTensor,
    pub heads: usize,
    pub head_dim: usize,
    /// Distance, in elements, between consecutive query/key heads.
    pub head_stride: usize,
    pub position: usize,
    pub rotary_dim: usize,
}

pub fn indexed_attention_norm_rope(
    gpu: &mut Gpu,
    p: &IndexedAttentionNormRope<'_>,
) -> HipResult<()> {
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
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let span = checked_product(p.heads - 1, p.head_stride, "indexed attention head span")
        .and_then(|base| checked_add(base, p.head_dim, "indexed attention head span"))?;
    let heads = checked_i32(p.heads, "indexed attention head count")?;
    let head_dim = checked_i32(p.head_dim, "indexed attention head width")?;
    let head_stride = checked_i32(p.head_stride, "indexed attention head stride")?;
    let position = checked_i32(p.position, "indexed attention position")?;
    let rotary_dim = checked_i32(p.rotary_dim, "indexed attention rotary width")?;
    let head_grid = checked_u32(p.heads, "indexed attention head grid")?;
    let dim_grid = blocks(p.head_dim)?;
    if p.values.numel() < span || p.norm.numel() != p.head_dim {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_norm_rope_f32",
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.values.buf.as_ptr());
    args.push_ptr(p.norm.buf.as_ptr());
    args.push_i32(heads);
    args.push_i32(head_dim);
    args.push_i32(head_stride);
    args.push_i32(position);
    args.push_i32(rotary_dim);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "indexed_attention_norm_rope_f32",
        [head_grid, dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct IndexedAttentionCacheAppend<'a> {
    pub key: &'a GpuTensor,
    pub value: &'a GpuTensor,
    pub full_keys: &'a GpuTensor,
    pub full_values: &'a GpuTensor,
    pub position: usize,
    pub kv_width: usize,
}

pub fn indexed_attention_cache_append(
    gpu: &mut Gpu,
    p: &IndexedAttentionCacheAppend<'_>,
) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.key, p.value, p.full_keys, p.full_values] {
        ensure_f32(tensor)?;
    }
    if p.kv_width == 0
        || p.key.numel() != p.kv_width
        || p.value.numel() != p.kv_width
        || p.full_keys.numel() != p.full_values.numel()
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let end = checked_product(p.position, p.kv_width, "indexed attention cache offset")
        .and_then(|base| checked_add(base, p.kv_width, "indexed attention cache offset"))?;
    if end > p.full_keys.numel() {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let position = checked_i32(p.position, "indexed attention cache position")?;
    let kv_width = checked_i32(p.kv_width, "indexed attention cache width")?;
    let grid = blocks(p.kv_width)?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_cache_append_f32",
    )?;
    let mut args = KernargBlob::new();
    for tensor in [p.key, p.value, p.full_keys, p.full_values] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    args.push_i32(position);
    args.push_i32(kv_width);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "indexed_attention_cache_append_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct IndexedAttentionSelect<'a> {
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

pub fn indexed_attention_select(gpu: &mut Gpu, p: &IndexedAttentionSelect<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.query, p.pooled] {
        ensure_f32(tensor)?;
    }
    if p.selected.dtype != DType::Raw
        || p.index_heads == 0
        || p.index_dim == 0
        || p.compress == 0
        || p.capacity == 0
        || p.visible == 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let query_elements = checked_product(p.index_heads, p.index_dim, "QSA select query extent")?;
    let pooled_elements = checked_product(p.block_count, p.index_dim, "QSA select pooled extent")?;
    checked_product(p.block_count, p.compress, "QSA select token extent")?;
    let selected_bytes = p
        .capacity
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let block_count = checked_i32(p.block_count, "QSA select block count")?;
    let index_heads = checked_i32(p.index_heads, "QSA select index-head count")?;
    let index_dim = checked_i32(p.index_dim, "QSA select index width")?;
    let budget_blocks = checked_i32(p.budget_blocks, "QSA select budget")?;
    let compress = checked_i32(p.compress, "QSA select compression")?;
    let visible = checked_i32(p.visible, "QSA select visible count")?;
    let capacity = checked_i32(p.capacity, "QSA select capacity")?;
    if p.query.numel() != query_elements
        || p.pooled.numel() < pooled_elements
        || p.selected.numel() < selected_bytes
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let (kernel_name, block, shared_mem) =
        match p.block_count.checked_mul(std::mem::size_of::<f32>()) {
            Some(bytes)
                if bytes <= QSA_SELECT_DYNAMIC_LDS_LIMIT_BYTES && bytes <= u32::MAX as usize =>
            {
                (
                    "indexed_attention_select_f32",
                    [QSA_SELECT_PARALLEL_THREADS, 1, 1],
                    bytes as u32,
                )
            }
            _ => ("indexed_attention_select_f32_serial", [1, 1, 1], 0),
        };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel_name)?;
    let mut args = KernargBlob::new();
    for tensor in [p.query, p.pooled, p.selected] {
        args.push_ptr(tensor.buf.as_ptr());
    }
    for value in [
        block_count,
        index_heads,
        index_dim,
        budget_blocks,
        compress,
        visible,
        capacity,
    ] {
        args.push_i32(value);
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
pub struct IndexedAttentionReuseSelection<'a> {
    pub selected: &'a GpuTensor,
    pub selected_len: usize,
    pub position: usize,
    pub capacity: usize,
    pub selected_len_out: &'a GpuTensor,
}

pub fn indexed_attention_reuse_selection(
    gpu: &mut Gpu,
    p: &IndexedAttentionReuseSelection<'_>,
) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    let selected_bytes = p
        .capacity
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    if p.selected.dtype != DType::Raw
        || p.selected_len > p.capacity
        || p.capacity == 0
        || p.selected.numel() < selected_bytes
        || p.selected_len_out.dtype != DType::Raw
        || p.selected_len_out.numel() < std::mem::size_of::<i32>()
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let selected_len = checked_i32(p.selected_len, "QSA reuse selected length")?;
    let position = checked_i32(p.position, "QSA reuse position")?;
    let capacity = checked_i32(p.capacity, "QSA reuse capacity")?;
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_reuse_selection",
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.selected.buf.as_ptr());
    args.push_ptr(p.selected_len_out.buf.as_ptr());
    args.push_i32(selected_len);
    args.push_i32(position);
    args.push_i32(capacity);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "indexed_attention_reuse_selection",
        [1, 1, 1],
        [1, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct IndexedAttentionPoolRope<'a> {
    pub raw_keys: &'a GpuTensor,
    pub pooled: &'a GpuTensor,
    /// Learned index-key RMSNorm. `None` is reserved for synthetic parity
    /// buffers, which intentionally retain the plain-RMS behavior.
    pub norm: Option<&'a GpuTensor>,
    pub block_count: usize,
    pub compress: usize,
    pub index_dim: usize,
}

pub fn indexed_attention_pool_rope(
    gpu: &mut Gpu,
    p: &IndexedAttentionPoolRope<'_>,
) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.raw_keys, p.pooled] {
        ensure_f32(tensor)?;
    }
    if p.block_count == 0 || p.compress == 0 || p.index_dim == 0 || p.index_dim % 2 != 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    if let Some(norm) = p.norm {
        if norm.dtype != DType::BF16 || norm.numel() != p.index_dim {
            return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
        }
    }
    let raw_elements = checked_product3(
        p.block_count,
        p.compress,
        p.index_dim,
        "QSA pool/RoPE raw extent",
    )?;
    let pooled_elements =
        checked_product(p.block_count, p.index_dim, "QSA pool/RoPE pooled extent")?;
    let block_count = checked_i32(p.block_count, "QSA pool/RoPE block count")?;
    let compress = checked_i32(p.compress, "QSA pool/RoPE compression")?;
    let index_dim = checked_i32(p.index_dim, "QSA pool/RoPE index width")?;
    let block_grid = checked_u32(p.block_count, "QSA pool/RoPE block grid")?;
    let dim_grid = blocks(p.index_dim)?;
    if p.raw_keys.numel() < raw_elements || p.pooled.numel() < pooled_elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public(
        "tensor_ops",
        TENSOR_OPS_SRC,
        "indexed_attention_pool_rope_f32",
    )?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.raw_keys.buf.as_ptr());
    args.push_ptr(p.pooled.buf.as_ptr());
    args.push_ptr(
        p.norm
            .map(|norm| norm.buf.as_ptr())
            .unwrap_or(std::ptr::null_mut()),
    );
    args.push_i32(block_count);
    args.push_i32(compress);
    args.push_i32(index_dim);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "indexed_attention_pool_rope_f32",
        [block_grid, dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

pub struct IndexedAttentionAttention<'a> {
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

pub fn indexed_attention_attention(
    gpu: &mut Gpu,
    p: &IndexedAttentionAttention<'_>,
) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    for tensor in [p.q_with_gate, p.full_keys, p.full_values, p.output] {
        ensure_f32(tensor)?;
    }
    if p.selected.dtype != DType::Raw
        || p.n_heads == 0
        || p.n_kv_heads == 0
        || p.head_dim == 0
        || p.n_heads % p.n_kv_heads != 0
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let head_elements = checked_product(p.n_heads, p.head_dim, "QSA attention head extent")?;
    let q_elements = checked_product(2, head_elements, "QSA attention query extent")?;
    let output_elements = head_elements;
    let full_elements = checked_product3(
        p.full_capacity,
        p.n_kv_heads,
        p.head_dim,
        "QSA attention cache extent",
    )?;
    let selected_bytes = p
        .selected_len
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let n_heads = checked_i32(p.n_heads, "QSA attention head count")?;
    let n_kv_heads = checked_i32(p.n_kv_heads, "QSA attention KV-head count")?;
    let head_dim = checked_i32(p.head_dim, "QSA attention head width")?;
    let selected_len = checked_i32(p.selected_len, "QSA attention selected length")?;
    let full_capacity = checked_i32(p.full_capacity, "QSA attention capacity")?;
    let head_grid = checked_u32(p.n_heads, "QSA attention head grid")?;
    let dim_grid = blocks(p.head_dim)?;
    if p.q_with_gate.numel() != q_elements
        || p.output.numel() != output_elements
        || p.full_keys.numel() < full_elements
        || p.full_values.numel() < p.full_keys.numel()
        || p.selected.numel() < selected_bytes
    {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let (kernel_name, shared_mem) =
        match p.selected_len.checked_mul(QSA_ATTENTION_LDS_BYTES_PER_ROW) {
            Some(bytes)
                if bytes <= QSA_ATTENTION_DYNAMIC_LDS_LIMIT_BYTES && bytes <= u32::MAX as usize =>
            {
                ("indexed_attention_attention_f32", bytes as u32)
            }
            _ => ("indexed_attention_attention_f32_serial", 0),
        };
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, kernel_name)?;
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
    for value in [n_heads, n_kv_heads, head_dim, selected_len, full_capacity] {
        args.push_i32(value);
    }
    args.pad_to(16);
    gpu.launch_kernel_blob(
        kernel_name,
        [head_grid, dim_grid, 1],
        [QSA_ATTENTION_PARALLEL_THREADS, 1, 1],
        shared_mem,
        args.as_mut_slice(),
    )
}

pub struct IndexedAttentionPool<'a> {
    pub raw_keys: &'a GpuTensor,
    pub pooled: &'a GpuTensor,
    pub block_count: usize,
    pub head_dim: usize,
}

pub fn indexed_attention_pool(gpu: &mut Gpu, p: &IndexedAttentionPool<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.raw_keys)?;
    ensure_f32(p.pooled)?;
    if p.block_count == 0 || p.head_dim == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let raw_elements = checked_product3(p.block_count, 4, p.head_dim, "QSA pool raw extent")?;
    let pooled_elements = checked_product(p.block_count, p.head_dim, "QSA pool output extent")?;
    let block_count = checked_i32(p.block_count, "QSA pool block count")?;
    let head_dim = checked_i32(p.head_dim, "QSA pool head width")?;
    let block_grid = checked_u32(p.block_count, "QSA pool block grid")?;
    let dim_grid = blocks(p.head_dim)?;
    if p.raw_keys.numel() < raw_elements || p.pooled.numel() < pooled_elements {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "indexed_attention_pool_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.raw_keys.buf.as_ptr());
    args.push_ptr(p.pooled.buf.as_ptr());
    args.push_i32(block_count);
    args.push_i32(head_dim);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "indexed_attention_pool_f32",
        [block_grid, dim_grid, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

/// In-place scalar scaling for learned HC branch projections.
pub struct ScaleF32<'a> {
    pub values: &'a GpuTensor,
    pub scale: f32,
}

pub fn scale_f32(gpu: &mut Gpu, p: &ScaleF32<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.values)?;
    let elements = checked_extent(p.values.numel(), "scale extent")?;
    if elements == 0 {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let elements_i = checked_i32(elements, "scale extent")?;
    let grid = blocks(elements)?;
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "scale_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.values.buf.as_ptr());
    args.push_i32(elements_i);
    args.push_f32(p.scale);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "scale_f32",
        [grid, 1, 1],
        [256, 1, 1],
        0,
        args.as_mut_slice(),
    )
}

/// Device-side greedy top-1 over contiguous F32 logits rows.
pub struct ArgmaxF32<'a> {
    pub logits: &'a GpuTensor,
    pub indices: &'a GpuTensor,
    pub rows: usize,
    pub vocab: usize,
}

pub fn argmax_f32(gpu: &mut Gpu, p: &ArgmaxF32<'_>) -> HipResult<()> {
    ensure_gfx1151(gpu)?;
    ensure_f32(p.logits)?;
    if p.rows == 0 || p.vocab == 0 || p.indices.dtype != DType::Raw {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    let logits_elements = checked_product(p.rows, p.vocab, "argmax logits extent")?;
    let indices_bytes = p
        .rows
        .checked_mul(std::mem::size_of::<i32>())
        .ok_or_else(|| HipError::new(0, &ComputeError::WrongShape.to_string()))?;
    let rows = checked_i32(p.rows, "argmax row count")?;
    let vocab = checked_i32(p.vocab, "argmax vocabulary width")?;
    let row_grid = checked_u32(p.rows, "argmax row grid")?;
    if p.logits.numel() != logits_elements || p.indices.numel() < indices_bytes {
        return Err(HipError::new(0, &ComputeError::WrongShape.to_string()));
    }
    gpu.ensure_kernel_public("tensor_ops", TENSOR_OPS_SRC, "argmax_f32")?;
    let mut args = KernargBlob::new();
    args.push_ptr(p.logits.buf.as_ptr());
    args.push_ptr(p.indices.buf.as_ptr());
    args.push_i32(rows);
    args.push_i32(vocab);
    args.pad_to(16);
    gpu.launch_kernel_blob(
        "argmax_f32",
        [row_grid, 1, 1],
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
            eprintln!("skip: ordinary-HIP tensor-op validation requires gfx1151");
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
    fn checked_extents_reject_signed_flattening_overflow() {
        let max = i32::MAX as usize;
        assert_eq!(checked_product(max, 1, "boundary").unwrap(), max);
        assert!(checked_product(max, 2, "boundary").is_err());
        assert!(checked_add(max, 1, "boundary").is_err());
        assert!(blocks(max).is_ok());
        assert!(blocks(max + 1).is_err());
    }

    #[test]
    fn raw_index_outputs_require_i32_byte_capacity() {
        let Some(mut gpu) = try_gfx1151_gpu() else {
            eprintln!("skip: no gfx1151 GPU");
            return;
        };

        let logits = null_tensor(&[1], DType::F32);
        let indices = null_tensor(&[1], DType::Raw);
        let error = argmax_f32(
            &mut gpu,
            &ArgmaxF32 {
                logits: &logits,
                indices: &indices,
                rows: 1,
                vocab: 1,
            },
        )
        .expect_err("argmax must require four Raw bytes per index");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);

        let query = null_tensor(&[1], DType::F32);
        let pooled = null_tensor(&[1], DType::F32);
        let selected = null_tensor(&[1], DType::Raw);
        let error = indexed_attention_select(
            &mut gpu,
            &IndexedAttentionSelect {
                query: &query,
                pooled: &pooled,
                selected: &selected,
                block_count: 1,
                index_heads: 1,
                index_dim: 1,
                budget_blocks: 1,
                compress: 1,
                visible: 1,
                capacity: 1,
            },
        )
        .expect_err("QSA selection must require four Raw bytes per index");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);

        let q_with_gate = null_tensor(&[2], DType::F32);
        let full_keys = null_tensor(&[1], DType::F32);
        let full_values = null_tensor(&[1], DType::F32);
        let output = null_tensor(&[1], DType::F32);
        let error = indexed_attention_attention(
            &mut gpu,
            &IndexedAttentionAttention {
                q_with_gate: &q_with_gate,
                full_keys: &full_keys,
                full_values: &full_values,
                selected: &selected,
                output: &output,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 1,
                selected_len: 1,
                full_capacity: 1,
            },
        )
        .expect_err("QSA attention must require four Raw bytes per index");
        assert!(error.to_string().contains("tensor shape mismatch"));
        assert_eq!(gpu.last_launched_kernel(), None);
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
        let params = GatedDeltaStep {
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

        let error = gated_delta_step(&mut gpu, &params).expect_err("invalid shape must fail");
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
        let params = IndexedAttentionReuseSelection {
            selected: &selected,
            selected_len: 1,
            position: 0,
            capacity: 1,
            selected_len_out: &selected_len_out,
        };

        let error = indexed_attention_reuse_selection(&mut gpu, &params)
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
            indexed_attention_reuse_selection(
                &mut gpu,
                &IndexedAttentionReuseSelection {
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
        indexed_attention_norm_rope(
            &mut gpu,
            &IndexedAttentionNormRope {
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
