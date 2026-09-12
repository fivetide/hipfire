// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Arch-private MQv2 projection views shared by the Qwen4 trunk and MTP.
//!
//! The packed row geometry and FWHT basis dispatch live here exactly once.  A
//! projection caller supplies logical `(m, k)` dimensions while the view keeps
//! encoded byte capacity for sealed expert-resource validation.

use hipfire_dispatch::families::gemv::WeightRef;
use rdna_compute::{DType, Gpu, GpuTensor};

const QT44_GROUP_BYTES: usize = 136;
const QT53_GROUP_BYTES: usize = 68;

pub(crate) fn row_stride(dtype: DType, k: usize) -> usize {
    match dtype {
        DType::MQ4G256V2 => k.div_ceil(256) * QT44_GROUP_BYTES,
        DType::MQ4G128V2 => k.div_ceil(128) * QT53_GROUP_BYTES,
        _ => k * dtype.size(),
    }
}

pub(crate) fn checked_bytes(rows: usize, stride: usize) -> Option<usize> {
    rows.checked_mul(stride)
}

fn alias_tensor(source: &GpuTensor) -> GpuTensor {
    GpuTensor {
        buf: unsafe { source.buf.alias() },
        shape: source.shape.clone(),
        dtype: source.dtype,
    }
}

fn packed_view(source: &GpuTensor, byte_offset: usize, byte_len: usize, dtype: DType) -> GpuTensor {
    GpuTensor {
        buf: source.buf.byte_view(byte_offset, byte_len),
        shape: vec![byte_len],
        dtype,
    }
}

/// A projection view with logical dimensions separated from encoded bytes.
/// Routed expert views use a packed byte-shaped `GpuTensor` because the sealed
/// live-resource validator intentionally checks encoded capacity, not logical
/// element count.
pub(crate) struct ProjectionView {
    pub(crate) tensor: GpuTensor,
    pub(crate) dtype: DType,
    pub(crate) m: usize,
    pub(crate) k: usize,
    pub(crate) row_stride: usize,
}

impl ProjectionView {
    pub(crate) fn from_source(source: &GpuTensor, m: usize, k: usize) -> Self {
        Self {
            tensor: alias_tensor(source),
            dtype: source.dtype,
            m,
            k,
            row_stride: row_stride(source.dtype, k),
        }
    }

    pub(crate) fn from_packed(
        source: &GpuTensor,
        offset: usize,
        bytes: usize,
        dtype: DType,
        m: usize,
        k: usize,
    ) -> Self {
        Self {
            tensor: packed_view(source, offset, bytes, dtype),
            dtype,
            m,
            k,
            row_stride: row_stride(dtype, k),
        }
    }

    pub(crate) fn dispatch_ref(&self) -> WeightRef<'_> {
        WeightRef {
            buf: &self.tensor,
            dtype: self.dtype,
            m: self.m,
            k: self.k,
            row_stride: self.row_stride,
            rotation: None,
            awq_scale: None,
        }
    }
}

/// Dispatch one logical projection with the same MQv2 basis convention as the
/// ordinary Qwen4 trunk.  Inputs/outputs are F32; BF16 remains unrotated.
pub(crate) fn dispatch_gemv(
    gpu: &mut Gpu,
    weight: &GpuTensor,
    input: &GpuTensor,
    rotation: &GpuTensor,
    output: &GpuTensor,
    m: usize,
    k: usize,
) -> hip_bridge::HipResult<()> {
    match weight.dtype {
        DType::MQ4G256V2 => {
            gpu.rotate_x_mq(input, rotation, k)?;
            gpu.gemv_mq4g256v2(weight, rotation, output, m, k)?;
        }
        DType::MQ4G128V2 => {
            gpu.rotate_x_mq_128_v2(input, rotation, k, 1)?;
            gpu.gemv_mq4g128v2(weight, rotation, output, m, k)?;
        }
        DType::BF16 => gpu.gemv_bf16_xf32(weight, input, output, m, k)?,
        dtype => {
            return Err(hip_bridge::HipError::new(
                0,
                &format!("Qwen4 projection has unsupported resident dtype {dtype:?}"),
            ));
        }
    }
    Ok(())
}
