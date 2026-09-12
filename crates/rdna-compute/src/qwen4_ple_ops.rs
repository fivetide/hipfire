// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 PLE GPU lowering.
//!
//! The CPU owner supplies completed contiguous BF16 rows.  These launchers only
//! perform bounded device-side conversion and equation-matching projection,
//! gating, normalization, and depthwise convolution.  Source I/O and row
//! lookup stay outside the GPU segment, so the launch path remains valid for
//! ordinary HIP and for graph capture after kernels have been warmed.

use std::ffi::c_void;

use hip_bridge::{HipError, HipResult, KernargBlob};

use crate::dispatch::{DType, Gpu, GpuTensor};

/// Embedded source for the Qwen4 PLE kernels.
pub const QWEN4_PLE_KERNEL_SRC: &str = include_str!("../../../kernels/src/qwen4_ple.hip");
const QWEN4_PLE_MODULE: &str = "qwen4_ple";
const BLOCK: u32 = 256;

impl Gpu {
    /// Widen contiguous little-endian BF16 rows into F32 on device.
    ///
    /// `staged` is flattened `[tokens, 16, 160]` BF16 and `output` is the same
    /// flattened logical shape in F32.  No host row I/O occurs here.
    pub fn qwen4_ple_gather_convert_bf16(
        &mut self,
        staged: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
    ) -> HipResult<()> {
        let rows = checked_product(tokens, 16, "PLE token rows")?;
        let elements = checked_product(rows, 160, "PLE row width")?;
        require_dtype(staged, DType::BF16, "PLE BF16 staging")?;
        require_dtype(output, DType::F32, "PLE F32 embedding output")?;
        require_numel(staged, elements, "PLE BF16 staging")?;
        require_numel(output, elements, "PLE F32 embedding output")?;
        if elements == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "PLE token count")?;
        let rows_i = checked_i32(16, "PLE rows per token")?;
        let width_i = checked_i32(160, "PLE row width")?;
        self.bind_thread()?;
        self.ensure_kernel(
            QWEN4_PLE_MODULE,
            QWEN4_PLE_KERNEL_SRC,
            "qwen4_ple_gather_convert_bf16",
        )?;
        let staged_ptr = staged.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &staged_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &rows_i as *const _ as *mut c_void,
            &width_i as *const _ as *mut c_void,
        ];
        let grid = checked_grid(elements, BLOCK, "PLE gather grid")?;
        self.launch_maybe_blob(
            "qwen4_ple_gather_convert_bf16",
            [grid, 1, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(staged_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(rows_i);
                blob.push_i32(width_i);
                blob
            },
        )
    }

    /// Row-major F32 projection `output = input · weight^T`.
    pub fn qwen4_ple_linear_f32(
        &mut self,
        input: &GpuTensor,
        weight: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> HipResult<()> {
        let input_elements = checked_product(tokens, in_dim, "PLE linear input")?;
        let weight_elements = checked_product(out_dim, in_dim, "PLE linear weight")?;
        let output_elements = checked_product(tokens, out_dim, "PLE linear output")?;
        require_dtype(input, DType::F32, "PLE linear input")?;
        require_dtype(weight, DType::F32, "PLE linear weight")?;
        require_dtype(output, DType::F32, "PLE linear output")?;
        require_numel(input, input_elements, "PLE linear input")?;
        require_numel(weight, weight_elements, "PLE linear weight")?;
        require_numel(output, output_elements, "PLE linear output")?;
        if output_elements == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "PLE linear token count")?;
        let in_dim_i = checked_i32(in_dim, "PLE linear input width")?;
        let out_dim_i = checked_i32(out_dim, "PLE linear output width")?;
        self.bind_thread()?;
        self.ensure_kernel(
            QWEN4_PLE_MODULE,
            QWEN4_PLE_KERNEL_SRC,
            "qwen4_ple_linear_f32",
        )?;
        let input_ptr = input.buf.as_ptr();
        let weight_ptr = weight.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &input_ptr as *const _ as *mut c_void,
            &weight_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &in_dim_i as *const _ as *mut c_void,
            &out_dim_i as *const _ as *mut c_void,
        ];
        let grid_x = checked_grid(out_dim, BLOCK, "PLE linear output grid")?;
        let grid_y = checked_u32(tokens, "PLE linear token grid")?;
        self.launch_maybe_blob(
            "qwen4_ple_linear_f32",
            [grid_x, grid_y, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(input_ptr);
                blob.push_ptr(weight_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(in_dim_i);
                blob.push_i32(out_dim_i);
                blob
            },
        )
    }

    /// Project/gate the shared value across all PLE residual branches.
    ///
    /// `key` and `query` are `[tokens, hc_count * hidden_size]`, `value` is
    /// `[tokens, hidden_size]`, and each normalization weight is
    /// `[hc_count * hidden_size]`.  The kernel applies grouped RMS with
    /// `(1 + weight)`, signed-square-root dot gating, and sigmoid activation.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen4_ple_gate_f32(
        &mut self,
        key: &GpuTensor,
        query: &GpuTensor,
        value: &GpuTensor,
        key_norm: &GpuTensor,
        query_norm: &GpuTensor,
        gated: &GpuTensor,
        tokens: usize,
        hc_count: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> HipResult<()> {
        let channels = checked_product(hc_count, hidden_size, "PLE gated channels")?;
        require_dtype(key, DType::F32, "PLE key projection")?;
        require_dtype(query, DType::F32, "PLE query projection")?;
        require_dtype(value, DType::F32, "PLE value projection")?;
        require_dtype(key_norm, DType::F32, "PLE key norm")?;
        require_dtype(query_norm, DType::F32, "PLE query norm")?;
        require_dtype(gated, DType::F32, "PLE gated output")?;
        require_numel(
            key,
            checked_product(tokens, channels, "PLE key elements")?,
            "PLE key projection",
        )?;
        require_numel(
            query,
            checked_product(tokens, channels, "PLE query elements")?,
            "PLE query projection",
        )?;
        require_numel(
            value,
            checked_product(tokens, hidden_size, "PLE value elements")?,
            "PLE value projection",
        )?;
        require_numel(key_norm, channels, "PLE key norm")?;
        require_numel(query_norm, channels, "PLE query norm")?;
        require_numel(
            gated,
            checked_product(tokens, channels, "PLE gated elements")?,
            "PLE gated output",
        )?;
        if tokens == 0 || channels == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "PLE gate token count")?;
        let hc_i = checked_i32(hc_count, "PLE branch count")?;
        let hidden_i = checked_i32(hidden_size, "PLE hidden width")?;
        self.bind_thread()?;
        self.ensure_kernel(QWEN4_PLE_MODULE, QWEN4_PLE_KERNEL_SRC, "qwen4_ple_gate_f32")?;
        let key_ptr = key.buf.as_ptr();
        let query_ptr = query.buf.as_ptr();
        let value_ptr = value.buf.as_ptr();
        let key_norm_ptr = key_norm.buf.as_ptr();
        let query_norm_ptr = query_norm.buf.as_ptr();
        let gated_ptr = gated.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &key_ptr as *const _ as *mut c_void,
            &query_ptr as *const _ as *mut c_void,
            &value_ptr as *const _ as *mut c_void,
            &key_norm_ptr as *const _ as *mut c_void,
            &query_norm_ptr as *const _ as *mut c_void,
            &gated_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &hc_i as *const _ as *mut c_void,
            &hidden_i as *const _ as *mut c_void,
            &epsilon as *const _ as *mut c_void,
        ];
        let grid_x = checked_u32(hc_count, "PLE gate branch grid")?;
        let grid_y = checked_u32(tokens, "PLE gate token grid")?;
        self.launch_maybe_blob(
            "qwen4_ple_gate_f32",
            [grid_x, grid_y, 1],
            [1, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(key_ptr);
                blob.push_ptr(query_ptr);
                blob.push_ptr(value_ptr);
                blob.push_ptr(key_norm_ptr);
                blob.push_ptr(query_norm_ptr);
                blob.push_ptr(gated_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(hc_i);
                blob.push_i32(hidden_i);
                blob.push_f32(epsilon);
                blob
            },
        )
    }
    /// Variant of [`Self::qwen4_ple_gate_f32`] for resident BF16 norm
    /// weights.  Qwen4 keeps these norms native BF16; widening happens in
    /// the kernel and never through a host-side conversion.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen4_ple_gate_bf16(
        &mut self,
        key: &GpuTensor,
        query: &GpuTensor,
        value: &GpuTensor,
        key_norm: &GpuTensor,
        query_norm: &GpuTensor,
        gated: &GpuTensor,
        tokens: usize,
        hc_count: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> HipResult<()> {
        let channels = checked_product(hc_count, hidden_size, "PLE gated channels")?;
        require_dtype(key, DType::F32, "PLE key projection")?;
        require_dtype(query, DType::F32, "PLE query projection")?;
        require_dtype(value, DType::F32, "PLE value projection")?;
        require_dtype(key_norm, DType::BF16, "PLE BF16 key norm")?;
        require_dtype(query_norm, DType::BF16, "PLE BF16 query norm")?;
        require_dtype(gated, DType::F32, "PLE gated output")?;
        require_numel(key, checked_product(tokens, channels, "PLE key elements")?, "PLE key projection")?;
        require_numel(query, checked_product(tokens, channels, "PLE query elements")?, "PLE query projection")?;
        require_numel(value, checked_product(tokens, hidden_size, "PLE value elements")?, "PLE value projection")?;
        require_numel(key_norm, channels, "PLE BF16 key norm")?;
        require_numel(query_norm, channels, "PLE BF16 query norm")?;
        require_numel(gated, checked_product(tokens, channels, "PLE gated elements")?, "PLE gated output")?;
        if tokens == 0 || channels == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "PLE gate token count")?;
        let hc_i = checked_i32(hc_count, "PLE branch count")?;
        let hidden_i = checked_i32(hidden_size, "PLE hidden width")?;
        self.bind_thread()?;
        self.ensure_kernel(QWEN4_PLE_MODULE, QWEN4_PLE_KERNEL_SRC, "qwen4_ple_gate_bf16_f32")?;
        let key_ptr = key.buf.as_ptr();
        let query_ptr = query.buf.as_ptr();
        let value_ptr = value.buf.as_ptr();
        let key_norm_ptr = key_norm.buf.as_ptr();
        let query_norm_ptr = query_norm.buf.as_ptr();
        let gated_ptr = gated.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &key_ptr as *const _ as *mut c_void,
            &query_ptr as *const _ as *mut c_void,
            &value_ptr as *const _ as *mut c_void,
            &key_norm_ptr as *const _ as *mut c_void,
            &query_norm_ptr as *const _ as *mut c_void,
            &gated_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &hc_i as *const _ as *mut c_void,
            &hidden_i as *const _ as *mut c_void,
            &epsilon as *const _ as *mut c_void,
        ];
        let grid_x = checked_u32(hc_count, "PLE gate branch grid")?;
        let grid_y = checked_u32(tokens, "PLE gate token grid")?;
        self.launch_maybe_blob(
            "qwen4_ple_gate_bf16_f32",
            [grid_x, grid_y, 1],
            [1, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(key_ptr);
                blob.push_ptr(query_ptr);
                blob.push_ptr(value_ptr);
                blob.push_ptr(key_norm_ptr);
                blob.push_ptr(query_norm_ptr);
                blob.push_ptr(gated_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(hc_i);
                blob.push_i32(hidden_i);
                blob.push_f32(epsilon);
                blob
            },
        )
    }


    /// Apply grouped `(1 + weight)` RMS normalization to F32 rows.
    pub fn qwen4_ple_norm_f32(
        &mut self,
        input: &GpuTensor,
        weight: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        groups: usize,
        group_size: usize,
        epsilon: f32,
    ) -> HipResult<()> {
        let channels = checked_product(groups, group_size, "PLE normalization channels")?;
        let elements = checked_product(tokens, channels, "PLE normalization elements")?;
        require_dtype(input, DType::F32, "PLE normalization input")?;
        require_dtype(weight, DType::F32, "PLE normalization weight")?;
        require_dtype(output, DType::F32, "PLE normalization output")?;
        require_numel(input, elements, "PLE normalization input")?;
        require_numel(weight, channels, "PLE normalization weight")?;
        require_numel(output, elements, "PLE normalization output")?;
        if elements == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "PLE normalization token count")?;
        let groups_i = checked_i32(groups, "PLE normalization group count")?;
        let group_size_i = checked_i32(group_size, "PLE normalization group width")?;
        self.bind_thread()?;
        self.ensure_kernel(QWEN4_PLE_MODULE, QWEN4_PLE_KERNEL_SRC, "qwen4_ple_norm_f32")?;
        let input_ptr = input.buf.as_ptr();
        let weight_ptr = weight.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &input_ptr as *const _ as *mut c_void,
            &weight_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &groups_i as *const _ as *mut c_void,
            &group_size_i as *const _ as *mut c_void,
            &epsilon as *const _ as *mut c_void,
        ];
        let grid_x = checked_u32(groups, "PLE normalization group grid")?;
        let grid_y = checked_u32(tokens, "PLE normalization token grid")?;
        self.launch_maybe_blob(
            "qwen4_ple_norm_f32",
            [grid_x, grid_y, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(input_ptr);
                blob.push_ptr(weight_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(groups_i);
                blob.push_i32(group_size_i);
                blob.push_f32(epsilon);
                blob
            },
        )
    }
    /// Grouped RMS normalization against a resident BF16 norm vector.
    pub fn qwen4_ple_norm_bf16(
        &mut self,
        input: &GpuTensor,
        weight: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        groups: usize,
        group_size: usize,
        epsilon: f32,
    ) -> HipResult<()> {
        let channels = checked_product(groups, group_size, "PLE normalization channels")?;
        let elements = checked_product(tokens, channels, "PLE normalization elements")?;
        require_dtype(input, DType::F32, "PLE normalization input")?;
        require_dtype(weight, DType::BF16, "PLE BF16 normalization weight")?;
        require_dtype(output, DType::F32, "PLE normalization output")?;
        require_numel(input, elements, "PLE normalization input")?;
        require_numel(weight, channels, "PLE BF16 normalization weight")?;
        require_numel(output, elements, "PLE normalization output")?;
        if elements == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "PLE normalization token count")?;
        let groups_i = checked_i32(groups, "PLE normalization group count")?;
        let group_size_i = checked_i32(group_size, "PLE normalization group width")?;
        self.bind_thread()?;
        self.ensure_kernel(QWEN4_PLE_MODULE, QWEN4_PLE_KERNEL_SRC, "qwen4_ple_norm_bf16")?;
        let input_ptr = input.buf.as_ptr();
        let weight_ptr = weight.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &input_ptr as *const _ as *mut c_void,
            &weight_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &groups_i as *const _ as *mut c_void,
            &group_size_i as *const _ as *mut c_void,
            &epsilon as *const _ as *mut c_void,
        ];
        let grid_x = checked_u32(groups, "PLE normalization group grid")?;
        let grid_y = checked_u32(tokens, "PLE normalization token grid")?;
        self.launch_maybe_blob(
            "qwen4_ple_norm_bf16",
            [grid_x, grid_y, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(input_ptr);
                blob.push_ptr(weight_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(groups_i);
                blob.push_i32(group_size_i);
                blob.push_f32(epsilon);
                blob
            },
        )
    }


    /// Depthwise causal convolution with SiLU and gated-value residual add.
    ///
    /// `state` is `[history_rows, channels]`, where
    /// `history_rows=(kernel_size-1)*dilation`; the kernel updates it in place
    /// after producing all token outputs.  This state is separate from GDN.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen4_ple_depthwise_conv_silu_add_f32(
        &mut self,
        gated: &GpuTensor,
        normed: &GpuTensor,
        conv_weight: &GpuTensor,
        state: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> HipResult<()> {
        let elements = checked_product(tokens, channels, "PLE convolution elements")?;
        let history_rows = kernel_size
            .checked_sub(1)
            .and_then(|value| value.checked_mul(dilation))
            .ok_or_else(|| HipError::new(0, "PLE convolution history size overflow"))?;
        let history_elements = checked_product(history_rows, channels, "PLE convolution history")?;
        let kernel_elements = checked_product(channels, kernel_size, "PLE convolution weights")?;
        require_dtype(gated, DType::F32, "PLE gated values")?;
        require_dtype(normed, DType::F32, "PLE normalized values")?;
        require_dtype(conv_weight, DType::F32, "PLE convolution weights")?;
        require_dtype(state, DType::F32, "PLE convolution state")?;
        require_dtype(output, DType::F32, "PLE convolution output")?;
        require_numel(gated, elements, "PLE gated values")?;
        require_numel(normed, elements, "PLE normalized values")?;
        require_numel(conv_weight, kernel_elements, "PLE convolution weights")?;
        require_numel(state, history_elements, "PLE convolution state")?;
        require_numel(output, elements, "PLE convolution output")?;
        if tokens == 0 || channels == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "PLE convolution token count")?;
        let channels_i = checked_i32(channels, "PLE convolution channel count")?;
        let kernel_i = checked_i32(kernel_size, "PLE convolution kernel size")?;
        let dilation_i = checked_i32(dilation, "PLE convolution dilation")?;
        self.bind_thread()?;
        self.ensure_kernel(
            QWEN4_PLE_MODULE,
            QWEN4_PLE_KERNEL_SRC,
            "qwen4_ple_depthwise_conv_silu_add_f32",
        )?;
        let gated_ptr = gated.buf.as_ptr();
        let normed_ptr = normed.buf.as_ptr();
        let weight_ptr = conv_weight.buf.as_ptr();
        let state_ptr = state.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &gated_ptr as *const _ as *mut c_void,
            &normed_ptr as *const _ as *mut c_void,
            &weight_ptr as *const _ as *mut c_void,
            &state_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &channels_i as *const _ as *mut c_void,
            &kernel_i as *const _ as *mut c_void,
            &dilation_i as *const _ as *mut c_void,
        ];
        let grid = checked_grid(channels, BLOCK, "PLE convolution channel grid")?;
        self.launch_maybe_blob(
            "qwen4_ple_depthwise_conv_silu_add_f32",
            [grid, 1, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(gated_ptr);
                blob.push_ptr(normed_ptr);
                blob.push_ptr(weight_ptr);
                blob.push_ptr(state_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(channels_i);
                blob.push_i32(kernel_i);
                blob.push_i32(dilation_i);
                blob
            },
        )
    }
    /// BF16 convolution-weight variant used by the assembled Qwen4 PLE layer.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen4_ple_depthwise_conv_silu_add_bf16(
        &mut self,
        gated: &GpuTensor,
        normed: &GpuTensor,
        conv_weight: &GpuTensor,
        state: &GpuTensor,
        output: &GpuTensor,
        tokens: usize,
        channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> HipResult<()> {
        let elements = checked_product(tokens, channels, "PLE convolution elements")?;
        let history_rows = kernel_size
            .checked_sub(1)
            .and_then(|value| value.checked_mul(dilation))
            .ok_or_else(|| HipError::new(0, "PLE convolution history size overflow"))?;
        let history_elements = checked_product(history_rows, channels, "PLE convolution history")?;
        let kernel_elements = checked_product(channels, kernel_size, "PLE convolution weights")?;
        require_dtype(gated, DType::F32, "PLE gated values")?;
        require_dtype(normed, DType::F32, "PLE normalized values")?;
        require_dtype(conv_weight, DType::BF16, "PLE BF16 convolution weights")?;
        require_dtype(state, DType::F32, "PLE convolution state")?;
        require_dtype(output, DType::F32, "PLE convolution output")?;
        require_numel(gated, elements, "PLE gated values")?;
        require_numel(normed, elements, "PLE normalized values")?;
        require_numel(conv_weight, kernel_elements, "PLE BF16 convolution weights")?;
        require_numel(state, history_elements, "PLE convolution state")?;
        require_numel(output, elements, "PLE convolution output")?;
        if tokens == 0 || channels == 0 {
            return Ok(());
        }
        let tokens_i = checked_i32(tokens, "PLE convolution token count")?;
        let channels_i = checked_i32(channels, "PLE convolution channel count")?;
        let kernel_i = checked_i32(kernel_size, "PLE convolution kernel size")?;
        let dilation_i = checked_i32(dilation, "PLE convolution dilation")?;
        self.bind_thread()?;
        self.ensure_kernel(
            QWEN4_PLE_MODULE,
            QWEN4_PLE_KERNEL_SRC,
            "qwen4_ple_depthwise_conv_silu_add_bf16",
        )?;
        let gated_ptr = gated.buf.as_ptr();
        let normed_ptr = normed.buf.as_ptr();
        let weight_ptr = conv_weight.buf.as_ptr();
        let state_ptr = state.buf.as_ptr();
        let output_ptr = output.buf.as_ptr();
        let mut params: Vec<*mut c_void> = vec![
            &gated_ptr as *const _ as *mut c_void,
            &normed_ptr as *const _ as *mut c_void,
            &weight_ptr as *const _ as *mut c_void,
            &state_ptr as *const _ as *mut c_void,
            &output_ptr as *const _ as *mut c_void,
            &tokens_i as *const _ as *mut c_void,
            &channels_i as *const _ as *mut c_void,
            &kernel_i as *const _ as *mut c_void,
            &dilation_i as *const _ as *mut c_void,
        ];
        let grid = checked_grid(channels, BLOCK, "PLE convolution channel grid")?;
        self.launch_maybe_blob(
            "qwen4_ple_depthwise_conv_silu_add_bf16",
            [grid, 1, 1],
            [BLOCK, 1, 1],
            0,
            &mut params,
            || {
                let mut blob = KernargBlob::new();
                blob.push_ptr(gated_ptr);
                blob.push_ptr(normed_ptr);
                blob.push_ptr(weight_ptr);
                blob.push_ptr(state_ptr);
                blob.push_ptr(output_ptr);
                blob.push_i32(tokens_i);
                blob.push_i32(channels_i);
                blob.push_i32(kernel_i);
                blob.push_i32(dilation_i);
                blob
            },
        )
    }

}

fn require_dtype(tensor: &GpuTensor, expected: DType, what: &'static str) -> HipResult<()> {
    if tensor.dtype == expected {
        Ok(())
    } else {
        Err(HipError::new(
            0,
            &format!(
                "{what} must have dtype {expected:?}, got {:?}",
                tensor.dtype
            ),
        ))
    }
}

fn require_numel(tensor: &GpuTensor, expected: usize, what: &'static str) -> HipResult<()> {
    if tensor.numel() == expected {
        Ok(())
    } else {
        Err(HipError::new(
            0,
            &format!(
                "{what} must have {expected} elements, got {}",
                tensor.numel()
            ),
        ))
    }
}

fn checked_product(left: usize, right: usize, what: &'static str) -> HipResult<usize> {
    left.checked_mul(right)
        .ok_or_else(|| HipError::new(0, &format!("{what} size overflow")))
}

fn checked_i32(value: usize, what: &'static str) -> HipResult<i32> {
    i32::try_from(value).map_err(|_| HipError::new(0, &format!("{what} exceeds i32")))
}

fn checked_u32(value: usize, what: &'static str) -> HipResult<u32> {
    u32::try_from(value).map_err(|_| HipError::new(0, &format!("{what} exceeds u32")))
}

fn checked_grid(elements: usize, block: u32, what: &'static str) -> HipResult<u32> {
    let elements = checked_u32(elements, what)?;
    Ok(elements.div_ceil(block))
}
