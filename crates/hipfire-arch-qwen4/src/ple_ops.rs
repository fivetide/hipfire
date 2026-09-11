// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! CPU-testable Qwen4 PLE projection, gating, and short-convolution equations.
//!
//! The row owner resolves and stages `[tokens, 16, 160]` BF16 rows.  This
//! module consumes the already converted embedding rows and residual branches;
//! it performs no source I/O, GPU allocation, scheduling, or fallback.  The
//! same storage ordering is used by the ordinary HIP lowering:
//! `[token, branch, hidden]`, with the shared projected value repeated across
//! all four residual branches.

use std::fmt;

/// Qwen4's widened residual branch count.
pub const PLE_HC_COUNT: usize = 4;
/// Qwen4's native hidden width.
pub const PLE_HIDDEN_SIZE: usize = 2560;
/// Flattened n-gram embedding width (`16 * 160`).
pub const PLE_EMBED_DIM: usize = 2560;
/// Native depthwise short-convolution kernel size.
pub const PLE_CONV_KERNEL_SIZE: usize = 4;
/// Native n-gram dilation.
pub const PLE_CONV_DILATION: usize = 3;
/// Native short-convolution history rows.
pub const PLE_CONV_HISTORY_ROWS: usize = (PLE_CONV_KERNEL_SIZE - 1) * PLE_CONV_DILATION;
/// RMS epsilon in the pinned Qwen4 checkpoint.
pub const PLE_RMS_EPS: f32 = 1.0e-6;

/// Shape or arithmetic failure in a PLE operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PleOpError {
    Length {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    Shape(&'static str),
    InvalidConfig(&'static str),
    DimensionOverflow,
}

impl fmt::Display for PleOpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length {
                what,
                expected,
                actual,
            } => write!(f, "{what}: expected {expected} elements, got {actual}"),
            Self::Shape(what) => write!(f, "invalid Qwen4 PLE shape: {what}"),
            Self::InvalidConfig(what) => write!(f, "invalid Qwen4 PLE configuration: {what}"),
            Self::DimensionOverflow => f.write_str("Qwen4 PLE dimension arithmetic overflowed"),
        }
    }
}

impl std::error::Error for PleOpError {}

/// Projection and normalization weights for one PLE layer.
///
/// All matrices are row-major and use the usual inference orientation:
/// `output = input · weight^T`.  Thus `key_proj` is `[hc*hidden, embed]`,
/// `value_proj` is `[hidden, embed]`, and `conv_kernel` is
/// `[hc*hidden, kernel]`.
#[derive(Debug, Clone, Copy)]
pub struct PleProjectionWeights<'a> {
    pub key_proj: &'a [f32],
    pub value_proj: &'a [f32],
    pub norm_key: &'a [f32],
    pub norm_query: &'a [f32],
    pub norm_conv: &'a [f32],
    pub conv_kernel: &'a [f32],
}

impl<'a> PleProjectionWeights<'a> {
    pub const fn new(
        key_proj: &'a [f32],
        value_proj: &'a [f32],
        norm_key: &'a [f32],
        norm_query: &'a [f32],
        norm_conv: &'a [f32],
        conv_kernel: &'a [f32],
    ) -> Self {
        Self {
            key_proj,
            value_proj,
            norm_key,
            norm_query,
            norm_conv,
            conv_kernel,
        }
    }

    pub fn validate(
        &self,
        hc_count: usize,
        hidden_size: usize,
        embed_dim: usize,
        kernel_size: usize,
    ) -> Result<(), PleOpError> {
        let channels = hc_count
            .checked_mul(hidden_size)
            .ok_or(PleOpError::DimensionOverflow)?;
        let key_len = channels
            .checked_mul(embed_dim)
            .ok_or(PleOpError::DimensionOverflow)?;
        let value_len = hidden_size
            .checked_mul(embed_dim)
            .ok_or(PleOpError::DimensionOverflow)?;
        let conv_len = channels
            .checked_mul(kernel_size)
            .ok_or(PleOpError::DimensionOverflow)?;
        check_len("PLE key projection", self.key_proj.len(), key_len)?;
        check_len("PLE value projection", self.value_proj.len(), value_len)?;
        check_len("PLE key norm", self.norm_key.len(), channels)?;
        check_len("PLE query norm", self.norm_query.len(), channels)?;
        check_len("PLE convolution norm", self.norm_conv.len(), channels)?;
        check_len(
            "PLE depthwise convolution",
            self.conv_kernel.len(),
            conv_len,
        )?;
        Ok(())
    }
}

/// Mutable PLE short-convolution state and reusable scratch space.
///
/// `history` stores the oldest-to-newest `(kernel_size - 1) * dilation`
/// normalized rows.  It is intentionally separate from GDN convolution state:
/// PLE history is consumed by the n-gram branch only and is included in native
/// MTP snapshots by the architecture owner.
pub struct PleLayerState {
    hc_count: usize,
    hidden_size: usize,
    embed_dim: usize,
    kernel_size: usize,
    dilation: usize,
    history: Vec<f32>,
    key_raw: Vec<f32>,
    key_normed: Vec<f32>,
    query_normed: Vec<f32>,
    value: Vec<f32>,
    gated_value: Vec<f32>,
    gated_normed: Vec<f32>,
}

impl fmt::Debug for PleLayerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PleLayerState")
            .field("hc_count", &self.hc_count)
            .field("hidden_size", &self.hidden_size)
            .field("embed_dim", &self.embed_dim)
            .field("kernel_size", &self.kernel_size)
            .field("dilation", &self.dilation)
            .field("history_rows", &self.history_rows())
            .finish()
    }
}

impl PleLayerState {
    pub fn new(
        hc_count: usize,
        hidden_size: usize,
        embed_dim: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> Result<Self, PleOpError> {
        if hc_count == 0 || hidden_size == 0 || embed_dim == 0 {
            return Err(PleOpError::InvalidConfig(
                "branch, hidden, and embedding widths must be nonzero",
            ));
        }
        if kernel_size == 0 || dilation == 0 {
            return Err(PleOpError::InvalidConfig(
                "convolution kernel and dilation must be nonzero",
            ));
        }
        let channels = hc_count
            .checked_mul(hidden_size)
            .ok_or(PleOpError::DimensionOverflow)?;
        let history_rows = kernel_size
            .checked_sub(1)
            .and_then(|v| v.checked_mul(dilation))
            .ok_or(PleOpError::DimensionOverflow)?;
        let history_len = history_rows
            .checked_mul(channels)
            .ok_or(PleOpError::DimensionOverflow)?;
        Ok(Self {
            hc_count,
            hidden_size,
            embed_dim,
            kernel_size,
            dilation,
            history: vec![0.0; history_len],
            key_raw: vec![0.0; channels],
            key_normed: vec![0.0; channels],
            query_normed: vec![0.0; channels],
            value: vec![0.0; hidden_size],
            gated_value: vec![0.0; channels],
            gated_normed: vec![0.0; channels],
        })
    }

    pub fn qwen4() -> Self {
        Self::new(
            PLE_HC_COUNT,
            PLE_HIDDEN_SIZE,
            PLE_EMBED_DIM,
            PLE_CONV_KERNEL_SIZE,
            PLE_CONV_DILATION,
        )
        .expect("Qwen4 PLE constants are valid")
    }

    pub const fn hc_count(&self) -> usize {
        self.hc_count
    }

    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub const fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    pub const fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    pub const fn dilation(&self) -> usize {
        self.dilation
    }

    pub const fn channels(&self) -> usize {
        self.hc_count * self.hidden_size
    }

    pub const fn history_rows(&self) -> usize {
        (self.kernel_size - 1) * self.dilation
    }

    pub fn history(&self) -> &[f32] {
        &self.history
    }

    pub fn history_mut(&mut self) -> &mut [f32] {
        &mut self.history
    }

    pub fn reset(&mut self) {
        self.history.fill(0.0);
        self.key_raw.fill(0.0);
        self.key_normed.fill(0.0);
        self.query_normed.fill(0.0);
        self.value.fill(0.0);
        self.gated_value.fill(0.0);
        self.gated_normed.fill(0.0);
    }

    /// Apply one token and advance the nine-row Qwen4 PLE history.
    pub fn apply_token(
        &mut self,
        embedding: &[f32],
        hidden_states: &[f32],
        weights: &PleProjectionWeights<'_>,
        output: &mut [f32],
    ) -> Result<(), PleOpError> {
        self.apply_token_masked(embedding, hidden_states, weights, output, true)
    }

    /// Apply a sequence in causal order, carrying history across every row.
    pub fn apply_sequence(
        &mut self,
        embeddings: &[f32],
        hidden_states: &[f32],
        token_count: usize,
        weights: &PleProjectionWeights<'_>,
        output: &mut [f32],
    ) -> Result<(), PleOpError> {
        self.validate_sequence_lengths(embeddings, hidden_states, token_count, output)?;
        weights.validate(
            self.hc_count,
            self.hidden_size,
            self.embed_dim,
            self.kernel_size,
        )?;
        for token in 0..token_count {
            let embedding_begin = token * self.embed_dim;
            let hidden_begin = token * self.channels();
            let output_begin = token * self.channels();
            self.apply_token_masked(
                &embeddings[embedding_begin..embedding_begin + self.embed_dim],
                &hidden_states[hidden_begin..hidden_begin + self.channels()],
                weights,
                &mut output[output_begin..output_begin + self.channels()],
                true,
            )?;
        }
        Ok(())
    }

    /// Apply a sequence with an explicit active-row mask.
    ///
    /// Inactive rows still advance the causal convolution with a zero current
    /// row, matching a padded causal stream without allowing masked values to
    /// enter the projection/gate path.
    pub fn apply_sequence_masked(
        &mut self,
        embeddings: &[f32],
        hidden_states: &[f32],
        active: &[bool],
        weights: &PleProjectionWeights<'_>,
        output: &mut [f32],
    ) -> Result<(), PleOpError> {
        let token_count = active.len();
        self.validate_sequence_lengths(embeddings, hidden_states, token_count, output)?;
        weights.validate(
            self.hc_count,
            self.hidden_size,
            self.embed_dim,
            self.kernel_size,
        )?;
        for (token, &is_active) in active.iter().enumerate() {
            let embedding_begin = token * self.embed_dim;
            let hidden_begin = token * self.channels();
            let output_begin = token * self.channels();
            self.apply_token_masked(
                &embeddings[embedding_begin..embedding_begin + self.embed_dim],
                &hidden_states[hidden_begin..hidden_begin + self.channels()],
                weights,
                &mut output[output_begin..output_begin + self.channels()],
                is_active,
            )?;
        }
        Ok(())
    }

    fn validate_sequence_lengths(
        &self,
        embeddings: &[f32],
        hidden_states: &[f32],
        token_count: usize,
        output: &[f32],
    ) -> Result<(), PleOpError> {
        let expected_embedding = token_count
            .checked_mul(self.embed_dim)
            .ok_or(PleOpError::DimensionOverflow)?;
        let expected_hidden = token_count
            .checked_mul(self.channels())
            .ok_or(PleOpError::DimensionOverflow)?;
        check_len(
            "PLE sequence embeddings",
            embeddings.len(),
            expected_embedding,
        )?;
        check_len(
            "PLE sequence hidden states",
            hidden_states.len(),
            expected_hidden,
        )?;
        check_len("PLE sequence output", output.len(), expected_hidden)?;
        Ok(())
    }

    fn apply_token_masked(
        &mut self,
        embedding: &[f32],
        hidden_states: &[f32],
        weights: &PleProjectionWeights<'_>,
        output: &mut [f32],
        active: bool,
    ) -> Result<(), PleOpError> {
        check_len("PLE embedding", embedding.len(), self.embed_dim)?;
        check_len("PLE hidden states", hidden_states.len(), self.channels())?;
        check_len("PLE output", output.len(), self.channels())?;
        weights.validate(
            self.hc_count,
            self.hidden_size,
            self.embed_dim,
            self.kernel_size,
        )?;

        if active {
            matmul_row(
                weights.key_proj,
                embedding,
                &mut self.key_raw,
                self.embed_dim,
            );
            matmul_row(
                weights.value_proj,
                embedding,
                &mut self.value,
                self.embed_dim,
            );
            rms_norm_groups(
                &self.key_raw,
                weights.norm_key,
                self.hc_count,
                self.hidden_size,
                PLE_RMS_EPS,
                true,
                &mut self.key_normed,
            );
            rms_norm_groups(
                hidden_states,
                weights.norm_query,
                self.hc_count,
                self.hidden_size,
                PLE_RMS_EPS,
                true,
                &mut self.query_normed,
            );
            for branch in 0..self.hc_count {
                let begin = branch * self.hidden_size;
                let end = begin + self.hidden_size;
                let dot = self.key_normed[begin..end]
                    .iter()
                    .zip(&self.query_normed[begin..end])
                    .map(|(&key, &query)| key * query)
                    .sum::<f32>()
                    / (self.hidden_size as f32).sqrt();
                let sign = if dot >= 0.0 { 1.0 } else { -1.0 };
                let transformed = sign * dot.abs().max(1.0e-6).sqrt();
                let gate = sigmoid(transformed);
                for (dst, &value) in self.gated_value[begin..end].iter_mut().zip(&self.value) {
                    *dst = gate * value;
                }
            }
            rms_norm_groups(
                &self.gated_value,
                weights.norm_conv,
                self.hc_count,
                self.hidden_size,
                PLE_RMS_EPS,
                true,
                &mut self.gated_normed,
            );
        } else {
            self.key_raw.fill(0.0);
            self.key_normed.fill(0.0);
            self.query_normed.fill(0.0);
            self.value.fill(0.0);
            self.gated_value.fill(0.0);
            self.gated_normed.fill(0.0);
        }

        // The reference uses [history rows, current token] as the causal
        // convolution window.  Since history_rows=(kernel-1)*dilation, the
        // current row is exactly the final index for the final tap.
        for channel in 0..self.channels() {
            let mut total = 0.0f32;
            for tap in 0..self.kernel_size {
                let source = tap * self.dilation;
                let value = if source < self.history_rows() {
                    self.history[source * self.channels() + channel]
                } else {
                    self.gated_normed[channel]
                };
                total += value * weights.conv_kernel[channel * self.kernel_size + tap];
            }
            output[channel] = self.gated_value[channel] + silu(total);
        }
        if self.history_rows() > 0 {
            let channels = self.channels();
            self.history.copy_within(channels.., 0);
            let begin = self.history.len() - channels;
            self.history[begin..].copy_from_slice(&self.gated_normed);
        }
        Ok(())
    }
}

/// Decode a contiguous token-major PLE staging buffer from BF16 to F32.
///
/// Input ordering is `[tokens, 16, 160]` and output is its flattened F32
/// representation.  The operation intentionally requires exact lengths so a
/// short row can never be silently padded with zeros.
pub fn decode_bf16_rows(staged: &[u8], output: &mut [f32]) -> Result<(), PleOpError> {
    if staged.len() % 2 != 0 {
        return Err(PleOpError::Shape(
            "BF16 PLE staging byte length must be even",
        ));
    }
    let expected = staged.len() / 2;
    check_len("decoded BF16 PLE rows", output.len(), expected)?;
    for (dst, bytes) in output.iter_mut().zip(staged.chunks_exact(2)) {
        *dst = f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16);
    }
    Ok(())
}

fn check_len(what: &'static str, actual: usize, expected: usize) -> Result<(), PleOpError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PleOpError::Length {
            what,
            expected,
            actual,
        })
    }
}

fn matmul_row(weight: &[f32], input: &[f32], output: &mut [f32], input_width: usize) {
    debug_assert_eq!(weight.len(), output.len() * input_width);
    for (row, dst) in weight.chunks_exact(input_width).zip(output.iter_mut()) {
        *dst = row
            .iter()
            .zip(input)
            .map(|(&weight, &value)| weight * value)
            .sum();
    }
}

fn rms_norm_groups(
    input: &[f32],
    weight: &[f32],
    groups: usize,
    group_size: usize,
    epsilon: f32,
    zero_centered: bool,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), groups * group_size);
    debug_assert_eq!(weight.len(), groups * group_size);
    debug_assert_eq!(output.len(), groups * group_size);
    for group in 0..groups {
        let begin = group * group_size;
        let end = begin + group_size;
        let source = &input[begin..end];
        let mean_square =
            source.iter().map(|&value| value * value).sum::<f32>() / group_size as f32;
        let inverse = (mean_square + epsilon.max(1.0e-12)).sqrt().recip();
        for offset in 0..group_size {
            let multiplier = if zero_centered {
                1.0 + weight[begin + offset]
            } else {
                weight[begin + offset]
            };
            output[begin + offset] = source[offset] * inverse * multiplier;
        }
    }
}

#[inline]
pub fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        let exponential = (-value).exp();
        1.0 / (1.0 + exponential)
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[inline]
pub fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // hc=2, hidden=2, embed=2, kernel=2.  Identity projections make the
        // causal-state transition easy to compare without a large fixture.
        (
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0, 1.0],
            vec![0.0; 4],
            vec![0.0; 4],
            vec![0.0; 4],
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        )
    }

    #[test]
    fn bf16_decode_preserves_stored_bits() {
        let staged = [0x00, 0x3f, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x7f];
        let mut output = [0.0; 4];
        decode_bf16_rows(&staged, &mut output).unwrap();
        assert_eq!(output[0].to_bits(), 0x3f00_0000);
        assert_eq!(output[1].to_bits(), 0x3f80_0000);
        assert_eq!(output[2], 0.0);
        assert!(output[3].is_infinite());
    }

    #[test]
    fn sequence_and_incremental_paths_share_causal_history() {
        let (key, value, key_norm, query_norm, conv_norm, conv) = tiny_weights();
        let weights =
            PleProjectionWeights::new(&key, &value, &key_norm, &query_norm, &conv_norm, &conv);
        let embeddings = [1.0, 2.0, 2.0, 1.0];
        let hidden = [2.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 3.0];
        let mut sequence = PleLayerState::new(2, 2, 2, 2, 1).unwrap();
        let mut sequence_output = [0.0; 8];
        sequence
            .apply_sequence(&embeddings, &hidden, 2, &weights, &mut sequence_output)
            .unwrap();

        let mut incremental = PleLayerState::new(2, 2, 2, 2, 1).unwrap();
        let mut first = [0.0; 4];
        let mut second = [0.0; 4];
        incremental
            .apply_token(&embeddings[..2], &hidden[..4], &weights, &mut first)
            .unwrap();
        incremental
            .apply_token(&embeddings[2..], &hidden[4..], &weights, &mut second)
            .unwrap();
        assert_eq!(&sequence_output[..4], &first);
        assert_eq!(&sequence_output[4..], &second);
        assert_eq!(sequence.history(), incremental.history());
    }

    #[test]
    fn malformed_projection_is_rejected_before_math() {
        let weights = PleProjectionWeights::new(&[1.0], &[], &[], &[], &[], &[]);
        let error = weights.validate(2, 2, 2, 2).unwrap_err();
        assert!(matches!(error, PleOpError::Length { .. }));
    }
}
