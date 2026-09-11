// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! GPU-owned Qwen4 request state and a small CPU reference state for contracts.
//!
//! `Qwen4State` is the production owner: every recurrent, convolution, QSA,
//! full-KV, PLE-convolution, and HC-feedback buffer is a `GpuTensor` and is
//! released through `free_gpu`.  The reference structs are explicitly named
//! `Reference*` and are only used by CPU equation tests/parity probes.

use crate::config::{LayerType, Qwen4Config};
use crate::ple::{PleHashMetadata, PleHistory, PLE_HEAD_COUNT};
use rdna_compute::{DType, Gpu, GpuTensor};
use std::fmt;

/// Reference-only GDN state used by CPU equation tests.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceGdnLayerState {
    pub recurrent: Vec<f32>,
    pub conv_history: Vec<f32>,
    pub conv_cursor: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_channels: usize,
    pub conv_kernel: usize,
}

impl ReferenceGdnLayerState {
    pub fn new(config: &Qwen4Config) -> Result<Self, StateError> {
        let key_heads = config.linear_num_key_heads;
        let value_heads = config.linear_num_value_heads;
        let key_dim = config.linear_key_head_dim;
        let value_dim = config.linear_value_head_dim;
        let conv_channels = key_heads
            .checked_mul(key_dim)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_add(value_heads.checked_mul(value_dim)?))
            .ok_or(StateError::DimensionOverflow)?;
        let recurrent_len = value_heads
            .checked_mul(value_dim)
            .and_then(|v| v.checked_mul(key_dim))
            .ok_or(StateError::DimensionOverflow)?;
        let conv_len = conv_channels
            .checked_mul(config.linear_conv_kernel_dim.saturating_sub(1))
            .ok_or(StateError::DimensionOverflow)?;
        Ok(Self {
            recurrent: vec![0.0; recurrent_len],
            conv_history: vec![0.0; conv_len],
            conv_cursor: 0,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_channels,
            conv_kernel: config.linear_conv_kernel_dim,
        })
    }

    pub fn reset(&mut self) {
        self.recurrent.fill(0.0);
        self.conv_history.fill(0.0);
        self.conv_cursor = 0;
    }

    pub fn conv_history_rows(&self) -> usize {
        self.conv_kernel.saturating_sub(1)
    }

    pub fn push_conv_row(&mut self, row: &[f32]) -> Result<(), StateError> {
        if row.len() != self.conv_channels {
            return Err(StateError::Length {
                what: "reference GDN convolution row",
                expected: self.conv_channels,
                actual: row.len(),
            });
        }
        let rows = self.conv_history_rows();
        if rows == 0 {
            return Ok(());
        }
        let start = self.conv_cursor * self.conv_channels;
        self.conv_history[start..start + self.conv_channels].copy_from_slice(row);
        self.conv_cursor = (self.conv_cursor + 1) % rows;
        Ok(())
    }
}

/// Reference-only PLE convolution state.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferencePleState {
    pub history: PleHistory,
    pub conv_history: Vec<f32>,
    pub conv_cursor: usize,
    pub channels: usize,
    pub kernel: usize,
    pub dilation: usize,
}

impl ReferencePleState {
    pub fn new(config: &Qwen4Config, eos_token_id: u32) -> Result<Self, StateError> {
        let rows = config
            .ple_conv_kernel_size
            .checked_sub(1)
            .ok_or(StateError::DimensionOverflow)?
            .checked_mul(3)
            .ok_or(StateError::DimensionOverflow)?;
        let len = rows
            .checked_mul(config.hidden_size)
            .ok_or(StateError::DimensionOverflow)?;
        Ok(Self {
            history: PleHistory::new(eos_token_id),
            conv_history: vec![0.0; len],
            conv_cursor: 0,
            channels: config.hidden_size,
            kernel: config.ple_conv_kernel_size,
            dilation: 3,
        })
    }

    pub fn reset(&mut self) {
        self.history.reset();
        self.conv_history.fill(0.0);
        self.conv_cursor = 0;
    }

    pub fn hash_token(&mut self, metadata: &PleHashMetadata, token: u32) -> [u64; PLE_HEAD_COUNT] {
        self.history.hash_token(metadata, token)
    }

    pub fn push_conv_row(&mut self, row: &[f32]) -> Result<(), StateError> {
        if row.len() != self.channels {
            return Err(StateError::Length {
                what: "reference PLE convolution row",
                expected: self.channels,
                actual: row.len(),
            });
        }
        let rows = self.conv_history.len() / self.channels;
        let offset = self.conv_cursor * self.channels;
        self.conv_history[offset..offset + self.channels].copy_from_slice(row);
        self.conv_cursor = (self.conv_cursor + 1) % rows.max(1);
        Ok(())
    }
}

/// Reference-only PLE/QSA metadata used by CPU probes.  Full KV values are
/// ordinary vectors here; production Qwen4State keeps them on the device.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceQsaLayerState {
    pub full_keys: Vec<f32>,
    pub full_values: Vec<f32>,
    pub raw_index_keys: Vec<f32>,
    pub pooled_keys: Vec<f32>,
    pub pooled_positions: Vec<usize>,
    pub partial_keys: Vec<f32>,
    pub partial_values: Vec<f32>,
    pub partial_positions: Vec<usize>,
    pub selected_indices: Vec<usize>,
    pub position: usize,
}

impl ReferenceQsaLayerState {
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            full_keys: Vec::with_capacity(max_seq_len),
            full_values: Vec::with_capacity(max_seq_len),
            raw_index_keys: Vec::with_capacity(max_seq_len),
            pooled_keys: Vec::new(),
            pooled_positions: Vec::new(),
            partial_keys: Vec::new(),
            partial_values: Vec::new(),
            partial_positions: Vec::new(),
            selected_indices: Vec::new(),
            position: 0,
        }
    }
}

/// Reference-only HC feedback.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceHyperConnectionState {
    pub feedback: Vec<f32>,
}

/// A production GDN state block.  The tensors are F32 and must be freed by
/// `Qwen4State::free_gpu`; no `Drop` attempts to call HIP.
pub struct GdnGpuState {
    pub recurrent: GpuTensor,
    pub conv: GpuTensor,
}

/// A production QSA block.  Full K/V and raw/indexer buffers are allocated at
/// max sequence capacity once; `*_len` fields are active append lengths.
pub struct QsaGpuState {
    pub full_keys: GpuTensor,
    pub full_values: GpuTensor,
    pub raw_index_keys: GpuTensor,
    pub pooled_keys: GpuTensor,
    pub partial_keys: GpuTensor,
    pub partial_values: GpuTensor,
    pub selected_indices: GpuTensor,
    pub full_capacity: usize,
    pub raw_capacity: usize,
    pub pooled_capacity: usize,
    pub partial_capacity: usize,
    pub selected_capacity: usize,
    pub position_capacity: usize,
    pub full_len: usize,
    pub raw_len: usize,
    pub pooled_len: usize,
    pub partial_len: usize,
    pub selected_len: usize,
    pub position: usize,
}

impl QsaGpuState {
    fn mark(&self) -> QsaMark {
        QsaMark {
            full_len: self.full_len,
            raw_len: self.raw_len,
            pooled_len: self.pooled_len,
            partial_len: self.partial_len,
            selected_len: self.selected_len,
            position: self.position,
        }
    }

    fn mark_capacity(&self) -> QsaMarkCapacity {
        QsaMarkCapacity {
            full: self.full_capacity,
            raw: self.raw_capacity,
            pooled: self.pooled_capacity,
            partial: self.partial_capacity,
            selected: self.selected_capacity,
            position: self.position_capacity,
        }
    }
}

/// The sole production request-state owner for Qwen4.
pub struct Qwen4State {
    pub gdn: Vec<GdnGpuState>,
    pub qsa: Vec<QsaGpuState>,
    pub ple_conv: GpuTensor,
    pub hyper_feedback: GpuTensor,
    pub ple_history: PleHistory,
    pub position: usize,
    pub max_seq_len: usize,
    pub qsa_selected_capacity: usize,
}

impl Qwen4State {
    /// Allocate all mutable device buffers once.  QSA full K/V and raw key
    /// buffers are fixed-capacity arenas; only active lengths are mutable.
    pub fn new(
        gpu: &mut Gpu,
        config: &Qwen4Config,
        max_seq_len: usize,
    ) -> Result<Self, StateError> {
        config.validate().map_err(StateError::Config)?;
        if max_seq_len == 0 || max_seq_len > config.max_position_embeddings {
            return Err(StateError::InvalidCapacity);
        }
        let gdn_recurrent = config
            .linear_num_value_heads
            .checked_mul(config.linear_value_head_dim)
            .and_then(|v| v.checked_mul(config.linear_key_head_dim))
            .ok_or(StateError::DimensionOverflow)?;
        let conv_channels = config
            .linear_num_key_heads
            .checked_mul(config.linear_key_head_dim)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| {
                v.checked_add(
                    config
                        .linear_num_value_heads
                        .checked_mul(config.linear_value_head_dim)?,
                )
            })
            .ok_or(StateError::DimensionOverflow)?;
        let conv_len = conv_channels
            .checked_mul(config.linear_conv_kernel_dim.saturating_sub(1))
            .ok_or(StateError::DimensionOverflow)?;
        let full_width = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or(StateError::DimensionOverflow)?;
        let raw_width = config
            .indexer_kv_heads
            .checked_mul(config.indexer_head_dim)
            .ok_or(StateError::DimensionOverflow)?;
        let pooled_capacity = max_seq_len
            .checked_add(config.indexer_compress_ratio - 1)
            .ok_or(StateError::DimensionOverflow)?
            / config.indexer_compress_ratio;
        let selected_capacity = config.qsa_selected_capacity();
        let mut allocated = Vec::new();
        let allocation = (|| -> Result<(), StateError> {
            for kind in &config.layer_types {
                match kind {
                    LayerType::LinearAttention => {
                        allocated.push(
                            gpu.zeros(&[gdn_recurrent], DType::F32)
                                .map_err(StateError::Hip)?,
                        );
                        allocated.push(
                            gpu.zeros(&[conv_len], DType::F32)
                                .map_err(StateError::Hip)?,
                        );
                    }
                    LayerType::FullAttention => {
                        let full_elements = max_seq_len
                            .checked_mul(full_width)
                            .ok_or(StateError::DimensionOverflow)?;
                        let raw_elements = max_seq_len
                            .checked_mul(raw_width)
                            .ok_or(StateError::DimensionOverflow)?;
                        let pooled_elements = pooled_capacity
                            .checked_mul(raw_width)
                            .ok_or(StateError::DimensionOverflow)?;
                        let partial_key_elements = config
                            .indexer_compress_ratio
                            .checked_mul(raw_width)
                            .ok_or(StateError::DimensionOverflow)?;
                        let partial_value_elements = config
                            .indexer_compress_ratio
                            .checked_mul(full_width)
                            .ok_or(StateError::DimensionOverflow)?;
                        let selected_bytes = selected_capacity
                            .checked_mul(std::mem::size_of::<i32>())
                            .ok_or(StateError::DimensionOverflow)?;
                        for (elements, dtype) in [
                            (full_elements, DType::F32),
                            (full_elements, DType::F32),
                            (raw_elements, DType::F32),
                            (pooled_elements, DType::F32),
                            (partial_key_elements, DType::F32),
                            (partial_value_elements, DType::F32),
                            (selected_bytes, DType::Raw),
                        ] {
                            allocated.push(gpu.zeros(&[elements], dtype).map_err(StateError::Hip)?);
                        }
                    }
                }
            }
            let ple_elements = 9usize
                .checked_mul(config.hidden_size)
                .ok_or(StateError::DimensionOverflow)?;
            allocated.push(
                gpu.zeros(&[ple_elements], DType::F32)
                    .map_err(StateError::Hip)?,
            );
            let feedback_elements = config
                .hc_count
                .checked_mul(config.hidden_size)
                .ok_or(StateError::DimensionOverflow)?;
            allocated.push(
                gpu.zeros(&[feedback_elements], DType::F32)
                    .map_err(StateError::Hip)?,
            );
            Ok(())
        })();
        if let Err(error) = allocation {
            for tensor in allocated {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(error);
        }

        let expected_tensors = config
            .layer_types
            .iter()
            .map(|kind| match kind {
                LayerType::LinearAttention => 2,
                LayerType::FullAttention => 7,
            })
            .sum::<usize>()
            .checked_add(2)
            .ok_or(StateError::DimensionOverflow)?;
        if allocated.len() != expected_tensors {
            for tensor in allocated {
                let _ = gpu.free_tensor(tensor);
            }
            return Err(StateError::AllocationBookkeeping);
        }

        let mut tensors = allocated.into_iter();
        let mut next_tensor = || {
            tensors
                .next()
                .expect("Qwen4 state tensor count checked before reconstruction")
        };
        let mut gdn = Vec::with_capacity(config.n_linear_layers());
        let mut qsa = Vec::with_capacity(config.n_full_layers());
        for kind in &config.layer_types {
            match kind {
                LayerType::LinearAttention => {
                    let recurrent = next_tensor();
                    let conv = next_tensor();
                    gdn.push(GdnGpuState { recurrent, conv });
                }
                LayerType::FullAttention => {
                    let full_keys = next_tensor();
                    let full_values = next_tensor();
                    let raw_index_keys = next_tensor();
                    let pooled_keys = next_tensor();
                    let partial_keys = next_tensor();
                    let partial_values = next_tensor();
                    let selected_indices = next_tensor();
                    qsa.push(QsaGpuState {
                        full_keys,
                        full_values,
                        raw_index_keys,
                        pooled_keys,
                        partial_keys,
                        partial_values,
                        selected_indices,
                        full_capacity: max_seq_len,
                        raw_capacity: max_seq_len,
                        pooled_capacity,
                        partial_capacity: config.indexer_compress_ratio,
                        selected_capacity,
                        position_capacity: max_seq_len,
                        full_len: 0,
                        raw_len: 0,
                        pooled_len: 0,
                        partial_len: 0,
                        selected_len: 0,
                        position: 0,
                    });
                }
            }
        }
        let ple_conv = next_tensor();
        let hyper_feedback = next_tensor();
        Ok(Self {
            gdn,
            qsa,
            ple_conv,
            hyper_feedback,
            ple_history: PleHistory::new(config.eos_token_id),
            position: 0,
            max_seq_len,
            qsa_selected_capacity: selected_capacity,
        })
    }

    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), StateError> {
        for layer in &self.gdn {
            gpu.hip
                .memset(&layer.recurrent.buf, 0, layer.recurrent.buf.size())
                .map_err(StateError::Hip)?;
            gpu.hip
                .memset(&layer.conv.buf, 0, layer.conv.buf.size())
                .map_err(StateError::Hip)?;
        }
        for layer in &mut self.qsa {
            for tensor in [
                &layer.full_keys,
                &layer.full_values,
                &layer.raw_index_keys,
                &layer.pooled_keys,
                &layer.partial_keys,
                &layer.partial_values,
                &layer.selected_indices,
            ] {
                gpu.hip
                    .memset(&tensor.buf, 0, tensor.buf.size())
                    .map_err(StateError::Hip)?;
            }
            layer.full_len = 0;
            layer.raw_len = 0;
            layer.pooled_len = 0;
            layer.partial_len = 0;
            layer.selected_len = 0;
            layer.position = 0;
        }
        gpu.hip
            .memset(&self.ple_conv.buf, 0, self.ple_conv.buf.size())
            .map_err(StateError::Hip)?;
        gpu.hip
            .memset(&self.hyper_feedback.buf, 0, self.hyper_feedback.buf.size())
            .map_err(StateError::Hip)?;
        self.ple_history.reset();
        self.position = 0;
        Ok(())
    }

    /// Take a GPU snapshot of all fixed in-place state.  Full-capacity QSA/KV
    /// arenas are not cloned; their append marks are recorded and restore only
    /// rewinds active lengths/positions.
    pub fn snapshot(&self, gpu: &mut Gpu) -> Result<Qwen4StateSnapshot, StateError> {
        let mut recurrent = Vec::with_capacity(self.gdn.len());
        let mut conv = Vec::with_capacity(self.gdn.len());
        let mut ple_conv = None;
        let mut hyper_feedback = None;
        let result = (|| -> Result<Qwen4StateSnapshot, StateError> {
            for layer in &self.gdn {
                let r = gpu
                    .zeros(&layer.recurrent.shape, DType::F32)
                    .map_err(StateError::Hip)?;
                if let Err(error) = gpu.copy_d2d(&layer.recurrent, &r, layer.recurrent.byte_size())
                {
                    let _ = gpu.free_tensor(r);
                    return Err(StateError::Hip(error));
                }
                recurrent.push(r);

                let c = gpu
                    .zeros(&layer.conv.shape, DType::F32)
                    .map_err(StateError::Hip)?;
                if let Err(error) = gpu.copy_d2d(&layer.conv, &c, layer.conv.byte_size()) {
                    let _ = gpu.free_tensor(c);
                    return Err(StateError::Hip(error));
                }
                conv.push(c);
            }

            let p = gpu
                .zeros(&self.ple_conv.shape, DType::F32)
                .map_err(StateError::Hip)?;
            if let Err(error) = gpu.copy_d2d(&self.ple_conv, &p, self.ple_conv.byte_size()) {
                let _ = gpu.free_tensor(p);
                return Err(StateError::Hip(error));
            }
            ple_conv = Some(p);

            let h = gpu
                .zeros(&self.hyper_feedback.shape, DType::F32)
                .map_err(StateError::Hip)?;
            if let Err(error) =
                gpu.copy_d2d(&self.hyper_feedback, &h, self.hyper_feedback.byte_size())
            {
                let _ = gpu.free_tensor(h);
                return Err(StateError::Hip(error));
            }
            hyper_feedback = Some(h);

            let ple_conv = ple_conv.take().ok_or(StateError::AllocationBookkeeping)?;
            let hyper_feedback = hyper_feedback
                .take()
                .ok_or(StateError::AllocationBookkeeping)?;
            Ok(Qwen4StateSnapshot {
                recurrent: std::mem::take(&mut recurrent),
                conv: std::mem::take(&mut conv),
                ple_conv,
                hyper_feedback,
                ple_history: self.ple_history,
                position: self.position,
                qsa: self
                    .qsa
                    .iter()
                    .map(|layer| QsaMark {
                        full_len: layer.full_len,
                        raw_len: layer.raw_len,
                        pooled_len: layer.pooled_len,
                        partial_len: layer.partial_len,
                        selected_len: layer.selected_len,
                        position: layer.position,
                    })
                    .collect(),
            })
        })();

        if result.is_err() {
            for tensor in recurrent.into_iter().chain(conv) {
                let _ = gpu.free_tensor(tensor);
            }
            if let Some(tensor) = ple_conv {
                let _ = gpu.free_tensor(tensor);
            }
            if let Some(tensor) = hyper_feedback {
                let _ = gpu.free_tensor(tensor);
            }
        }
        result
    }

    /// Restore and consume a snapshot.  Snapshot buffers are always freed,
    /// including on a copy failure, by the explicit owner below.
    pub fn restore(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
    ) -> Result<(), StateError> {
        let result = (|| {
            if snapshot.recurrent.len() != self.gdn.len()
                || snapshot.conv.len() != self.gdn.len()
                || snapshot.qsa.len() != self.qsa.len()
            {
                return Err(StateError::SnapshotShape);
            }
            if snapshot.position > self.max_seq_len {
                return Err(StateError::SnapshotLength);
            }

            // Validate every row mark before the first device copy.  A malformed
            // snapshot therefore cannot partially mutate recurrent state or
            // append-length metadata.
            let capacities: Vec<_> = self.qsa.iter().map(QsaGpuState::mark_capacity).collect();
            let mut restored_marks: Vec<_> = self.qsa.iter().map(QsaGpuState::mark).collect();
            apply_qsa_marks(&mut restored_marks, &snapshot.qsa, &capacities)?;

            for ((layer, recurrent), conv) in
                self.gdn.iter().zip(&snapshot.recurrent).zip(&snapshot.conv)
            {
                gpu.copy_d2d(recurrent, &layer.recurrent, layer.recurrent.byte_size())
                    .map_err(StateError::Hip)?;
                gpu.copy_d2d(conv, &layer.conv, layer.conv.byte_size())
                    .map_err(StateError::Hip)?;
            }
            gpu.copy_d2d(
                &snapshot.ple_conv,
                &self.ple_conv,
                self.ple_conv.byte_size(),
            )
            .map_err(StateError::Hip)?;
            gpu.copy_d2d(
                &snapshot.hyper_feedback,
                &self.hyper_feedback,
                self.hyper_feedback.byte_size(),
            )
            .map_err(StateError::Hip)?;
            for (layer, mark) in self.qsa.iter_mut().zip(restored_marks) {
                layer.full_len = mark.full_len;
                layer.raw_len = mark.raw_len;
                layer.pooled_len = mark.pooled_len;
                layer.partial_len = mark.partial_len;
                layer.selected_len = mark.selected_len;
                layer.position = mark.position;
            }
            self.ple_history = snapshot.ple_history;
            self.position = snapshot.position;
            Ok(())
        })();
        snapshot.free_gpu(gpu);
        result
    }

    pub fn commit(
        &mut self,
        snapshot: Qwen4StateSnapshot,
        gpu: &mut Gpu,
    ) -> Result<(), StateError> {
        snapshot.free_gpu(gpu);
        Ok(())
    }

    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), StateError> {
        let mut first = None;
        let mut free = |tensor: GpuTensor| {
            if let Err(error) = gpu.free_tensor(tensor) {
                if first.is_none() {
                    first = Some(error);
                }
            }
        };
        for layer in self.gdn {
            free(layer.recurrent);
            free(layer.conv);
        }
        for layer in self.qsa {
            free(layer.full_keys);
            free(layer.full_values);
            free(layer.raw_index_keys);
            free(layer.pooled_keys);
            free(layer.partial_keys);
            free(layer.partial_values);
            free(layer.selected_indices);
        }
        free(self.ple_conv);
        free(self.hyper_feedback);
        first.map_or(Ok(()), |error| Err(StateError::Hip(error)))
    }

    pub fn qsa_mut(&mut self, full_layer_index: usize) -> Option<&mut QsaGpuState> {
        self.qsa.get_mut(full_layer_index)
    }

    pub fn gdn_mut(&mut self, linear_layer_index: usize) -> Option<&mut GdnGpuState> {
        self.gdn.get_mut(linear_layer_index)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QsaMark {
    full_len: usize,
    raw_len: usize,
    pooled_len: usize,
    partial_len: usize,
    selected_len: usize,
    position: usize,
}

#[derive(Debug, Clone, Copy)]
struct QsaMarkCapacity {
    full: usize,
    raw: usize,
    pooled: usize,
    partial: usize,
    selected: usize,
    position: usize,
}

fn validate_qsa_marks(marks: &[QsaMark], capacities: &[QsaMarkCapacity]) -> Result<(), StateError> {
    if marks.len() != capacities.len() {
        return Err(StateError::SnapshotShape);
    }
    for (mark, capacity) in marks.iter().zip(capacities) {
        if mark.full_len > capacity.full
            || mark.raw_len > capacity.raw
            || mark.pooled_len > capacity.pooled
            || mark.partial_len > capacity.partial
            || mark.selected_len > capacity.selected
            || mark.position > capacity.position
        {
            return Err(StateError::SnapshotLength);
        }
    }
    Ok(())
}

fn apply_qsa_marks(
    current: &mut [QsaMark],
    incoming: &[QsaMark],
    capacities: &[QsaMarkCapacity],
) -> Result<(), StateError> {
    validate_qsa_marks(incoming, capacities)?;
    if current.len() != incoming.len() {
        return Err(StateError::SnapshotShape);
    }
    current.copy_from_slice(incoming);
    Ok(())
}

/// GPU snapshot owner.  It never clones full-capacity QSA/KV arenas.
pub struct Qwen4StateSnapshot {
    recurrent: Vec<GpuTensor>,
    conv: Vec<GpuTensor>,
    ple_conv: GpuTensor,
    hyper_feedback: GpuTensor,
    ple_history: PleHistory,
    qsa: Vec<QsaMark>,
    position: usize,
}

impl Qwen4StateSnapshot {
    pub fn free_gpu(self, gpu: &mut Gpu) {
        for tensor in self
            .recurrent
            .into_iter()
            .chain(self.conv)
            .chain([self.ple_conv, self.hyper_feedback])
        {
            let _ = gpu.free_tensor(tensor);
        }
    }
}

#[derive(Debug)]
pub enum StateError {
    Config(String),
    Hip(hip_bridge::HipError),
    DimensionOverflow,
    InvalidCapacity,
    Length {
        what: &'static str,
        expected: usize,
        actual: usize,
    },
    AllocationBookkeeping,
    SnapshotShape,
    SnapshotLength,
}

impl fmt::Display for StateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "Qwen4 state config: {message}"),
            Self::Hip(error) => write!(f, "Qwen4 state HIP: {error}"),
            Self::DimensionOverflow => write!(f, "Qwen4 state dimension overflow"),
            Self::InvalidCapacity => write!(f, "Qwen4 state capacity is invalid"),
            Self::Length {
                what,
                expected,
                actual,
            } => write!(f, "{what}: expected {expected}, got {actual}"),
            Self::AllocationBookkeeping => write!(f, "Qwen4 state allocation bookkeeping failure"),
            Self::SnapshotShape => write!(f, "Qwen4 snapshot shape mismatch"),
            Self::SnapshotLength => write!(f, "Qwen4 snapshot mark exceeds state capacity"),
        }
    }
}

impl std::error::Error for StateError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_state_has_nine_ple_history_rows() {
        let layers: Vec<_> = (0..48)
            .map(|layer| {
                if layer % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        let value = serde_json::json!({
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "dtype": "bfloat16",
                "hidden_size": 2560,
                "vocab_size": 248320,
                "num_hidden_layers": 48,
                "max_position_embeddings": 262144,
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "partial_rotary_factor": 0.25,
                "rope_parameters": {"rope_theta": 10000000.0},
                "attention_bias": false,
                "layer_types": layers,
                "full_attention_interval": 4,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "mamba_ssm_dtype": "float32",
                "indexer_n_heads": 4,
                "indexer_kv_heads": 1,
                "indexer_head_dim": 128,
                "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "moe_intermediate_size": 640,
                "shared_expert_intermediate_size": 640,
                "hc_count": 4,
                "hc_lowrank": 320,
                "ple_layer_ids": [2],
                "ple_conv_kernel_size": 4,
                "ple_embed_dim": 2560,
                "split_ngram_parts": 128,
                "heads_per_ngram": 8,
                "ngram_size": 3,
                "ngram_vocab_size_base": 20000000,
                "make_ngram_vocab_size_divisible_by": 128,
                "eos_token_id": 248044,
                "tie_word_embeddings": false,
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "output_gate_type": "sigmoid",
                "mtp": {
                    "hybrid": true,
                    "layer_types": ["full_attention"],
                    "num_hidden_layers": 1,
                    "rope_theta": 10000000.0
                }
            }
        });
        let config = Qwen4Config::from_value(&value).expect("reference config");
        let state = ReferencePleState::new(&config, config.eos_token_id).expect("PLE state");
        assert_eq!(state.conv_history.len() / state.channels, 9);
    }

    #[test]
    fn malformed_qsa_snapshot_leaves_marks_unchanged() {
        let capacities = vec![QsaMarkCapacity {
            full: 4,
            raw: 4,
            pooled: 1,
            partial: 4,
            selected: 5,
            position: 4,
        }];
        let original = vec![QsaMark {
            full_len: 2,
            raw_len: 2,
            pooled_len: 1,
            partial_len: 2,
            selected_len: 2,
            position: 2,
        }];
        let malformed = vec![QsaMark {
            full_len: 5,
            raw_len: 2,
            pooled_len: 1,
            partial_len: 2,
            selected_len: 2,
            position: 2,
        }];
        let mut restored = original.clone();
        let error = apply_qsa_marks(&mut restored, &malformed, &capacities).unwrap_err();
        assert!(matches!(error, StateError::SnapshotLength));
        assert_eq!(restored, original);
    }
}
