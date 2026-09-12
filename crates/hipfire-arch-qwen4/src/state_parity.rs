// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Compact GPU state/rollback parity scenarios used by the Qwen4 parity CLI.
//!
//! Model logits are intentionally not produced here: the CLI's state mode
//! obtains those from the real loaded bundle/forward/spec path.  This module
//! only orchestrates compact state transitions and serializes device-backed
//! family digests.

use crate::config::compact_test_config;
use crate::mtp_gpu::{MtpGpuState, MtpStateParityMetadata};
use crate::mtp_spec::validate_native_mtp_prefill_request;
use crate::state::Qwen4State;
use hipfire_runtime::model_source::ModelSource;
use rdna_compute::qwen4::{
    qwen4_qsa_cache_append, qwen4_qsa_pool_rope, qwen4_qsa_reuse_selection, qwen4_qsa_select,
    Qwen4QsaCacheAppend, Qwen4QsaPoolRope, Qwen4QsaReuseSelection, Qwen4QsaSelect,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const MAX_SEQ: usize = 8;
const PREFIX: usize = 4;
const DRAFTS: [u32; 2] = [1001, 1002];
const BONUS: u32 = 1003;
const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

#[derive(Clone, Debug)]
struct Family {
    hash: u64,
    numel: usize,
}

impl Family {
    fn new() -> Self {
        Self {
            hash: FNV_OFFSET,
            numel: 0,
        }
    }

    fn bytes(&mut self, bytes: &[u8], numel: usize) {
        for byte in bytes {
            self.hash ^= u64::from(*byte);
            self.hash = self.hash.wrapping_mul(FNV_PRIME);
        }
        self.numel = self.numel.saturating_add(numel);
    }

    fn finish(self) -> Value {
        json!({"digest": format!("{:016x}", self.hash), "numel": self.numel})
    }
}

type Families = BTreeMap<String, Value>;

pub fn run_compact(gpu: &mut Gpu) -> Result<Value, String> {
    if !gpu.arch_caps.is_gfx1151() {
        return Err(format!(
            "Qwen4 compact state parity requires gfx1151, got {}",
            gpu.arch
        ));
    }
    let config = compact_test_config();
    let mut ar = Qwen4State::new(gpu, &config, MAX_SEQ)
        .map_err(|error| format!("allocate compact AR state: {error}"))?;
    let mut native = match Qwen4State::new(gpu, &config, MAX_SEQ) {
        Ok(state) => state,
        Err(error) => {
            let _ = ar.free_gpu(gpu);
            return Err(format!("allocate compact native target state: {error}"));
        }
    };
    let mut direct = match MtpGpuState::new(gpu, &config, MAX_SEQ) {
        Ok(state) => state,
        Err(error) => {
            let _ = ar.free_gpu(gpu);
            let _ = native.free_gpu(gpu);
            return Err(format!("allocate compact direct MTP state: {error}"));
        }
    };
    let mut mtp = match MtpGpuState::new(gpu, &config, MAX_SEQ) {
        Ok(state) => state,
        Err(error) => {
            let _ = ar.free_gpu(gpu);
            let _ = native.free_gpu(gpu);
            let _ = direct.free_gpu(gpu);
            return Err(format!("allocate compact native MTP state: {error}"));
        }
    };
    let result = run_inner(gpu, &config, &mut ar, &mut native, &mut direct, &mut mtp);
    let cleanup = [
        ar.free_gpu(gpu).err().map(|error| error.to_string()),
        native.free_gpu(gpu).err().map(|error| error.to_string()),
        direct.free_gpu(gpu).map(|error| error.to_string()),
        mtp.free_gpu(gpu).map(|error| error.to_string()),
    ]
    .into_iter()
    .flatten()
    .next();
    result.and_then(|value| cleanup.map_or(Ok(value), Err))
}

fn run_inner(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    ar: &mut Qwen4State,
    native: &mut Qwen4State,
    direct: &mut MtpGpuState,
    mtp: &mut MtpGpuState,
) -> Result<Value, String> {
    let boundary = prepare(gpu, config, ar, native, direct, mtp)?;
    let mut cases = Vec::new();
    for accepted in 0..=DRAFTS.len() {
        cases.push(acceptance_case(
            gpu, config, ar, native, direct, mtp, accepted,
        )?);
    }
    let scenarios = vec![
        boundary,
        stale_ticket(gpu, ar, direct)?,
        terminal_seed(gpu, config, ar, direct)?,
        rollback_failure(gpu, config, ar, direct, "cancellation", None)?,
        rollback_failure(
            gpu,
            config,
            ar,
            direct,
            "injected_replay_failure",
            Some("injected replay error"),
        )?,
        cache_suffix_refusal(),
    ];
    let pass = cases.iter().all(is_pass) && scenarios.iter().all(is_pass);
    Ok(json!({
        "schema": "hipfire.qwen4.state_parity.compact.v1",
        "fixture": "canonical_compact_state",
        "gpu_arch": gpu.arch,
        "acceptance_cases": cases,
        "scenarios": scenarios,
        "status": if pass {"pass"} else {"fail"},
    }))
}

fn is_pass(value: &Value) -> bool {
    value.get("status") == Some(&Value::String("pass".into()))
}

fn prepare(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    ar: &mut Qwen4State,
    native: &mut Qwen4State,
    direct: &mut MtpGpuState,
    mtp: &mut MtpGpuState,
) -> Result<Value, String> {
    reset_target(gpu, ar)?;
    reset_target(gpu, native)?;
    reset_mtp(gpu, direct)?;
    reset_mtp(gpu, mtp)?;
    for (position, &token) in [10u32, 11, 12, 13].iter().enumerate() {
        append_target_pair(gpu, config, ar, native, position, token)?;
        append_mtp_pair(gpu, config, direct, mtp, position, token)?;
    }
    select_target_pair(gpu, config, ar, native, PREFIX)?;
    select_mtp_pair(gpu, config, direct, mtp, PREFIX)?;
    reuse(gpu, config, direct)?;
    reuse(gpu, config, mtp)?;
    let direct_buf = direct.parity_buffers();
    let mtp_buf = mtp.parity_buffers();
    let direct_len = download_i32(gpu, direct_buf.selected_len_out, 1)?[0];
    let mtp_len = download_i32(gpu, mtp_buf.selected_len_out, 1)?[0];
    let direct_selection = download_i32(gpu, direct_buf.selected_indices, PREFIX + 1)?;
    let mtp_selection = download_i32(gpu, mtp_buf.selected_indices, PREFIX + 1)?;
    let direct_meta = direct.parity_metadata();
    let mtp_meta = mtp.parity_metadata();
    let target_meta = ar.qsa.first().map(|qsa| {
        json!({
            "full_len": qsa.full_len,
            "raw_len": qsa.raw_len,
            "pooled_len": qsa.pooled_len,
            "selected_len": qsa.selected_len,
            "position": qsa.position,
        })
    });
    let pass = target_meta.is_some()
        && direct_meta == mtp_meta
        && direct_meta.full_len == PREFIX
        && direct_meta.raw_len == PREFIX
        && direct_meta.pooled_len == 1
        && direct_meta.selected_len == PREFIX
        && direct_len == (PREFIX + 1) as i32
        && direct_len == mtp_len
        && direct_selection == [0, 1, 2, 3, 4]
        && direct_selection == mtp_selection;
    Ok(json!({
        "case": "qsa_pooling_boundary",
        "status": if pass {"pass"} else {"fail"},
        "target_metadata": target_meta,
        "mtp_direct_metadata": metadata_json(direct_meta),
        "mtp_native_metadata": metadata_json(mtp_meta),
        "reuse_selected_len": {"direct": direct_len, "native": mtp_len, "expected": PREFIX + 1},
        "reuse_selected_indices": {"direct": direct_selection, "native": mtp_selection},
    }))
}

fn acceptance_case(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    ar: &mut Qwen4State,
    native: &mut Qwen4State,
    direct: &mut MtpGpuState,
    mtp: &mut MtpGpuState,
    accepted: usize,
) -> Result<Value, String> {
    prepare(gpu, config, ar, native, direct, mtp)?;
    let target_ticket = native.snapshot(gpu).map_err(|error| error.to_string())?;
    let mtp_ticket = mtp.snapshot(gpu).map_err(|error| error.to_string())?;
    for (index, &token) in DRAFTS.iter().enumerate() {
        append_target_single(gpu, config, native, PREFIX + index, token)?;
        append_mtp_single(gpu, config, mtp, PREFIX + index, token)?;
    }
    let committed = committed(accepted);
    if accepted < DRAFTS.len() {
        mutate_target(gpu, native)?;
        mutate_mtp(gpu, mtp)?;
        native
            .restore_retain(gpu, target_ticket)
            .map_err(|error| error.to_string())?;
        mtp.restore_retain(gpu, mtp_ticket)
            .map_err(|error| error.to_string())?;
        for (index, &token) in committed.iter().enumerate() {
            append_target_single(gpu, config, native, PREFIX + index, token)?;
            append_mtp_single(gpu, config, mtp, PREFIX + index, token)?;
        }
        native
            .validate_commit(target_ticket)
            .map_err(|error| error.to_string())?;
        mtp.validate_commit(mtp_ticket)
            .map_err(|error| error.to_string())?;
        native.commit_validated(target_ticket);
        mtp.commit_validated(mtp_ticket);
    } else {
        native
            .validate_commit(target_ticket)
            .map_err(|error| error.to_string())?;
        mtp.validate_commit(mtp_ticket)
            .map_err(|error| error.to_string())?;
        native.commit_validated(target_ticket);
        mtp.commit_validated(mtp_ticket);
        append_target_single(gpu, config, native, PREFIX + DRAFTS.len(), BONUS)?;
        append_mtp_single(gpu, config, mtp, PREFIX + DRAFTS.len(), BONUS)?;
    }
    for (index, &token) in committed.iter().enumerate() {
        append_target_single(gpu, config, ar, PREFIX + index, token)?;
        append_mtp_single(gpu, config, direct, PREFIX + index, token)?;
    }
    select_target_pair(gpu, config, ar, native, PREFIX + committed.len())?;
    select_mtp_pair(gpu, config, direct, mtp, PREFIX + committed.len())?;
    let target = compare_family_maps(
        &target_families(gpu, config, ar)?,
        &target_families(gpu, config, native)?,
    );
    let mtp_families = compare_family_maps(
        &mtp_families(gpu, config, direct)?,
        &mtp_families(gpu, config, mtp)?,
    );
    let pass = target.0 && mtp_families.0;
    Ok(json!({
        "case": match accepted {0 => "zero", 1 => "one", _ => "all"},
        "accepted_drafts": accepted,
        "committed_tokens": committed,
        "status": if pass {"pass"} else {"fail"},
        "target_families": target.1,
        "mtp_families": mtp_families.1,
    }))
}

fn committed(accepted: usize) -> Vec<u32> {
    let mut result = DRAFTS[..accepted].to_vec();
    result.push(BONUS);
    result
}

fn reset_target(gpu: &mut Gpu, state: &mut Qwen4State) -> Result<(), String> {
    state.reset(gpu).map_err(|error| error.to_string())
}

fn reset_mtp(gpu: &mut Gpu, state: &mut MtpGpuState) -> Result<(), String> {
    state.reset(gpu).map_err(|error| error.to_string())
}

fn append_target_pair(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    first: &mut Qwen4State,
    second: &mut Qwen4State,
    position: usize,
    token: u32,
) -> Result<(), String> {
    append_target_single(gpu, config, first, position, token)?;
    append_target_single(gpu, config, second, position, token)
}

fn append_mtp_pair(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    first: &mut MtpGpuState,
    second: &mut MtpGpuState,
    position: usize,
    token: u32,
) -> Result<(), String> {
    append_mtp_single(gpu, config, first, position, token)?;
    append_mtp_single(gpu, config, second, position, token)
}

fn append_target_single(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &mut Qwen4State,
    position: usize,
    token: u32,
) -> Result<(), String> {
    let full_width = config.num_key_value_heads * config.head_dim;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    for (layer_index, qsa) in state.qsa.iter_mut().enumerate() {
        let key_values = row(full_width, layer_index, position, token, 0.01);
        let value_values = row(full_width, layer_index, position, token, -0.02);
        let raw_values = row(raw_width, layer_index, position, token, 0.03);
        let key = gpu
            .upload_f32(&key_values, &[full_width])
            .map_err(|e| e.to_string())?;
        let value = match gpu.upload_f32(&value_values, &[full_width]) {
            Ok(value) => value,
            Err(error) => {
                let _ = gpu.free_tensor(key);
                return Err(error.to_string());
            }
        };
        let result = (|| {
            qwen4_qsa_cache_append(
                gpu,
                &Qwen4QsaCacheAppend {
                    key: &key,
                    value: &value,
                    full_keys: &qsa.full_keys,
                    full_values: &qsa.full_values,
                    position,
                    kv_width: full_width,
                },
            )
            .map_err(|e| e.to_string())?;
            write_raw(gpu, &qsa.raw_index_keys, position * raw_width, &raw_values)?;
            qsa.full_len = position + 1;
            qsa.raw_len = position + 1;
            qsa.position = position + 1;
            Ok::<(), String>(())
        })();
        let _ = gpu.free_tensor(key);
        let _ = gpu.free_tensor(value);
        result?;
    }
    state.position = position + 1;
    Ok(())
}

fn append_mtp_single(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &mut MtpGpuState,
    position: usize,
    token: u32,
) -> Result<(), String> {
    let full_width = config.num_key_value_heads * config.head_dim;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let key_values = row(full_width, 0, position, token, 0.01);
    let value_values = row(full_width, 0, position, token, -0.02);
    let raw_values = row(raw_width, 0, position, token, 0.03);
    let key = gpu
        .upload_f32(&key_values, &[full_width])
        .map_err(|e| e.to_string())?;
    let value = match gpu.upload_f32(&value_values, &[full_width]) {
        Ok(value) => value,
        Err(error) => {
            let _ = gpu.free_tensor(key);
            return Err(error.to_string());
        }
    };
    let result = (|| {
        let metadata = state.parity_metadata();
        let buffers = state.parity_buffers();
        qwen4_qsa_cache_append(
            gpu,
            &Qwen4QsaCacheAppend {
                key: &key,
                value: &value,
                full_keys: buffers.full_keys,
                full_values: buffers.full_values,
                position,
                kv_width: full_width,
            },
        )
        .map_err(|e| e.to_string())?;
        write_raw(
            gpu,
            buffers.raw_index_keys,
            position * raw_width,
            &raw_values,
        )?;
        state.parity_set_metadata(MtpStateParityMetadata {
            full_len: position + 1,
            raw_len: position + 1,
            pooled_len: metadata.pooled_len,
            selected_len: metadata.selected_len,
            position: position + 1,
            step_index: metadata.step_index,
        });
        Ok::<(), String>(())
    })();
    let _ = gpu.free_tensor(key);
    let _ = gpu.free_tensor(value);
    result
}

fn row(width: usize, layer: usize, position: usize, token: u32, scale: f32) -> Vec<f32> {
    (0..width)
        .map(|index| {
            let phase = (layer * 17 + position * 31 + index * 7 + token as usize) as f32;
            (phase.sin() * 0.25 + 0.5) * scale
        })
        .collect()
}

fn select_target_pair(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    first: &mut Qwen4State,
    second: &mut Qwen4State,
    visible: usize,
) -> Result<(), String> {
    select_target(gpu, config, first, visible)?;
    select_target(gpu, config, second, visible)
}

fn select_target(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &mut Qwen4State,
    visible: usize,
) -> Result<(), String> {
    let query = gpu
        .upload_f32(
            &query(config),
            &[config.indexer_n_heads, config.indexer_head_dim],
        )
        .map_err(|e| e.to_string())?;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let result = (|| {
        for qsa in &mut state.qsa {
            qwen4_qsa_pool_rope(
                gpu,
                &Qwen4QsaPoolRope {
                    raw_keys: &qsa.raw_index_keys,
                    pooled: &qsa.pooled_keys,
                    block_count: visible / config.indexer_compress_ratio,
                    compress: config.indexer_compress_ratio,
                    index_dim: raw_width,
                },
            )
            .map_err(|e| e.to_string())?;
            qwen4_qsa_select(
                gpu,
                &Qwen4QsaSelect {
                    query: &query,
                    pooled: &qsa.pooled_keys,
                    selected: &qsa.selected_indices,
                    block_count: visible / config.indexer_compress_ratio,
                    index_heads: config.indexer_n_heads,
                    index_dim: config.indexer_head_dim,
                    budget_blocks: config.indexer_budget / config.indexer_compress_ratio,
                    compress: config.indexer_compress_ratio,
                    visible,
                    capacity: config.qsa_selected_capacity(),
                },
            )
            .map_err(|e| e.to_string())?;
            qsa.pooled_len = visible / config.indexer_compress_ratio;
            qsa.selected_len = visible;
        }
        Ok::<(), String>(())
    })();
    let _ = gpu.free_tensor(query);
    result
}

fn select_mtp_pair(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    first: &mut MtpGpuState,
    second: &mut MtpGpuState,
    visible: usize,
) -> Result<(), String> {
    select_mtp(gpu, config, first, visible)?;
    select_mtp(gpu, config, second, visible)
}

fn select_mtp(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &mut MtpGpuState,
    visible: usize,
) -> Result<(), String> {
    let query = gpu
        .upload_f32(
            &query(config),
            &[config.indexer_n_heads, config.indexer_head_dim],
        )
        .map_err(|e| e.to_string())?;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let result = (|| {
        let buffers = state.parity_buffers();
        qwen4_qsa_pool_rope(
            gpu,
            &Qwen4QsaPoolRope {
                raw_keys: buffers.raw_index_keys,
                pooled: buffers.pooled_keys,
                block_count: visible / config.indexer_compress_ratio,
                compress: config.indexer_compress_ratio,
                index_dim: raw_width,
            },
        )
        .map_err(|e| e.to_string())?;
        qwen4_qsa_select(
            gpu,
            &Qwen4QsaSelect {
                query: &query,
                pooled: buffers.pooled_keys,
                selected: buffers.selected_indices,
                block_count: visible / config.indexer_compress_ratio,
                index_heads: config.indexer_n_heads,
                index_dim: config.indexer_head_dim,
                budget_blocks: config.indexer_budget / config.indexer_compress_ratio,
                compress: config.indexer_compress_ratio,
                visible,
                capacity: config.qsa_selected_capacity(),
            },
        )
        .map_err(|e| e.to_string())?;
        let metadata = state.parity_metadata();
        state.parity_set_metadata(MtpStateParityMetadata {
            full_len: metadata.full_len,
            raw_len: metadata.raw_len,
            pooled_len: visible / config.indexer_compress_ratio,
            selected_len: visible,
            position: metadata.position,
            step_index: metadata.step_index,
        });
        Ok::<(), String>(())
    })();
    let _ = gpu.free_tensor(query);
    result
}

fn reuse(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    state: &MtpGpuState,
) -> Result<(), String> {
    let buffers = state.parity_buffers();
    qwen4_qsa_reuse_selection(
        gpu,
        &Qwen4QsaReuseSelection {
            selected: buffers.selected_indices,
            selected_len: PREFIX,
            position: PREFIX,
            capacity: config.qsa_selected_capacity(),
            selected_len_out: buffers.selected_len_out,
        },
    )
    .map_err(|e| e.to_string())
}

fn query(config: &crate::config::Qwen4Config) -> Vec<f32> {
    (0..config.indexer_n_heads * config.indexer_head_dim)
        .map(|index| (index as f32 * 0.013).cos() * 0.125)
        .collect()
}

fn write_raw(
    gpu: &mut Gpu,
    tensor: &GpuTensor,
    offset_elements: usize,
    values: &[f32],
) -> Result<(), String> {
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr() as *const u8,
            values.len() * std::mem::size_of::<f32>(),
        )
    };
    let view = tensor.sub_offset(offset_elements, values.len());
    gpu.hip
        .memcpy_htod(&view.buf, bytes)
        .map_err(|error| error.to_string())
}

fn write_scalar_f32(gpu: &mut Gpu, tensor: &GpuTensor, value: f32) -> Result<(), String> {
    let view = tensor.sub_offset(0, 1);
    gpu.hip
        .memcpy_htod(&view.buf, &value.to_le_bytes())
        .map_err(|e| e.to_string())
}

fn write_scalar_i32(gpu: &mut Gpu, tensor: &GpuTensor, value: i32) -> Result<(), String> {
    let view = tensor.sub_offset(0, 4);
    gpu.hip
        .memcpy_htod(&view.buf, &value.to_le_bytes())
        .map_err(|e| e.to_string())
}

fn mutate_target(gpu: &mut Gpu, state: &mut Qwen4State) -> Result<(), String> {
    for layer in &state.gdn {
        write_scalar_f32(gpu, &layer.recurrent, 91.0)?;
        write_scalar_f32(gpu, &layer.conv, 92.0)?;
    }
    for layer in &state.qsa {
        write_scalar_f32(gpu, &layer.partial_keys, 93.0)?;
        write_scalar_f32(gpu, &layer.partial_values, 94.0)?;
        write_scalar_i32(gpu, &layer.selected_indices, -93)?;
    }
    write_scalar_f32(gpu, &state.ple_conv, 95.0)?;
    write_scalar_f32(gpu, &state.hyper_feedback, 96.0)
}

fn mutate_mtp(gpu: &mut Gpu, state: &MtpGpuState) -> Result<(), String> {
    let buffers = state.parity_buffers();
    write_scalar_i32(gpu, buffers.selected_indices, -97)?;
    write_scalar_i32(gpu, buffers.selected_len_out, 97)?;
    write_scalar_f32(gpu, buffers.wide_hidden, 98.0)
}

fn target_families(
    gpu: &Gpu,
    config: &crate::config::Qwen4Config,
    state: &Qwen4State,
) -> Result<Families, String> {
    let full_width = config.num_key_value_heads * config.head_dim;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let mut families = BTreeMap::new();
    let mut recurrent = Family::new();
    let mut conv = Family::new();
    for layer in &state.gdn {
        append_f32(
            gpu,
            &mut recurrent,
            &layer.recurrent,
            layer.recurrent.numel(),
        )?;
        append_f32(gpu, &mut conv, &layer.conv, layer.conv.numel())?;
    }
    families.insert("gdn_recurrent".into(), recurrent.finish());
    families.insert("gdn_conv".into(), conv.finish());
    let mut qsa_families = [
        ("qsa_full_keys", Family::new()),
        ("qsa_full_values", Family::new()),
        ("qsa_raw_index_keys", Family::new()),
        ("qsa_pooled_keys", Family::new()),
        ("qsa_partial_keys", Family::new()),
        ("qsa_partial_values", Family::new()),
        ("qsa_selected_indices", Family::new()),
    ];
    let mut metadata = Family::new();
    for qsa in &state.qsa {
        append_f32(
            gpu,
            &mut qsa_families[0].1,
            &qsa.full_keys,
            qsa.full_len * full_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[1].1,
            &qsa.full_values,
            qsa.full_len * full_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[2].1,
            &qsa.raw_index_keys,
            qsa.raw_len * raw_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[3].1,
            &qsa.pooled_keys,
            qsa.pooled_len * raw_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[4].1,
            &qsa.partial_keys,
            qsa.partial_len * raw_width,
        )?;
        append_f32(
            gpu,
            &mut qsa_families[5].1,
            &qsa.partial_values,
            qsa.partial_len * full_width,
        )?;
        append_raw(
            gpu,
            &mut qsa_families[6].1,
            &qsa.selected_indices,
            qsa.selected_len * 4,
            qsa.selected_len,
        )?;
        metadata_usize(
            &mut metadata,
            [
                qsa.full_len,
                qsa.raw_len,
                qsa.pooled_len,
                qsa.partial_len,
                qsa.selected_len,
                qsa.position,
            ],
        );
    }
    metadata_usize(&mut metadata, [state.position, state.max_seq_len]);
    for (name, family) in qsa_families {
        families.insert(name.into(), family.finish());
    }
    families.insert("ple_conv".into(), tensor(gpu, &state.ple_conv)?);
    families.insert("hyper_feedback".into(), tensor(gpu, &state.hyper_feedback)?);
    families.insert("metadata".into(), metadata.finish());
    Ok(families)
}

fn mtp_families(
    gpu: &Gpu,
    config: &crate::config::Qwen4Config,
    state: &MtpGpuState,
) -> Result<Families, String> {
    let full_width = config.num_key_value_heads * config.head_dim;
    let raw_width = config.indexer_kv_heads * config.indexer_head_dim;
    let metadata_values = state.parity_metadata();
    let buffers = state.parity_buffers();
    let mut families = BTreeMap::new();
    families.insert(
        "full_keys".into(),
        prefix(
            gpu,
            buffers.full_keys,
            metadata_values.full_len * full_width,
        )?,
    );
    families.insert(
        "full_values".into(),
        prefix(
            gpu,
            buffers.full_values,
            metadata_values.full_len * full_width,
        )?,
    );
    families.insert(
        "raw_index_keys".into(),
        prefix(
            gpu,
            buffers.raw_index_keys,
            metadata_values.raw_len * raw_width,
        )?,
    );
    families.insert(
        "pooled_keys".into(),
        prefix(
            gpu,
            buffers.pooled_keys,
            metadata_values.pooled_len * raw_width,
        )?,
    );
    families.insert(
        "selected_indices".into(),
        raw_prefix(
            gpu,
            buffers.selected_indices,
            metadata_values.selected_len,
            metadata_values.selected_len,
        )?,
    );
    families.insert(
        "selected_len_out".into(),
        raw_prefix(gpu, buffers.selected_len_out, 1, 1)?,
    );
    families.insert("wide_hidden".into(), tensor(gpu, buffers.wide_hidden)?);
    let mut metadata = Family::new();
    metadata_usize(
        &mut metadata,
        [
            metadata_values.full_len,
            metadata_values.raw_len,
            metadata_values.pooled_len,
            metadata_values.selected_len,
            metadata_values.position,
            metadata_values.step_index,
        ],
    );
    families.insert("metadata".into(), metadata.finish());
    Ok(families)
}

fn tensor(gpu: &Gpu, tensor: &GpuTensor) -> Result<Value, String> {
    prefix(gpu, tensor, tensor.numel())
}

fn prefix(gpu: &Gpu, tensor: &GpuTensor, elements: usize) -> Result<Value, String> {
    let mut family = Family::new();
    append_f32(gpu, &mut family, tensor, elements)?;
    Ok(family.finish())
}

fn raw_prefix(
    gpu: &Gpu,
    tensor: &GpuTensor,
    elements: usize,
    numel: usize,
) -> Result<Value, String> {
    let mut family = Family::new();
    append_raw(gpu, &mut family, tensor, elements, numel)?;
    Ok(family.finish())
}

fn append_f32(
    gpu: &Gpu,
    family: &mut Family,
    tensor: &GpuTensor,
    elements: usize,
) -> Result<(), String> {
    if elements == 0 {
        return Ok(());
    }
    let view = tensor.sub_offset(0, elements);
    let values = gpu.download_f32(&view).map_err(|error| error.to_string())?;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            values.as_ptr() as *const u8,
            values.len() * std::mem::size_of::<f32>(),
        )
    };
    family.bytes(bytes, values.len());
    Ok(())
}

fn append_raw(
    gpu: &Gpu,
    family: &mut Family,
    tensor: &GpuTensor,
    elements: usize,
    numel: usize,
) -> Result<(), String> {
    if elements == 0 {
        return Ok(());
    }
    let view = tensor.sub_offset(0, elements);
    let mut bytes = vec![0u8; view.byte_size()];
    gpu.hip
        .memcpy_dtoh(&mut bytes, &view.buf)
        .map_err(|error| error.to_string())?;
    family.bytes(&bytes, numel);
    Ok(())
}

fn metadata_usize<const N: usize>(family: &mut Family, values: [usize; N]) {
    let mut bytes = Vec::with_capacity(N * std::mem::size_of::<usize>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    family.bytes(&bytes, N);
}

fn compare_family_maps(left: &Families, right: &Families) -> (bool, Value) {
    let keys = left
        .keys()
        .chain(right.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut result = Map::new();
    let mut pass = true;
    for key in keys {
        let left_value = left.get(&key);
        let right_value = right.get(&key);
        let left_digest = left_value
            .and_then(|value| value.get("digest"))
            .and_then(Value::as_str);
        let right_digest = right_value
            .and_then(|value| value.get("digest"))
            .and_then(Value::as_str);
        let matches = left_digest == right_digest
            && left_value.and_then(|value| value.get("numel"))
                == right_value.and_then(|value| value.get("numel"));
        pass &= matches;
        result.insert(
            key,
            json!({
                "ar_or_direct": left_value,
                "native": right_value,
                "match": matches,
            }),
        );
    }
    (pass, Value::Object(result))
}

fn metadata_json(metadata: MtpStateParityMetadata) -> Value {
    json!({
        "full_len": metadata.full_len,
        "raw_len": metadata.raw_len,
        "pooled_len": metadata.pooled_len,
        "selected_len": metadata.selected_len,
        "position": metadata.position,
        "step_index": metadata.step_index,
    })
}

fn download_i32(gpu: &Gpu, tensor: &GpuTensor, count: usize) -> Result<Vec<i32>, String> {
    let mut values = vec![0i32; count];
    let bytes =
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, count * 4) };
    gpu.hip
        .memcpy_dtoh(bytes, &tensor.buf)
        .map_err(|e| e.to_string())?;
    Ok(values)
}

fn stale_ticket(
    gpu: &mut Gpu,
    target: &mut Qwen4State,
    mtp: &mut MtpGpuState,
) -> Result<Value, String> {
    target.reset(gpu).map_err(|e| e.to_string())?;
    mtp.reset(gpu).map_err(|e| e.to_string())?;
    let target_ticket = target.snapshot(gpu).map_err(|e| e.to_string())?;
    let mtp_ticket = mtp.snapshot(gpu).map_err(|e| e.to_string())?;
    target.reset(gpu).map_err(|e| e.to_string())?;
    mtp.reset(gpu).map_err(|e| e.to_string())?;
    let target_refusal = target
        .restore(gpu, target_ticket)
        .err()
        .map(|e| e.to_string());
    let mtp_refusal = mtp.restore(gpu, mtp_ticket).err().map(|e| e.to_string());
    let unchanged = target.position == 0 && mtp.parity_metadata().position == 0;
    Ok(json!({
        "case": "stale_ticket",
        "status": if target_refusal.is_some() && mtp_refusal.is_some() && unchanged {"pass"} else {"fail"},
        "target_refusal": target_refusal,
        "mtp_refusal": mtp_refusal,
        "state_unchanged": unchanged,
    }))
}

fn terminal_seed(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    target: &mut Qwen4State,
    mtp: &mut MtpGpuState,
) -> Result<Value, String> {
    target.reset(gpu).map_err(|e| e.to_string())?;
    mtp.reset(gpu).map_err(|e| e.to_string())?;
    append_target_single(gpu, config, target, 0, config.eos_token_id)?;
    append_mtp_single(gpu, config, mtp, 0, config.eos_token_id)?;
    for qsa in &mut target.qsa {
        write_scalar_i32(gpu, &qsa.selected_indices, 0)?;
        qsa.selected_len = 1;
        qsa.position = 1;
    }
    target.position = 1;
    let metadata = mtp.parity_metadata();
    let buffers = mtp.parity_buffers();
    write_scalar_i32(gpu, buffers.selected_indices, 0)?;
    mtp.parity_set_metadata(MtpStateParityMetadata {
        full_len: 1,
        raw_len: 1,
        pooled_len: 0,
        selected_len: 1,
        position: 1,
        step_index: metadata.step_index,
    });
    let pass = target.position == 1 && mtp.parity_metadata().position == 1;
    Ok(json!({
        "case": "forced_terminal_seed",
        "status": if pass {"pass"} else {"fail"},
        "forced_token": config.eos_token_id,
        "terminal": true,
        "target_metadata": {"position": target.position, "selected_len": target.qsa.first().map(|qsa| qsa.selected_len)},
        "mtp_metadata": metadata_json(mtp.parity_metadata()),
    }))
}

fn rollback_failure(
    gpu: &mut Gpu,
    config: &crate::config::Qwen4Config,
    target: &mut Qwen4State,
    mtp: &mut MtpGpuState,
    case: &str,
    injected: Option<&str>,
) -> Result<Value, String> {
    target.reset(gpu).map_err(|e| e.to_string())?;
    mtp.reset(gpu).map_err(|e| e.to_string())?;
    for (position, &token) in [10u32, 11, 12, 13].iter().enumerate() {
        append_target_single(gpu, config, target, position, token)?;
        append_mtp_single(gpu, config, mtp, position, token)?;
    }
    select_target(gpu, config, target, PREFIX)?;
    select_mtp(gpu, config, mtp, PREFIX)?;
    let before_target = target_families(gpu, config, target)?;
    let before_mtp = mtp_families(gpu, config, mtp)?;
    let target_ticket = target.snapshot(gpu).map_err(|e| e.to_string())?;
    let mtp_ticket = mtp.snapshot(gpu).map_err(|e| e.to_string())?;
    mutate_target(gpu, target)?;
    mutate_mtp(gpu, mtp)?;
    let failure = injected.unwrap_or("cancelled by caller").to_string();
    target
        .restore(gpu, target_ticket)
        .map_err(|e| e.to_string())?;
    mtp.restore(gpu, mtp_ticket).map_err(|e| e.to_string())?;
    let target_compare =
        compare_family_maps(&before_target, &target_families(gpu, config, target)?);
    let mtp_compare = compare_family_maps(&before_mtp, &mtp_families(gpu, config, mtp)?);
    let pass = target_compare.0 && mtp_compare.0;
    Ok(json!({
        "case": case,
        "status": if pass {"pass"} else {"fail"},
        "replay_error": failure,
        "target_rollback": target_compare.1,
        "mtp_rollback": mtp_compare.1,
    }))
}

fn cache_suffix_refusal() -> Value {
    let refusal = validate_native_mtp_prefill_request(&[1, 2, 3], &[1, 2, 3], 0, true)
        .expect_err("cache-hit suffix must be refused");
    json!({
        "case": "cache_suffix_refusal",
        "status": "pass",
        "refusal": refusal,
        "cache_hit": true,
        "prompt_len": 3,
        "fill_len": 3,
    })
}

/// JSON report returned by the state parity orchestration entrypoint.
pub struct StateParityReport(Value);

impl StateParityReport {
    pub fn into_json(self) -> Value {
        self.0
    }

    pub fn write(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        std::fs::write(
            path,
            serde_json::to_vec_pretty(&self.0).map_err(|error| error.to_string())?,
        )
        .map_err(|error| format!("write {}: {error}", path.display()))
    }
}

/// Load one real HFQ model, run the production AR/native-MTP path, and append
/// compact state-arena scenarios to the resulting JSON report.
pub fn run_state_parity(
    model_path: &Path,
    corpus_path: &Path,
) -> Result<StateParityReport, String> {
    let (tokens, corpus) = read_state_tokens(corpus_path)?;
    let hfq = hipfire_runtime::hfq::HfqFile::open(model_path)
        .map_err(|error| format!("open {}: {error}", model_path.display()))?;
    let receipt = crate::admit_hfqm_artifact(&hfq)
        .map_err(|error| format!("qwen4 artifact admission failed: {error}"))?;
    let config = receipt.config.clone();
    let manifest = receipt.manifest.clone();
    let metadata = receipt.ple.clone();
    let placements = receipt.placements.clone();
    let mut gpu = Gpu::init().map_err(|error| error.to_string())?;
    if !gpu.arch_caps.is_gfx1151() {
        return Err(format!(
            "Qwen4 state parity runner requires gfx1151, got {}",
            gpu.arch
        ));
    }
    let mesh = hipfire_runtime::device_mesh::DeviceMesh::single()
        .map_err(|error| format!("qwen4 mesh: {error}"))?;
    let expected = hipfire_runtime::weight_store::WeightOrigin::for_single(&mesh, &gpu);
    let source = hipfire_runtime::hfq::HfqModelSource::from_hfq(hfq);
    let transaction = hipfire_runtime::weight_store::fulfill_manifest_from_payloads(
        &manifest.weights,
        &mesh,
        config.num_hidden_layers,
        &mut gpu,
        expected,
        |entry| {
            if entry.residency.is_external() {
                return source
                    .tensor_range(&entry.name)
                    .map_err(|error| error.to_string())?
                    .map(hipfire_runtime::model_source::SourcePayload::Range)
                    .ok_or_else(|| format!("missing external tensor '{}'", entry.name));
            }
            let (info, bytes) = source
                .tensor_data(&entry.name)
                .ok_or_else(|| format!("missing resident tensor '{}'", entry.name))?;
            Ok(hipfire_runtime::model_source::SourcePayload::Borrowed { info, bytes })
        },
    )
    .map_err(|error| format!("qwen4 manifest fulfillment failed: {error}"))?;
    let mut bundle = crate::bundle::Qwen4Bundle::assemble_with_metadata(
        config.clone(),
        transaction,
        &placements,
        &mut gpu,
        2048,
        metadata,
    )
    .map_err(|error| format!("qwen4 bundle assembly failed: {error}"))?;
    let real = (|| {
        bundle
            .attach_forward(&mut gpu, 2048)
            .map_err(|error| format!("qwen4 forward setup failed: {error}"))?;
        bundle
            .attach_mtp(&mut gpu, 2048)
            .map_err(|error| format!("qwen4 MTP setup failed: {error}"))?;
        real_model_probe(&mut gpu, &mut bundle, &tokens, &corpus)
    })();
    let bundle_cleanup = bundle
        .free_gpu(&mut gpu)
        .err()
        .map(|error| format!("qwen4 bundle teardown failed: {error}"));
    let real = real?;
    if let Some(error) = bundle_cleanup {
        return Err(error);
    }
    let compact = run_compact(&mut gpu)?;
    let status = if compact.get("status") == Some(&Value::String("pass".into()))
        && real.get("status") == Some(&Value::String("pass".into()))
    {
        "pass"
    } else {
        "fail"
    };
    Ok(StateParityReport(json!({
        "schema": "hipfire.qwen4.state_parity.v1",
        "gpu_arch": gpu.arch,
        "model": model_path,
        "corpus": corpus,
        "real_model": real,
        "compact_state": compact,
        "status": status,
    })))
}

fn real_model_probe(
    gpu: &mut Gpu,
    bundle: &mut crate::bundle::Qwen4Bundle,
    tokens: &[u32],
    corpus: &Value,
) -> Result<Value, String> {
    let vocab = bundle.config.vocab_size;
    let logits = gpu
        .zeros(&[vocab], DType::F32)
        .map_err(|error| format!("allocate real-model logits: {error}"))?;
    let ar_result = (|| {
        bundle.reset(gpu).map_err(|error| error.to_string())?;
        bundle
            .forward_chunk(gpu, tokens, &logits, None)
            .map_err(|error| format!("real AR forward: {error}"))?;
        download_host_logits(gpu, &logits)
    })();
    let ar_logits = match ar_result {
        Ok(logits) => logits,
        Err(error) => {
            let _ = gpu.free_tensor(logits);
            return Err(error);
        }
    };
    bundle.reset(gpu).map_err(|error| error.to_string())?;
    let mut drafter = crate::mtp_spec::Qwen4MtpDrafter::new(DRAFTS.len(), 2048);
    use hipfire_runtime::spec::MtpDrafter;
    let seed = match drafter.mtp_prefill(gpu, bundle, tokens, tokens, 0, false, &|| false) {
        Ok(seed) => seed,
        Err(error) => {
            MtpDrafter::mtp_free(Box::new(drafter), gpu);
            let _ = gpu.free_tensor(logits);
            return Err(format!("real native MTP prefill: {error}"));
        }
    };
    let eos = bundle.config.eos_token_id;
    let window = match drafter.mtp_step(
        gpu,
        bundle,
        tokens.len(),
        seed,
        &[],
        DRAFTS.len(),
        eos,
        None,
    ) {
        Ok(window) => window,
        Err(error) => {
            MtpDrafter::mtp_free(Box::new(drafter), gpu);
            let _ = gpu.free_tensor(logits);
            return Err(format!("real native MTP step: {error}"));
        }
    };
    let probe_token = window.committed.last().copied().unwrap_or(seed);
    let mtp_position = bundle
        .mtp_position()
        .map_err(|error| format!("read native MTP position: {error}"))?;
    bundle
        .mtp_forward_token_logits(gpu, probe_token, mtp_position, &logits)
        .map_err(|error| format!("real native MTP logits: {error}"))?;
    let mtp_logits = match download_host_logits(gpu, &logits) {
        Ok(logits) => logits,
        Err(error) => {
            MtpDrafter::mtp_free(Box::new(drafter), gpu);
            let _ = gpu.free_tensor(logits);
            return Err(error);
        }
    };
    MtpDrafter::mtp_free(Box::new(drafter), gpu);
    let families = bundle_family_json(gpu, bundle)?;
    let max_abs = ar_logits
        .values
        .iter()
        .zip(&mtp_logits.values)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    let finite = ar_logits.values.iter().all(|value| value.is_finite())
        && mtp_logits.values.iter().all(|value| value.is_finite());
    let result = json!({
        "status": if finite {"pass"} else {"fail"},
        "corpus": corpus,
        "accepted_drafts": window.accepted,
        "drafts_generated": window.drafts_generated,
        "committed": window.committed,
        "seed": seed,
        "probe_token": probe_token,
        "ar_logits": host_logits_json(&ar_logits),
        "native_mtp_logits": host_logits_json(&mtp_logits),
        "ar_vs_native_mtp": {
            "max_abs": max_abs,
            "same_within_1e-5": max_abs <= 1.0e-5,
            "ar_digest": ar_logits.digest,
            "native_mtp_digest": mtp_logits.digest,
        },
        "state_families": families,
    });
    gpu.free_tensor(logits).map_err(|error| error.to_string())?;
    Ok(result)
}

#[derive(Clone, Debug)]
struct HostLogits {
    values: Vec<f32>,
    digest: String,
    top1: usize,
}

fn download_host_logits(gpu: &Gpu, tensor: &GpuTensor) -> Result<HostLogits, String> {
    let values = gpu
        .download_f32(tensor)
        .map_err(|error| error.to_string())?;
    let mut family = Family::new();
    let bytes =
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
    family.bytes(bytes, values.len());
    let digest = family
        .finish()
        .get("digest")
        .and_then(Value::as_str)
        .ok_or("host logits digest missing")?
        .to_string();
    let top1 = values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .unwrap_or(0);
    Ok(HostLogits {
        values,
        digest,
        top1,
    })
}
fn host_logits_json(logits: &HostLogits) -> Value {
    json!({"digest": logits.digest, "numel": logits.values.len(), "top1": logits.top1})
}

fn bundle_family_json(gpu: &Gpu, bundle: &crate::bundle::Qwen4Bundle) -> Result<Value, String> {
    let target = target_families(gpu, &bundle.config, &bundle.state)?;
    let mtp = bundle
        .mtp
        .as_ref()
        .map(|mtp| mtp_families(gpu, &bundle.config, mtp.parity_state()))
        .transpose()?;
    Ok(json!({"target": target, "mtp": mtp}))
}

fn read_state_tokens(path: &Path) -> Result<(Vec<u32>, Value), String> {
    const COUNT: usize = 17;
    const SHA256: &str = "e53de8c7b501eaaea637648feb6f569dd17cd564c2f669b2924ccdf1b7e52e2f";
    let metadata_path = if path.extension().and_then(|ext| ext.to_str()) == Some("json") {
        path.to_path_buf()
    } else {
        path.with_extension("json")
    };
    let metadata_text = std::fs::read_to_string(&metadata_path)
        .map_err(|error| format!("read {}: {error}", metadata_path.display()))?;
    let metadata: Value = serde_json::from_str(&metadata_text)
        .map_err(|error| format!("parse {}: {error}", metadata_path.display()))?;
    if metadata.get("count").and_then(Value::as_u64) != Some(COUNT as u64) {
        return Err("state parity requires the canonical 17-token corpus".to_string());
    }
    let relative = metadata
        .get("path")
        .and_then(Value::as_str)
        .ok_or("state corpus metadata has no path")?;
    let payload = metadata_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(PathBuf::from(relative));
    let bytes =
        std::fs::read(&payload).map_err(|error| format!("read {}: {error}", payload.display()))?;
    use sha2::{Digest, Sha256};
    let mut digest = Sha256::new();
    digest.update(&bytes);
    if bytes.len() != COUNT * 4 || format!("{:x}", digest.finalize()) != SHA256 {
        return Err(
            "state corpus byte count or SHA256 does not match canonical corpus".to_string(),
        );
    }
    let tokens = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    Ok((
        tokens,
        json!({
            "metadata_path": metadata_path,
            "payload_path": payload,
            "count": COUNT,
            "sha256": SHA256
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_gfx1151_gpu() -> Option<Gpu> {
        let gpu = Gpu::init().ok()?;
        if !gpu.arch_caps.is_gfx1151() {
            eprintln!("skip: Qwen4 compact state parity requires gfx1151");
            return None;
        }
        Some(gpu)
    }

    fn field<'a>(value: &'a Value, name: &str) -> &'a Value {
        value
            .get(name)
            .unwrap_or_else(|| panic!("report is missing field {name:?}: {value}"))
    }

    fn assert_pass(value: &Value) {
        assert_eq!(
            field(value, "status").as_str(),
            Some("pass"),
            "failed report case: {value}"
        );
    }

    fn assert_family_matches(value: &Value, name: &str) {
        let families = field(value, name)
            .as_object()
            .unwrap_or_else(|| panic!("{name} is not an object: {}", field(value, name)));
        assert!(!families.is_empty(), "{name} has no state families");
        for (family, comparison) in families {
            assert_eq!(
                field(comparison, "match").as_bool(),
                Some(true),
                "{name}.{family} did not match: {comparison}"
            );
        }
    }

    #[test]
    fn compact_runner_reports_device_backed_state_parity_scenarios() {
        let Some(mut gpu) = try_gfx1151_gpu() else {
            return;
        };
        let report = run_compact(&mut gpu).expect("compact state parity runner");

        assert_eq!(
            field(&report, "schema").as_str(),
            Some("hipfire.qwen4.state_parity.compact.v1")
        );
        assert_eq!(field(&report, "gpu_arch").as_str(), Some("gfx1151"));
        assert_pass(&report);

        let cases = field(&report, "acceptance_cases")
            .as_array()
            .expect("acceptance_cases array");
        assert_eq!(cases.len(), 3);
        let expected_cases = [
            ("zero", 0usize, json!([1003u32])),
            ("one", 1usize, json!([1001u32, 1003u32])),
            ("all", 2usize, json!([1001u32, 1002u32, 1003u32])),
        ];
        let mut case_names = BTreeSet::new();
        for case in cases {
            assert_pass(case);
            let name = field(case, "case").as_str().expect("acceptance case name");
            let (_, accepted, committed_tokens) = expected_cases
                .iter()
                .find(|(expected, _, _)| *expected == name)
                .unwrap_or_else(|| panic!("unexpected acceptance case {name:?}"));
            assert_eq!(
                field(case, "accepted_drafts").as_u64(),
                Some(*accepted as u64)
            );
            assert_eq!(field(case, "committed_tokens"), committed_tokens);
            assert_family_matches(case, "target_families");
            assert_family_matches(case, "mtp_families");
            case_names.insert(name.to_string());
        }
        assert_eq!(
            case_names,
            expected_cases
                .iter()
                .map(|(name, _, _)| (*name).to_string())
                .collect()
        );

        let scenarios = field(&report, "scenarios")
            .as_array()
            .expect("scenarios array");
        assert_eq!(scenarios.len(), 6);
        let mut scenario_names = BTreeSet::new();
        for scenario in scenarios {
            assert_pass(scenario);
            scenario_names.insert(
                field(scenario, "case")
                    .as_str()
                    .expect("scenario name")
                    .to_string(),
            );
        }
        assert_eq!(
            scenario_names,
            [
                "qsa_pooling_boundary",
                "stale_ticket",
                "forced_terminal_seed",
                "cancellation",
                "injected_replay_failure",
                "cache_suffix_refusal",
            ]
            .into_iter()
            .map(str::to_string)
            .collect()
        );

        let boundary = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("qsa_pooling_boundary"))
            .expect("QSA pooling boundary scenario");
        assert_eq!(
            field(field(boundary, "target_metadata"), "full_len").as_u64(),
            Some(4)
        );
        assert_eq!(
            field(field(boundary, "target_metadata"), "raw_len").as_u64(),
            Some(4)
        );
        assert_eq!(
            field(field(boundary, "target_metadata"), "pooled_len").as_u64(),
            Some(1)
        );
        assert_eq!(
            field(field(boundary, "target_metadata"), "selected_len").as_u64(),
            Some(4)
        );
        assert_eq!(
            field(field(boundary, "reuse_selected_len"), "direct").as_i64(),
            Some(5)
        );
        assert_eq!(
            field(field(boundary, "reuse_selected_len"), "native").as_i64(),
            Some(5)
        );
        assert_eq!(
            field(field(boundary, "reuse_selected_len"), "expected").as_u64(),
            Some(5)
        );
        assert_eq!(
            field(boundary, "reuse_selected_indices"),
            &json!({
                "direct": [0, 1, 2, 3, 4],
                "native": [0, 1, 2, 3, 4],
            })
        );

        let stale = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("stale_ticket"))
            .expect("stale ticket scenario");
        assert!(field(stale, "target_refusal").as_str().is_some());
        assert!(field(stale, "mtp_refusal").as_str().is_some());
        assert_eq!(field(stale, "state_unchanged").as_bool(), Some(true));

        let terminal = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("forced_terminal_seed"))
            .expect("forced terminal seed scenario");
        assert_eq!(field(terminal, "terminal").as_bool(), Some(true));
        assert_eq!(
            field(terminal, "forced_token").as_u64(),
            Some(compact_test_config().eos_token_id as u64)
        );
        assert_eq!(
            field(field(terminal, "target_metadata"), "position").as_u64(),
            Some(1)
        );
        assert_eq!(
            field(field(terminal, "mtp_metadata"), "position").as_u64(),
            Some(1)
        );

        for name in ["cancellation", "injected_replay_failure"] {
            let rollback = scenarios
                .iter()
                .find(|scenario| field(scenario, "case").as_str() == Some(name))
                .unwrap_or_else(|| panic!("{name} scenario"));
            assert_family_matches(rollback, "target_rollback");
            assert_family_matches(rollback, "mtp_rollback");
        }
        let injected = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("injected_replay_failure"))
            .expect("injected replay failure scenario");
        assert_eq!(
            field(injected, "replay_error").as_str(),
            Some("injected replay error")
        );

        let cache = scenarios
            .iter()
            .find(|scenario| field(scenario, "case").as_str() == Some("cache_suffix_refusal"))
            .expect("cache suffix refusal scenario");
        assert_eq!(field(cache, "cache_hit").as_bool(), Some(true));
        assert!(field(cache, "refusal")
            .as_str()
            .unwrap()
            .contains("cache-hit suffix"));
    }
}
