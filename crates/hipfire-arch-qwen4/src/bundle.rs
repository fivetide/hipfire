// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Qwen4 model ownership and teardown boundary.
//!
//! A published bundle owns every model resource: the bounded SSD PLE reader,
//! mutable GPU state, finalized resident weights, and the attached canonical
//! load transaction/census (including external PLE descriptors).  No loader
//! local may outlive publication as a second owner.

use crate::config::Qwen4Config;
use crate::ple::PleHashMetadata;
use crate::ple_rows::{PleRows, PleRowsError};
use crate::state::{Qwen4State, Qwen4StateSnapshot, StateError};
use crate::weights::{
    ple_valid_rows_for_shard, ExternalRowsRef, Qwen4Manifest, Qwen4Placement, Qwen4Weights,
    WeightError, PLE_ROW_WIDTH, PLE_SHARD_COUNT, PLE_SHARD_ROWS,
};
use hipfire_runtime::model_source::SourceRangeDescriptor;
use hipfire_runtime::weight_manifest::{WeightEntry, WeightResidency};
use hipfire_runtime::weight_store::{WeightLoadTransaction, WeightStoreError};
use rdna_compute::Gpu;
use std::fmt;
use std::time::Duration;

const PLE_UNLOAD_TIMEOUT: Duration = Duration::from_secs(5);

/// Architecture-private owner for the fulfilled manifest census.
///
/// Assembly takes every resident handle into [`Qwen4Weights`], so rollback at
/// unload releases no duplicate device allocations.  The transaction still
/// owns the immutable projections, aliases, and external PLE descriptors and
/// is drained only after the reader, mutable state, and resident weights.
pub(crate) struct AttachedWeightStore {
    transaction: WeightLoadTransaction,
}

impl AttachedWeightStore {
    fn new(transaction: WeightLoadTransaction) -> Self {
        Self { transaction }
    }

    fn drain(self, gpu: &mut Gpu) -> hip_bridge::HipResult<()> {
        self.transaction.rollback(gpu)
    }

    fn external_descriptor(
        &self,
        name: &str,
        layer: Option<usize>,
        device: usize,
    ) -> Option<&SourceRangeDescriptor> {
        self.transaction.external_descriptor(name, layer, device)
    }

    pub(crate) fn external_rows_len(&self) -> usize {
        self.transaction.external_rows_len()
    }

    pub(crate) fn inventory_len(&self) -> usize {
        self.transaction.inventory_len()
    }

    pub(crate) fn origin(&self) -> Option<hipfire_runtime::weight_store::WeightOrigin> {
        self.transaction.origin()
    }
}

/// Published Qwen4 architecture owner.
pub struct Qwen4Bundle {
    pub config: Qwen4Config,
    pub weights: Qwen4Weights,
    pub state: Qwen4State,
    /// Bounded model-owned PLE reader/cache.  It must quiesce before source
    /// descriptors and the attached transaction are dropped.
    pub(crate) ple_rows: PleRows,
    /// Canonical load census and external descriptors.  This is deliberately
    /// not left in the loader or carrier after publication.
    pub(crate) weight_store: AttachedWeightStore,
}

impl Qwen4Bundle {
    /// Assemble a complete Single bundle using metadata parsed from the
    /// artifact's canonical `qwen4_ple` object.
    pub fn assemble(
        config: Qwen4Config,
        transaction: WeightLoadTransaction,
        placements: &[Qwen4Placement],
        gpu: &mut Gpu,
        max_seq_len: usize,
        metadata: PleHashMetadata,
    ) -> Result<Self, BundleError> {
        Self::assemble_with_metadata(config, transaction, placements, gpu, max_seq_len, metadata)
    }

    /// Assemble with validated metadata read from the artifact's exact I64
    /// arrays.  The transaction is consumed so no load-side owner can remain
    /// live after this method publishes the bundle.
    pub fn assemble_with_metadata(
        config: Qwen4Config,
        mut transaction: WeightLoadTransaction,
        placements: &[Qwen4Placement],
        gpu: &mut Gpu,
        max_seq_len: usize,
        metadata: PleHashMetadata,
    ) -> Result<Self, BundleError> {
        let weights = match Qwen4Weights::assemble(&mut transaction, &config, placements) {
            Ok(weights) => weights,
            Err(error) => {
                return Err(cleanup_transaction(
                    BundleError::Weights(error),
                    transaction.rollback(gpu),
                ));
            }
        };
        let descriptors = match ple_descriptors(&transaction, &weights.manifest) {
            Ok(descriptors) => descriptors,
            Err(error) => {
                let weight_result = weights.free_gpu(gpu);
                let cleanup = transaction.rollback(gpu);
                return Err(cleanup_bundle_failure(error, weight_result, cleanup));
            }
        };
        let ple_rows = match PleRows::new(descriptors, metadata) {
            Ok(rows) => rows,
            Err(error) => {
                let weight_result = weights.free_gpu(gpu);
                let cleanup = transaction.rollback(gpu);
                return Err(cleanup_bundle_failure(
                    BundleError::PleRows(error),
                    weight_result,
                    cleanup,
                ));
            }
        };
        let state = match Qwen4State::new(gpu, &config, max_seq_len) {
            Ok(state) => state,
            Err(error) => {
                // `unload` consumes the reader and joins its worker even on a
                // quiesce error, so source descriptors cannot outlive failure.
                let _ = ple_rows.unload(PLE_UNLOAD_TIMEOUT);
                let weight_result = weights.free_gpu(gpu);
                let cleanup = transaction.rollback(gpu);
                return Err(cleanup_bundle_failure(
                    BundleError::State(error),
                    weight_result,
                    cleanup,
                ));
            }
        };
        Ok(Self {
            config,
            weights,
            state,
            ple_rows,
            weight_store: AttachedWeightStore::new(transaction),
        })
    }

    pub fn manifest(&self) -> &Qwen4Manifest {
        &self.weights.manifest
    }

    pub fn external_descriptor(
        &self,
        placement: &Qwen4Placement,
    ) -> Option<&SourceRangeDescriptor> {
        self.weight_store
            .external_descriptor(&placement.name, placement.layer, placement.device)
    }

    /// Return all numerically ordered PLE shard descriptors for a reader that
    /// has not yet been attached.  The bundle's own reader is normally the
    /// consumer; this accessor is useful for diagnostics without exposing the
    /// transaction itself.
    pub fn ple_descriptors(&self) -> Vec<SourceRangeDescriptor> {
        let mut entries = self.manifest().external_entries().collect::<Vec<_>>();
        entries.sort_by_key(|entry| {
            ple_shard_index(&entry.name).unwrap_or(PLE_SHARD_COUNT)
        });
        entries
            .into_iter()
            .filter_map(|entry| {
                self.external_descriptor(&Qwen4Placement {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                })
                .cloned()
            })
            .collect()
    }


    pub fn ple_rows(&self) -> &PleRows {
        &self.ple_rows
    }

    pub fn ple_rows_mut(&mut self) -> &mut PleRows {
        &mut self.ple_rows
    }

    pub fn begin_ple_epoch(&self, epoch: u64) {
        self.ple_rows.begin_epoch(epoch);
    }

    pub fn attached_origin(&self) -> Option<hipfire_runtime::weight_store::WeightOrigin> {
        self.weight_store.origin()
    }

    pub fn attached_inventory_len(&self) -> usize {
        self.weight_store.inventory_len()
    }

    pub fn attached_external_rows_len(&self) -> usize {
        self.weight_store.external_rows_len()
    }

    pub fn reset(&mut self, gpu: &mut Gpu) -> Result<(), BundleError> {
        self.state.reset(gpu).map_err(BundleError::State)
    }

    pub fn snapshot(&self, gpu: &mut Gpu) -> Result<Qwen4StateSnapshot, BundleError> {
        self.state.snapshot(gpu).map_err(BundleError::State)
    }

    pub fn restore(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
    ) -> Result<(), BundleError> {
        self.state
            .restore(gpu, snapshot)
            .map_err(BundleError::State)
    }

    pub fn commit(
        &mut self,
        gpu: &mut Gpu,
        snapshot: Qwen4StateSnapshot,
    ) -> Result<(), BundleError> {
        self.state.commit(snapshot, gpu).map_err(BundleError::State)
    }

    /// Teardown is deliberately ordered: stop/quiesce PLE reads and release
    /// leases/page-cache resources, free mutable GPU state, free finalized
    /// resident weights, then drain the attached transaction/census.
    pub fn free_gpu(self, gpu: &mut Gpu) -> Result<(), BundleError> {
        let Qwen4Bundle {
            weights,
            state,
            ple_rows,
            weight_store,
            ..
        } = self;
        let ple_result = ple_rows
            .unload(PLE_UNLOAD_TIMEOUT)
            .map(|_| ())
            .map_err(BundleError::PleRows);
        let state_result = state.free_gpu(gpu).map_err(BundleError::State);
        let weight_result = weights.free_gpu(gpu).map_err(BundleError::Hip);
        let store_result = weight_store.drain(gpu).map_err(BundleError::Hip);
        first_bundle_error([ple_result, state_result, weight_result, store_result])
    }
}
const PLE_SHARD_NAME_PREFIX: &str =
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_";

fn ple_shard_index(name: &str) -> Result<usize, WeightError> {
    let suffix = name
        .strip_prefix(PLE_SHARD_NAME_PREFIX)
        .and_then(|name| name.strip_suffix(".weight"))
        .ok_or_else(|| {
            WeightError::DescriptorMismatch(format!("invalid PLE shard name '{name}'"))
        })?;
    if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(WeightError::DescriptorMismatch(format!(
            "invalid PLE shard name '{name}'"
        )));
    }
    let index = suffix.parse::<usize>().map_err(|_| {
        WeightError::DescriptorMismatch(format!("invalid PLE shard name '{name}'"))
    })?;
    if index >= PLE_SHARD_COUNT || index.to_string() != suffix {
        return Err(WeightError::DescriptorMismatch(format!(
            "PLE shard index {index} is outside canonical range in '{name}'"
        )));
    }
    Ok(index)
}

fn ordered_ple_entries<'a>(
    manifest: &'a Qwen4Manifest,
) -> Result<Vec<&'a WeightEntry>, WeightError> {
    let entries = manifest.external_entries().collect::<Vec<_>>();
    if entries.len() != PLE_SHARD_COUNT {
        return Err(WeightError::PleShardCount {
            expected: PLE_SHARD_COUNT,
            actual: entries.len(),
        });
    }
    let mut ordered = vec![None; PLE_SHARD_COUNT];
    for entry in entries {
        let index = ple_shard_index(&entry.name)?;
        if entry.layer != Some(1) || entry.logical_shape.as_slice() != [PLE_SHARD_ROWS, PLE_ROW_WIDTH]
        {
            return Err(WeightError::DescriptorMismatch(entry.name.clone()));
        }
        if ordered[index].replace(entry).is_some() {
            return Err(WeightError::DescriptorMismatch(format!(
                "duplicate PLE shard index {index}"
            )));
        }
    }
    ordered
        .into_iter()
        .enumerate()
        .map(|(index, entry)| {
            entry.ok_or_else(|| {
                WeightError::DescriptorMismatch(format!("missing PLE shard index {index}"))
            })
        })
        .collect()
}

fn ple_descriptors(
    transaction: &WeightLoadTransaction,
    manifest: &Qwen4Manifest,
) -> Result<Vec<SourceRangeDescriptor>, BundleError> {
    let entries = ordered_ple_entries(manifest).map_err(BundleError::Weights)?;
    let mut descriptors: Vec<SourceRangeDescriptor> = Vec::with_capacity(entries.len());
    for (index, entry) in entries.into_iter().enumerate() {
        let descriptor = transaction
            .external_descriptor(&entry.name, entry.layer, 0)
            .ok_or_else(|| {
                BundleError::Weights(WeightError::MissingExternalDescriptor {
                    name: entry.name.clone(),
                    layer: entry.layer,
                    device: 0,
                })
            })?
            .clone();
        let (row_bytes, valid_rows) = match entry.residency {
            WeightResidency::ExternalRows {
                row_bytes,
                valid_rows,
            } => (row_bytes, valid_rows),
            WeightResidency::Resident => {
                return Err(BundleError::Weights(WeightError::DescriptorMismatch(
                    entry.name.clone(),
                )))
            }
        };
        if row_bytes != PLE_ROW_WIDTH * 2 || valid_rows != ple_valid_rows_for_shard(index) {
            return Err(BundleError::Weights(WeightError::DescriptorMismatch(
                entry.name.clone(),
            )));
        }
        if let Some(first) = descriptors.first() {
            if first.source_identity() != descriptor.source_identity() {
                return Err(BundleError::Weights(WeightError::DescriptorMismatch(
                    format!("PLE shard {index} source identity differs"),
                )));
            }
        }
        ExternalRowsRef {
            name: entry.name.clone(),
            layer: 1,
            row_bytes,
            physical_rows: PLE_SHARD_ROWS,
            valid_rows,
        }
        .validate_descriptor(&descriptor)
        .map_err(BundleError::Weights)?;
        descriptors.push(descriptor);
    }
    Ok(descriptors)
}


fn cleanup_transaction(primary: BundleError, rollback: hip_bridge::HipResult<()>) -> BundleError {
    match rollback {
        Ok(()) => primary,
        Err(error) => BundleError::Rollback {
            cause: primary.to_string(),
            error,
        },
    }
}

fn cleanup_bundle_failure(
    primary: BundleError,
    weights: hip_bridge::HipResult<()>,
    transaction: hip_bridge::HipResult<()>,
) -> BundleError {
    match (weights, transaction) {
        (Ok(()), Ok(())) => primary,
        (Err(weight), Ok(())) => BundleError::Rollback {
            cause: format!("{primary}; resident weight cleanup failed"),
            error: weight,
        },
        (Ok(()), Err(transaction)) => BundleError::Rollback {
            cause: format!("{primary}; transaction cleanup failed"),
            error: transaction,
        },
        (Err(weight), Err(transaction)) => BundleError::Rollback {
            cause: format!("{primary}; resident and transaction cleanup failed: {transaction}"),
            error: weight,
        },
    }
}

fn first_bundle_error(results: [Result<(), BundleError>; 4]) -> Result<(), BundleError> {
    let mut first = None;
    for result in results {
        if let Err(error) = result {
            if first.is_none() {
                first = Some(error);
            }
        }
    }
    first.map_or(Ok(()), Err)
}

#[derive(Debug)]
pub enum BundleError {
    Config(String),
    Weights(WeightError),
    State(StateError),
    PleRows(PleRowsError),
    Hip(hip_bridge::HipError),
    Rollback {
        cause: String,
        error: hip_bridge::HipError,
    },
    Transaction(WeightStoreError),
}

impl From<WeightError> for BundleError {
    fn from(value: WeightError) -> Self {
        Self::Weights(value)
    }
}

impl From<WeightStoreError> for BundleError {
    fn from(value: WeightStoreError) -> Self {
        Self::Transaction(value)
    }
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "Qwen4 bundle config: {message}"),
            Self::Weights(error) => write!(f, "Qwen4 bundle weights: {error}"),
            Self::State(error) => write!(f, "Qwen4 bundle state: {error}"),
            Self::PleRows(error) => write!(f, "Qwen4 bundle PLE rows: {error}"),
            Self::Hip(error) => write!(f, "Qwen4 bundle HIP teardown: {error}"),
            Self::Rollback { cause, error } => {
                write!(f, "Qwen4 bundle cleanup after {cause}: {error}")
            }
            Self::Transaction(error) => write!(f, "Qwen4 bundle transaction: {error}"),
        }
    }
}

impl std::error::Error for BundleError {}
