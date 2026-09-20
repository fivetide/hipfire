// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Model-owned storage for Qwen4's SSD-resident per-layer embeddings (PLE).
//!
//! `PleRows` deliberately owns the reader worker and its bounded resources.  It
//! does not schedule generation, mutate model state, or perform GPU work.  A
//! caller computes a token-known prefetch before layer 0, waits for the
//! completed [`PleRowLease`] at layer 1, uploads the contiguous staging bytes,
//! and drops the lease after the device has consumed them.
//!
//! The production source is a vector of sealed [`SourceRangeDescriptor`]s,
//! one for each physical PLE shard.  Reads are positional and descriptor
//! checked; no mmap, source-wide `Vec`, or file-backed GPU pointer is used.

use crate::ple::{PleHashMetadata, PleHistory, PLE_HEAD_COUNT, PLE_ROW_WIDTH};
use crate::weights::{PLE_SHARD_COUNT, PLE_SHARD_ROWS};
use hipfire_runtime::model_source::{SourceError, SourceRangeDescriptor};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Number of n-gram rows needed for every token.
pub const PLE_ROWS_PER_TOKEN: usize = PLE_HEAD_COUNT;
/// A physical PLE row is 160 BF16 values.
pub const PLE_ROW_BYTES: usize = PLE_ROW_WIDTH * 2;
/// Userspace cache budget.  This is deliberately fixed rather than a public
/// runtime knob: the model's SSD path must remain bounded on every request.
pub const PLE_PAGE_CACHE_BYTES: usize = 256 * 1024 * 1024;
/// Target page size used by the userspace cache.
///
/// A page is both the cache unit and the read unit, so this constant bounds how
/// many bytes one 320-byte row can cost.  Requested rows are drawn at random
/// over the whole 102 GB physical table, so a page read almost never serves any
/// other requested row and the cost of one row is the whole page.  Measured on
/// gfx1151 with a 2 MiB target: a 291-token chunk requested 4656 rows in 4483
/// distinct windows and pulled ~9.4 GB for ~1.5 MB of useful bytes — 492 ms of
/// page-cache copy (1810 ms cold) spent with the GPU idle waiting for the lease.
/// Matching the OS page (whose cache line the read lands in anyway) keeps the
/// window within the row's own file page instead of amplifying it ~6500x.
const PLE_PAGE_TARGET_BYTES: usize = 4 * 1024;
/// Page size rounded down to a whole number of physical rows.
///
/// Keeping the public geometry row aligned avoids partial-row pages and makes
/// every positional read a multiple of the source row width.
pub const PLE_PAGE_BYTES: usize = (PLE_PAGE_TARGET_BYTES / PLE_ROW_BYTES) * PLE_ROW_BYTES;
/// Maximum of bytes in one coalesced read or one staging buffer, rounded down
/// to a whole number of rows.
pub const PLE_STAGING_BYTES: usize = (8 * 1024 * 1024 / PLE_ROW_BYTES) * PLE_ROW_BYTES;
/// At most two staging buffers exist: one completed output and one read buffer.
pub const PLE_MAX_STAGING_BUFFERS: usize = 2;
/// Bounded queue of request ids.  A completed request also occupies one ticket
/// until its lease is consumed or the request is cancelled.
pub const PLE_READER_QUEUE_CAPACITY: usize = 64;

const PLE_ROWS_PER_PAGE: usize = PLE_PAGE_BYTES / PLE_ROW_BYTES;
const PLE_MAX_TOKENS_PER_PREFETCH: usize = PLE_STAGING_BYTES / (PLE_ROWS_PER_TOKEN * PLE_ROW_BYTES);

/// A page in the physical shard layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct PageKey {
    shard: usize,
    page: usize,
}

/// Checked location of one global PLE row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleRowLocation {
    /// Global row in the concatenated sixteen-head table.
    pub global_row: u64,
    /// Physical shard containing the row.
    pub shard: usize,
    /// Row number within that shard.
    pub local_row: usize,
    /// Page number within that shard.
    pub page: usize,
    /// Byte offset within the page.
    pub page_byte_offset: usize,
}

impl PleRowLocation {
    fn page_key(self) -> PageKey {
        PageKey {
            shard: self.shard,
            page: self.page,
        }
    }
}

/// Counters and bounded-resource measurements for diagnostics and unload
/// proofs.  These are snapshots; they do not grant access to mutable storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleCacheStats {
    pub capacity_bytes: usize,
    pub resident_bytes: usize,
    pub resident_pages: usize,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub reads: u64,
    pub coalesced_reads: u64,
    pub read_bytes: u64,
    pub evictions: u64,
    pub queue_depth: usize,
    pub outstanding_readers: usize,
    pub outstanding_leases: usize,
    pub staging_in_use: usize,
    pub staging_high_water: usize,
}

/// Proof returned by [`PleRows::quiesce`] and [`PleRows::unload`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PleQuiesceReport {
    pub queue_depth: usize,
    pub outstanding_readers: usize,
    pub outstanding_leases: usize,
    pub staging_in_use: usize,
    pub cache_bytes: usize,
    pub cache_pages: usize,
}

impl PleQuiesceReport {
    pub const fn is_clean(self) -> bool {
        self.queue_depth == 0
            && self.outstanding_readers == 0
            && self.outstanding_leases == 0
            && self.staging_in_use == 0
    }
}

/// Failure modes of the bounded PLE reader.  In particular, source read
/// failures are preserved as [`SourceError`] and never become zero rows.
#[derive(Debug)]
pub enum PleRowsError {
    Descriptor {
        index: usize,
        reason: String,
    },
    Source(SourceError),
    GlobalRowOutOfRange {
        row: u64,
        padded_rows: u64,
    },
    PaddingRow {
        row: u64,
        valid_rows: u64,
    },
    RequestTooLarge {
        tokens: usize,
        max_tokens: usize,
    },
    QueueFull,
    StagingUnavailable,
    Quiescing,
    EpochMismatch {
        requested: u64,
        current: u64,
    },
    EpochNotMonotonic {
        requested: u64,
        current: u64,
    },
    Canceled,
    UnknownTicket(u64),
    AlreadyConsumed(u64),
    WorkerStopped,
    InvalidLease {
        reason: String,
    },
    LeaseStale {
        lease_epoch: u64,
        current_epoch: u64,
    },
    QuiesceTimeout(PleQuiesceReport),
    WorkerJoin,
}

impl fmt::Display for PleRowsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptor { index, reason } => {
                write!(f, "PLE shard descriptor {index} is invalid: {reason}")
            }
            Self::Source(error) => write!(f, "PLE source read failed: {error}"),
            Self::GlobalRowOutOfRange { row, padded_rows } => {
                write!(f, "PLE row {row} is outside {padded_rows} physical rows")
            }
            Self::PaddingRow { row, valid_rows } => {
                write!(
                    f,
                    "PLE row {row} is padding (valid rows end at {valid_rows})"
                )
            }
            Self::RequestTooLarge { tokens, max_tokens } => {
                write!(
                    f,
                    "PLE prefetch has {tokens} tokens; maximum is {max_tokens}"
                )
            }
            Self::QueueFull => f.write_str("PLE reader queue is full"),
            Self::StagingUnavailable => f.write_str("PLE staging buffers are all in use"),
            Self::Quiescing => f.write_str("PLE rows are quiescing or unloaded"),
            Self::EpochMismatch { requested, current } => {
                write!(
                    f,
                    "PLE request epoch {requested} does not match current epoch {current}"
                )
            }
            Self::EpochNotMonotonic { requested, current } => {
                write!(
                    f,
                    "PLE epoch {requested} is not newer than current epoch {current}"
                )
            }
            Self::Canceled => f.write_str("PLE prefetch was canceled"),
            Self::UnknownTicket(id) => write!(f, "unknown PLE prefetch ticket {id}"),
            Self::AlreadyConsumed(id) => write!(f, "PLE prefetch ticket {id} was already consumed"),
            Self::WorkerStopped => f.write_str("PLE reader worker has stopped"),
            Self::InvalidLease { reason } => write!(f, "invalid PLE lease: {reason}"),
            Self::LeaseStale {
                lease_epoch,
                current_epoch,
            } => write!(
                f,
                "PLE lease epoch {lease_epoch} is stale at current epoch {current_epoch}"
            ),
            Self::QuiesceTimeout(report) => write!(f, "PLE quiesce timed out: {report:?}"),
            Self::WorkerJoin => f.write_str("PLE reader worker did not join cleanly"),
        }
    }
}

impl std::error::Error for PleRowsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Source(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SourceError> for PleRowsError {
    fn from(error: SourceError) -> Self {
        Self::Source(error)
    }
}

/// A completed, epoch-bound set of sixteen rows per token.
///
/// The lease owns exactly one of the two fixed staging buffers until dropped.
/// It intentionally exposes a borrowed byte slice rather than moving the
/// buffer out: callers cannot accidentally retain an unbounded third staging
/// allocation past model unload.
pub struct PleRowLease {
    inner: Arc<PleRowsInner>,
    epoch: u64,
    ids: Vec<[u64; PLE_ROWS_PER_TOKEN]>,
    bytes: Vec<u8>,
}

impl fmt::Debug for PleRowLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PleRowLease")
            .field("epoch", &self.epoch)
            .field("tokens", &self.ids.len())
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

impl PleRowLease {
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn token_count(&self) -> usize {
        self.ids.len()
    }

    /// Global row ids in token-major/head-minor order.
    pub fn row_ids(&self) -> &[[u64; PLE_ROWS_PER_TOKEN]] {
        &self.ids
    }

    /// Borrow the contiguous `[tokens, 16, 160]` BF16 staging bytes only
    /// while this lease's epoch is current.  A fallible view is intentional:
    /// returning an unconditional slice would let a reset publish stale rows.
    pub fn as_bytes(&self) -> Result<&[u8], PleRowsError> {
        self.try_bytes()
    }

    pub fn validate(&self) -> Result<(), PleRowsError> {
        self.inner.validate_lease_epoch(self.epoch)
    }

    /// Re-check the reader epoch after the caller's device upload has
    /// completed.  The upload itself is intentionally owned by the caller;
    /// this method closes the second half of the publication boundary so a
    /// reset cannot silently publish stale rows.
    pub fn validate_after_upload(&self) -> Result<(), PleRowsError> {
        self.validate()
    }

    pub fn try_bytes(&self) -> Result<&[u8], PleRowsError> {
        self.validate()?;
        Ok(&self.bytes)
    }

    pub fn row_bytes(&self, token: usize, head: usize) -> Result<&[u8], PleRowsError> {
        self.validate()?;
        if token >= self.ids.len() || head >= PLE_ROWS_PER_TOKEN {
            return Err(PleRowsError::InvalidLease {
                reason: format!("row index token={token}, head={head}"),
            });
        }
        let row = token
            .checked_mul(PLE_ROWS_PER_TOKEN)
            .and_then(|v| v.checked_add(head))
            .ok_or_else(|| PleRowsError::InvalidLease {
                reason: "row index overflow".to_string(),
            })?;
        let begin = row
            .checked_mul(PLE_ROW_BYTES)
            .ok_or_else(|| PleRowsError::InvalidLease {
                reason: "row byte offset overflow".to_string(),
            })?;
        Ok(&self.bytes[begin..begin + PLE_ROW_BYTES])
    }

    /// Copy all rows in one contiguous operation into caller-owned upload
    /// storage.  This is the CPU staging boundary; callers must not copy one
    /// row at a time into separate device allocations.  The epoch is checked
    /// both before and after the copy so a reset racing this operation is
    /// reported instead of publishing stale bytes.
    pub fn stage_into(&self, destination: &mut [u8]) -> Result<(), PleRowsError> {
        self.validate()?;
        if destination.len() != self.bytes.len() {
            return Err(PleRowsError::InvalidLease {
                reason: format!(
                    "staging destination has {} bytes, expected {}",
                    destination.len(),
                    self.bytes.len()
                ),
            });
        }
        destination.copy_from_slice(&self.bytes);
        self.validate_after_upload()
    }
}

impl Drop for PleRowLease {
    fn drop(&mut self) {
        let bytes = std::mem::take(&mut self.bytes);
        let mut state = self.inner.state.lock().expect("PLE state mutex poisoned");
        self.inner.return_staging_locked(&mut state, bytes);
        state.active_leases = state.active_leases.saturating_sub(1);
        self.inner.cv.notify_all();
    }
}

/// Handle for a queued PLE read.  Dropping it invalidates an unfinished or
/// completed result and returns its staging buffer once the reader is safe.
pub struct PlePrefetch {
    inner: Weak<PleRowsInner>,
    id: u64,
    epoch: u64,
    canceled: Arc<AtomicBool>,
    consumed: AtomicBool,
}

impl fmt::Debug for PlePrefetch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlePrefetch")
            .field("id", &self.id)
            .field("epoch", &self.epoch)
            .field("canceled", &self.canceled.load(Ordering::Acquire))
            .field("consumed", &self.consumed.load(Ordering::Acquire))
            .finish()
    }
}

impl PlePrefetch {
    pub const fn id(&self) -> u64 {
        self.id
    }

    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn row_ids(&self) -> Result<Vec<[u64; PLE_ROWS_PER_TOKEN]>, PleRowsError> {
        let inner = self.inner.upgrade().ok_or(PleRowsError::WorkerStopped)?;
        let state = inner.state.lock().expect("PLE state mutex poisoned");
        let ticket = state.tickets.get(&self.id).ok_or_else(|| {
            if self.canceled.load(Ordering::Acquire) {
                PleRowsError::Canceled
            } else {
                PleRowsError::UnknownTicket(self.id)
            }
        })?;
        Ok(ticket.ids.clone())
    }

    pub fn cancel(&self) -> Result<(), PleRowsError> {
        if self.consumed.load(Ordering::Acquire) {
            return Err(PleRowsError::AlreadyConsumed(self.id));
        }
        self.canceled.store(true, Ordering::Release);
        let inner = self.inner.upgrade().ok_or(PleRowsError::WorkerStopped)?;
        inner.cancel_ticket(self.id)
    }
}

impl Drop for PlePrefetch {
    fn drop(&mut self) {
        if !self.consumed.load(Ordering::Acquire) {
            self.canceled.store(true, Ordering::Release);
            if let Some(inner) = self.inner.upgrade() {
                let _ = inner.cancel_ticket(self.id);
            }
        }
    }
}

/// Model-owned bounded PLE reader/cache.
pub struct PleRows {
    inner: Arc<PleRowsInner>,
    worker: Option<JoinHandle<()>>,
}

impl fmt::Debug for PleRows {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let report = self.inner.report();
        f.debug_struct("PleRows")
            .field("shards", &self.inner.shard_count)
            .field("rows_per_shard", &self.inner.rows_per_shard)
            .field("report", &report)
            .finish()
    }
}

impl PleRows {
    /// Open production PLE storage from sealed shard descriptors.
    ///
    /// All descriptors must refer to one immutable source identity and have
    /// shape `[rows_per_shard, 160]`, BF16 dtype, and exactly `rows * 320`
    /// bytes.  The physical rows must exactly cover `metadata.padded_rows()`;
    /// requests are still checked against `metadata.valid_rows()` so padding
    /// can never be returned as an embedding.
    pub fn new(
        descriptors: Vec<SourceRangeDescriptor>,
        metadata: PleHashMetadata,
    ) -> Result<Self, PleRowsError> {
        let descriptors: Arc<[SourceRangeDescriptor]> = descriptors.into();
        validate_descriptors(&descriptors, &metadata)?;
        let source: Arc<dyn PositionalRowSource> = Arc::new(DescriptorRowSource {
            descriptors: descriptors.clone(),
        });
        Self::spawn(source, descriptors, metadata)
    }

    fn spawn(
        source: Arc<dyn PositionalRowSource>,
        descriptors: Arc<[SourceRangeDescriptor]>,
        metadata: PleHashMetadata,
    ) -> Result<Self, PleRowsError> {
        let rows_per_shard = descriptors
            .first()
            .and_then(|descriptor| descriptor.logical_shape().first().copied())
            .ok_or_else(|| PleRowsError::Descriptor {
                index: 0,
                reason: "no physical shards".to_string(),
            })?;
        let inner = Arc::new(PleRowsInner {
            source,
            stopped: AtomicBool::new(false),
            shard_count: descriptors.len(),
            _descriptors: descriptors,
            metadata,
            rows_per_shard,
            rows_per_page: PLE_ROWS_PER_PAGE,
            current_epoch: AtomicU64::new(0),
            epoch_started: AtomicBool::new(false),
            state: Mutex::new(PleRowsState::new()),
            cv: Condvar::new(),
        });
        let weak = Arc::downgrade(&inner);
        let worker = thread::Builder::new()
            .name("qwen4-ple-reader".to_string())
            .spawn(move || worker_loop(weak))
            .map_err(|error| PleRowsError::Descriptor {
                index: 0,
                reason: format!("cannot start bounded reader: {error}"),
            })?;
        Ok(Self {
            inner,
            worker: Some(worker),
        })
    }

    pub fn metadata(&self) -> &PleHashMetadata {
        &self.inner.metadata
    }

    pub fn shard_count(&self) -> usize {
        self.inner.shard_count
    }

    pub fn rows_per_shard(&self) -> usize {
        self.inner.rows_per_shard
    }

    pub const fn rows_per_page(&self) -> usize {
        PLE_ROWS_PER_PAGE
    }

    pub const fn max_tokens_per_prefetch(&self) -> usize {
        PLE_MAX_TOKENS_PER_PREFETCH
    }

    pub fn current_epoch(&self) -> u64 {
        self.inner.current_epoch.load(Ordering::Acquire)
    }

    /// Start a model/request epoch and invalidate every prior ticket.
    ///
    /// Epochs are single-use and strictly increasing after the first epoch.
    /// The first caller may choose epoch zero.  Invalidated queued tickets are
    /// removed immediately; an in-flight source read is allowed to finish
    /// under cancellation but can never publish a lease.
    pub fn begin_epoch(&self, epoch: u64) -> Result<(), PleRowsError> {
        let mut state = self.inner.state.lock().expect("PLE state mutex poisoned");
        if self.inner.stopped.load(Ordering::Acquire) {
            return Err(PleRowsError::WorkerStopped);
        }
        let current = self.current_epoch();
        let started = self.inner.epoch_started.load(Ordering::Acquire);
        if started && epoch <= current {
            return Err(PleRowsError::EpochNotMonotonic {
                requested: epoch,
                current,
            });
        }
        self.inner.current_epoch.store(epoch, Ordering::Release);
        self.inner.epoch_started.store(true, Ordering::Release);
        invalidate_tickets_locked(&self.inner, &mut state);
        self.inner.cv.notify_all();
        Ok(())
    }

    /// Invalidate the current request epoch, drain all queued/readers/leases,
    /// then resume the same model-owned cache for the next epoch.  Immutable
    /// cached pages survive; request tickets and output buffers do not.
    pub fn reset_epoch(&self, timeout: Duration) -> Result<u64, PleRowsError> {
        if self.inner.stopped.load(Ordering::Acquire) {
            return Err(PleRowsError::WorkerStopped);
        }
        let next = {
            let mut state = self.inner.state.lock().expect("PLE state mutex poisoned");
            let current = self.current_epoch();
            let next = current
                .checked_add(1)
                .ok_or(PleRowsError::EpochNotMonotonic {
                    requested: u64::MAX,
                    current: u64::MAX,
                })?;
            let started = self.inner.epoch_started.load(Ordering::Acquire);
            if started && next <= current {
                return Err(PleRowsError::EpochNotMonotonic {
                    requested: next,
                    current,
                });
            }
            self.inner.current_epoch.store(next, Ordering::Release);
            self.inner.epoch_started.store(true, Ordering::Release);
            invalidate_tickets_locked(&self.inner, &mut state);
            self.inner.cv.notify_all();
            next
        };
        self.quiesce(timeout)?;
        self.resume()?;
        Ok(next)
    }

    /// Compute all sixteen ids and enqueue a bounded, token-known prefetch.
    /// The supplied history is copied and advanced locally; callers should
    /// commit their own history only after the surrounding request commits.
    pub fn prefetch(
        &self,
        epoch: u64,
        history: PleHistory,
        tokens: &[u32],
    ) -> Result<PlePrefetch, PleRowsError> {
        if tokens.len() > PLE_MAX_TOKENS_PER_PREFETCH {
            return Err(PleRowsError::RequestTooLarge {
                tokens: tokens.len(),
                max_tokens: PLE_MAX_TOKENS_PER_PREFETCH,
            });
        }
        let current = self.current_epoch();
        if epoch != current {
            return Err(PleRowsError::EpochMismatch {
                requested: epoch,
                current,
            });
        }
        let (ids, locations, pages) = self.plan_rows(history, tokens)?;
        let required_bytes =
            locations
                .len()
                .checked_mul(PLE_ROW_BYTES)
                .ok_or(PleRowsError::RequestTooLarge {
                    tokens: tokens.len(),
                    max_tokens: PLE_MAX_TOKENS_PER_PREFETCH,
                })?;

        let mut state = self.inner.state.lock().expect("PLE state mutex poisoned");
        if !state.accepting || state.stop {
            return Err(PleRowsError::Quiescing);
        }
        let current = self.current_epoch();
        if epoch != current {
            return Err(PleRowsError::EpochMismatch {
                requested: epoch,
                current,
            });
        }
        if state.tickets.len() >= PLE_READER_QUEUE_CAPACITY {
            return Err(PleRowsError::QueueFull);
        }
        let id = state.next_ticket;
        state.next_ticket = state.next_ticket.wrapping_add(1);
        let canceled = Arc::new(AtomicBool::new(false));
        state.tickets.insert(
            id,
            TicketState {
                epoch,
                canceled: canceled.clone(),
                ids,
                locations,
                pages,
                required_bytes,
                output: None,
                status: TicketStatus::Pending,
            },
        );
        state.queue.push_back(id);
        self.inner.cv.notify_one();
        Ok(PlePrefetch {
            inner: Arc::downgrade(&self.inner),
            id,
            epoch,
            canceled,
            consumed: AtomicBool::new(false),
        })
    }

    /// Wait for a ticket's completed rows.  This is the layer-1 consumption
    /// boundary; all source reads and cache fills are complete on return.
    pub fn wait_completed_lease(&self, ticket: &PlePrefetch) -> Result<PleRowLease, PleRowsError> {
        let inner = ticket.inner.upgrade().ok_or(PleRowsError::WorkerStopped)?;
        if !Arc::ptr_eq(&inner, &self.inner) {
            return Err(PleRowsError::InvalidLease {
                reason: "ticket belongs to a different PLE resource".to_string(),
            });
        }
        if ticket
            .consumed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(PleRowsError::AlreadyConsumed(ticket.id));
        }
        let mut state = inner.state.lock().expect("PLE state mutex poisoned");
        loop {
            let done = match state.tickets.get(&ticket.id) {
                Some(ticket_state) => !matches!(ticket_state.status, TicketStatus::Pending),
                None => {
                    return Err(if ticket.canceled.load(Ordering::Acquire) {
                        PleRowsError::Canceled
                    } else {
                        PleRowsError::UnknownTicket(ticket.id)
                    })
                }
            };
            if done {
                break;
            }
            if state.stop && state.active_readers == 0 {
                return Err(PleRowsError::WorkerStopped);
            }
            state = inner.cv.wait(state).expect("PLE state mutex poisoned");
        }
        let ticket_state = state
            .tickets
            .remove(&ticket.id)
            .ok_or(PleRowsError::UnknownTicket(ticket.id))?;
        let ticket_canceled = ticket_state.canceled.load(Ordering::Acquire)
            || ticket_state.epoch != inner.current_epoch.load(Ordering::Acquire);
        match ticket_state.status {
            TicketStatus::Pending => unreachable!("completed wait observed pending ticket"),
            TicketStatus::Completed(Ok(())) if !ticket_canceled => {
                let bytes = ticket_state
                    .output
                    .ok_or_else(|| PleRowsError::InvalidLease {
                        reason: "completed ticket has no staging buffer".to_string(),
                    })?;
                state.active_leases += 1;
                let lease_inner = Arc::clone(&inner);
                drop(state);
                Ok(PleRowLease {
                    inner: lease_inner,
                    epoch: ticket_state.epoch,
                    ids: ticket_state.ids,
                    bytes,
                })
            }
            TicketStatus::Completed(Ok(())) => {
                if let Some(bytes) = ticket_state.output {
                    inner.return_staging_locked(&mut state, bytes);
                }
                inner.cv.notify_all();
                Err(PleRowsError::Canceled)
            }
            TicketStatus::Completed(Err(error)) => {
                if let Some(bytes) = ticket_state.output {
                    inner.return_staging_locked(&mut state, bytes);
                }
                inner.cv.notify_all();
                Err(error)
            }
        }
    }

    pub fn cancel(&self, ticket: &PlePrefetch) -> Result<(), PleRowsError> {
        ticket.cancel()
    }
    /// Alias that documents the required scheduling boundary in the caller.
    pub fn prefetch_before_layer0(
        &self,
        epoch: u64,
        history: PleHistory,
        tokens: &[u32],
    ) -> Result<PlePrefetch, PleRowsError> {
        self.prefetch(epoch, history, tokens)
    }

    /// Alias for the layer-1 completion boundary.  No GPU operation is
    /// performed here; the caller owns the single contiguous upload.
    pub fn consume_at_layer1(&self, ticket: &PlePrefetch) -> Result<PleRowLease, PleRowsError> {
        self.wait_completed_lease(ticket)
    }

    /// Stop accepting tickets, invalidate pending work, and wait for readers
    /// and leases to drain.  A timeout returns an unload proof showing exactly
    /// which owner is still live; it never pretends quiescence succeeded.
    pub fn quiesce(&self, timeout: Duration) -> Result<PleQuiesceReport, PleRowsError> {
        {
            let mut state = self.inner.state.lock().expect("PLE state mutex poisoned");
            state.accepting = false;
            invalidate_tickets_locked(&self.inner, &mut state);
            self.inner.cv.notify_all();
        }
        let deadline = Instant::now() + timeout;
        let mut state = self.inner.state.lock().expect("PLE state mutex poisoned");
        loop {
            if state.queue.is_empty() && state.active_readers == 0 && state.active_leases == 0 {
                purge_tickets_locked(&self.inner, &mut state);
                let report = self.inner.report_locked(&state);
                self.inner.cv.notify_all();
                return Ok(report);
            }
            let now = Instant::now();
            if now >= deadline {
                return Err(PleRowsError::QuiesceTimeout(
                    self.inner.report_locked(&state),
                ));
            }
            let remaining = deadline.saturating_duration_since(now);
            let (next, wait) = self
                .inner
                .cv
                .wait_timeout(state, remaining)
                .expect("PLE state mutex poisoned");
            state = next;
            if wait.timed_out() {
                return Err(PleRowsError::QuiesceTimeout(
                    self.inner.report_locked(&state),
                ));
            }
        }
    }

    fn quiesce_blocking(&self) -> Result<PleQuiesceReport, PleRowsError> {
        {
            let mut state = self.inner.state.lock().expect("PLE state mutex poisoned");
            state.accepting = false;
            invalidate_tickets_locked(&self.inner, &mut state);
            self.inner.cv.notify_all();
        }
        let mut state = self.inner.state.lock().expect("PLE state mutex poisoned");
        loop {
            if state.queue.is_empty() && state.active_readers == 0 && state.active_leases == 0 {
                purge_tickets_locked(&self.inner, &mut state);
                let report = self.inner.report_locked(&state);
                self.inner.cv.notify_all();
                return Ok(report);
            }
            state = self.inner.cv.wait(state).expect("PLE state mutex poisoned");
        }
    }

    pub fn resume(&self) -> Result<(), PleRowsError> {
        let mut state = self.inner.state.lock().expect("PLE state mutex poisoned");
        if state.stop || self.inner.stopped.load(Ordering::Acquire) {
            return Err(PleRowsError::WorkerStopped);
        }
        state.accepting = true;
        self.inner.cv.notify_all();
        Ok(())
    }

    /// Quiesce and join the model-owned reader with explicit blocking
    /// semantics.  Regular-file positional reads cannot be safely canceled,
    /// so this API never returns a timeout while a worker can still outlive
    /// its source owner.
    pub fn unload(mut self) -> Result<PleQuiesceReport, PleRowsError> {
        let report = self.quiesce_blocking()?;
        self.inner.stopped.store(true, Ordering::Release);
        self.inner.current_epoch.fetch_add(1, Ordering::AcqRel);
        {
            let mut state = self.inner.state.lock().expect("PLE state mutex poisoned");
            state.stop = true;
            self.inner.cv.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            worker.join().map_err(|_| PleRowsError::WorkerJoin)?;
        }
        Ok(report)
    }

    pub fn cache_stats(&self) -> PleCacheStats {
        self.inner.stats()
    }

    /// Drop all immutable userspace pages.  The source reader intentionally
    /// has no mmap or fd-pinning contract; dropping these pages is therefore a
    /// portable best-effort eviction of this cache, while the counters make
    /// kernel page-cache growth observable to the outer admission harness.
    pub fn evict_cached_pages(&self) -> usize {
        let mut state = self.inner.state.lock().expect("PLE state mutex poisoned");
        let dropped = state.cache.clear();
        self.inner.cv.notify_all();
        dropped
    }

    pub fn locate_row(&self, row: u64) -> Result<PleRowLocation, PleRowsError> {
        self.inner.locate_row(row)
    }

    fn plan_rows(
        &self,
        mut history: PleHistory,
        tokens: &[u32],
    ) -> Result<
        (
            Vec<[u64; PLE_ROWS_PER_TOKEN]>,
            Vec<PleRowLocation>,
            Vec<PageKey>,
        ),
        PleRowsError,
    > {
        let mut ids = Vec::with_capacity(tokens.len());
        let mut locations = Vec::with_capacity(tokens.len().saturating_mul(PLE_ROWS_PER_TOKEN));
        let mut pages = Vec::with_capacity(locations.capacity());
        for &token in tokens {
            let row_ids = history.hash_token(&self.inner.metadata, token);
            for &row in &row_ids {
                let location = self.inner.locate_row(row)?;
                pages.push(location.page_key());
                locations.push(location);
            }
            ids.push(row_ids);
        }
        pages.sort_unstable();
        pages.dedup();
        Ok((ids, locations, pages))
    }
}

impl Drop for PleRows {
    fn drop(&mut self) {
        self.inner.stopped.store(true, Ordering::Release);
        self.inner.current_epoch.fetch_add(1, Ordering::AcqRel);
        if let Ok(mut state) = self.inner.state.lock() {
            state.accepting = false;
            invalidate_tickets_locked(&self.inner, &mut state);
            state.stop = true;
            self.inner.cv.notify_all();
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

struct PleRowsInner {
    source: Arc<dyn PositionalRowSource>,
    shard_count: usize,
    _descriptors: Arc<[SourceRangeDescriptor]>,
    metadata: PleHashMetadata,
    rows_per_shard: usize,
    rows_per_page: usize,
    epoch_started: AtomicBool,
    current_epoch: AtomicU64,
    stopped: AtomicBool,
    state: Mutex<PleRowsState>,
    cv: Condvar,
}

impl PleRowsInner {
    fn locate_row(&self, row: u64) -> Result<PleRowLocation, PleRowsError> {
        if row >= self.metadata.padded_rows() {
            return Err(PleRowsError::GlobalRowOutOfRange {
                row,
                padded_rows: self.metadata.padded_rows(),
            });
        }
        if !self.metadata.is_valid_row(row) {
            return Err(PleRowsError::PaddingRow {
                row,
                valid_rows: self.metadata.valid_rows(),
            });
        }
        let shard_rows = self.rows_per_shard as u64;
        let shard = usize::try_from(row / shard_rows).map_err(|_| PleRowsError::InvalidLease {
            reason: "shard index overflow".to_string(),
        })?;
        let local_row =
            usize::try_from(row % shard_rows).map_err(|_| PleRowsError::InvalidLease {
                reason: "local row overflow".to_string(),
            })?;
        if shard >= self.shard_count {
            return Err(PleRowsError::GlobalRowOutOfRange {
                row,
                padded_rows: self.metadata.padded_rows(),
            });
        }
        let page = local_row / self.rows_per_page;
        let row_in_page = local_row % self.rows_per_page;
        let page_byte_offset =
            row_in_page
                .checked_mul(PLE_ROW_BYTES)
                .ok_or_else(|| PleRowsError::InvalidLease {
                    reason: "page byte offset overflow".to_string(),
                })?;
        Ok(PleRowLocation {
            global_row: row,
            shard,
            local_row,
            page,
            page_byte_offset,
        })
    }

    fn validate_lease_epoch(&self, lease_epoch: u64) -> Result<(), PleRowsError> {
        let current = self.current_epoch.load(Ordering::Acquire);
        if self.stopped.load(Ordering::Acquire) || current != lease_epoch {
            Err(PleRowsError::LeaseStale {
                lease_epoch,
                current_epoch: current,
            })
        } else {
            Ok(())
        }
    }

    fn report(&self) -> PleQuiesceReport {
        let state = self.state.lock().expect("PLE state mutex poisoned");
        self.report_locked(&state)
    }

    fn report_locked(&self, state: &PleRowsState) -> PleQuiesceReport {
        PleQuiesceReport {
            queue_depth: state.queue.len(),
            outstanding_readers: state.active_readers,
            outstanding_leases: state.active_leases,
            staging_in_use: state.staging_in_use,
            cache_bytes: state.cache.resident_bytes,
            cache_pages: state.cache.pages.len(),
        }
    }

    fn stats(&self) -> PleCacheStats {
        let state = self.state.lock().expect("PLE state mutex poisoned");
        PleCacheStats {
            capacity_bytes: PLE_PAGE_CACHE_BYTES,
            resident_bytes: state.cache.resident_bytes,
            resident_pages: state.cache.pages.len(),
            cache_hits: state.cache.hits,
            cache_misses: state.cache.misses,
            reads: state.cache.reads,
            coalesced_reads: state.cache.coalesced_reads,
            read_bytes: state.cache.read_bytes,
            evictions: state.cache.evictions,
            queue_depth: state.queue.len(),
            outstanding_readers: state.active_readers,
            outstanding_leases: state.active_leases,
            staging_in_use: state.staging_in_use,
            staging_high_water: state.staging_high_water,
        }
    }

    fn take_staging_locked(&self, state: &mut PleRowsState) -> Option<Vec<u8>> {
        let slot = state.staging_pool.iter().position(Option::is_some)?;
        let buffer = state.staging_pool[slot].take();
        state.staging_in_use += 1;
        state.staging_high_water = state.staging_high_water.max(state.staging_in_use);
        buffer
    }

    fn return_staging_locked(&self, state: &mut PleRowsState, mut buffer: Vec<u8>) {
        buffer.clear();
        if let Some(slot) = state.staging_pool.iter().position(Option::is_none) {
            state.staging_pool[slot] = Some(buffer);
        }
        state.staging_in_use = state.staging_in_use.saturating_sub(1);
    }

    fn cancel_ticket(&self, id: u64) -> Result<(), PleRowsError> {
        let mut state = self.state.lock().expect("PLE state mutex poisoned");
        let Some(ticket) = state.tickets.get(&id) else {
            return Ok(());
        };
        ticket.canceled.store(true, Ordering::Release);

        let queued = state.queue.iter().position(|queued_id| *queued_id == id);
        let completed = matches!(
            state.tickets.get(&id).map(|ticket| &ticket.status),
            Some(TicketStatus::Completed(_))
        );
        if let Some(position) = queued {
            state.queue.remove(position);
        }
        if queued.is_some() || completed {
            if let Some(ticket) = state.tickets.remove(&id) {
                if let Some(buffer) = ticket.output {
                    self.return_staging_locked(&mut state, buffer);
                }
            }
        }
        self.cv.notify_all();
        Ok(())
    }
    fn process_ticket(&self, id: u64, read_buffer: Vec<u8>) -> ProcessedTicket {
        let (epoch, canceled, pages, locations, output) = {
            let mut state = self.state.lock().expect("PLE state mutex poisoned");
            let Some(ticket) = state.tickets.get_mut(&id) else {
                return ProcessedTicket {
                    output: None,
                    read_buffer,
                    result: Err(PleRowsError::Canceled),
                };
            };
            (
                ticket.epoch,
                ticket.canceled.clone(),
                ticket.pages.clone(),
                ticket.locations.clone(),
                ticket.output.take(),
            )
        };
        let mut output = output.unwrap_or_default();
        let mut read_buffer = read_buffer;
        if canceled.load(Ordering::Acquire) {
            return ProcessedTicket {
                output: Some(output),
                read_buffer,
                result: Err(PleRowsError::Canceled),
            };
        }
        let current = self.current_epoch.load(Ordering::Acquire);
        if epoch != current {
            return ProcessedTicket {
                output: Some(output),
                read_buffer,
                result: Err(PleRowsError::EpochMismatch {
                    requested: epoch,
                    current,
                }),
            };
        }
        let result = self.fill_output(&locations, &pages, &mut output, &mut read_buffer, &canceled);
        ProcessedTicket {
            output: Some(output),
            read_buffer,
            result,
        }
    }

    /// Fill the final bounded lease directly while each page group is
    /// available.  Requested page sets may be much larger than the cache;
    /// no second pass assumes all pages remain resident.
    fn fill_output(
        &self,
        locations: &[PleRowLocation],
        pages: &[PageKey],
        output: &mut [u8],
        read_staging: &mut Vec<u8>,
        canceled: &AtomicBool,
    ) -> Result<(), PleRowsError> {
        let expected = locations.len().checked_mul(PLE_ROW_BYTES).ok_or_else(|| {
            PleRowsError::InvalidLease {
                reason: "staging size overflow".to_string(),
            }
        })?;
        if output.len() != expected {
            return Err(PleRowsError::InvalidLease {
                reason: "staging size does not match row plan".to_string(),
            });
        }
        let mut requested: HashMap<PageKey, Vec<RowCopy>> = HashMap::new();
        for (index, location) in locations.iter().copied().enumerate() {
            requested
                .entry(location.page_key())
                .or_default()
                .push(RowCopy {
                    output_offset: index * PLE_ROW_BYTES,
                    page_offset: location.page_byte_offset,
                });
        }

        let mut missing = Vec::new();
        for &page in pages {
            if canceled.load(Ordering::Acquire) {
                return Err(PleRowsError::Canceled);
            }
            let mut state = self.state.lock().expect("PLE state mutex poisoned");
            if let Some(cached) = state.cache.pages.get(&page) {
                copy_requested_rows(
                    page,
                    &cached.bytes,
                    requested.get(&page).map(Vec::as_slice).unwrap_or(&[]),
                    output,
                )?;
                state.cache.touch(page);
                state.cache.hits += 1;
            } else {
                state.cache.misses += 1;
                missing.push(page);
            }
        }

        let groups = self.coalesce_pages(&missing)?;
        for group in groups {
            if canceled.load(Ordering::Acquire) {
                return Err(PleRowsError::Canceled);
            }
            self.read_group(&group, &requested, output, read_staging, canceled)?;
        }
        Ok(())
    }

    fn coalesce_pages(&self, pages: &[PageKey]) -> Result<Vec<Vec<PageKey>>, PleRowsError> {
        let mut groups: Vec<Vec<PageKey>> = Vec::new();
        let mut current_bytes = 0usize;
        for &page in pages {
            let page_bytes = self.page_len(page)?;
            let append = groups.last().is_some_and(|group| {
                let previous = group[group.len() - 1];
                previous.shard == page.shard
                    && previous.page.checked_add(1) == Some(page.page)
                    && current_bytes
                        .checked_add(page_bytes)
                        .is_some_and(|total| total <= PLE_STAGING_BYTES)
            });
            if append {
                groups.last_mut().expect("group exists").push(page);
                current_bytes = current_bytes.checked_add(page_bytes).ok_or_else(|| {
                    PleRowsError::InvalidLease {
                        reason: "coalesced length overflow".to_string(),
                    }
                })?;
            } else {
                groups.push(vec![page]);
                current_bytes = page_bytes;
            }
        }
        Ok(groups)
    }

    fn page_len(&self, page: PageKey) -> Result<usize, PleRowsError> {
        if page.shard >= self.shard_count {
            return Err(PleRowsError::GlobalRowOutOfRange {
                row: self.metadata.padded_rows(),
                padded_rows: self.metadata.padded_rows(),
            });
        }
        let first_row = page.page.checked_mul(self.rows_per_page).ok_or_else(|| {
            PleRowsError::InvalidLease {
                reason: "page row offset overflow".to_string(),
            }
        })?;
        if first_row >= self.rows_per_shard {
            return Err(PleRowsError::InvalidLease {
                reason: "page exceeds physical shard".to_string(),
            });
        }
        let rows = (self.rows_per_shard - first_row).min(self.rows_per_page);
        rows.checked_mul(PLE_ROW_BYTES)
            .ok_or_else(|| PleRowsError::InvalidLease {
                reason: "page byte length overflow".to_string(),
            })
    }
    fn read_group(
        &self,
        group: &[PageKey],
        requested: &HashMap<PageKey, Vec<RowCopy>>,
        output: &mut [u8],
        read_staging: &mut Vec<u8>,
        canceled: &AtomicBool,
    ) -> Result<(), PleRowsError> {
        let first = *group.first().ok_or_else(|| PleRowsError::InvalidLease {
            reason: "empty coalesced page group".to_string(),
        })?;
        let first_offset = first
            .page
            .checked_mul(self.rows_per_page)
            .and_then(|row| row.checked_mul(PLE_ROW_BYTES))
            .ok_or_else(|| PleRowsError::InvalidLease {
                reason: "coalesced offset overflow".to_string(),
            })? as u64;
        let lengths: Vec<usize> = group
            .iter()
            .map(|&page| self.page_len(page))
            .collect::<Result<_, _>>()?;
        let total = lengths
            .iter()
            .try_fold(0usize, |sum, len| sum.checked_add(*len))
            .ok_or_else(|| PleRowsError::InvalidLease {
                reason: "coalesced length overflow".to_string(),
            })?;
        if total > PLE_STAGING_BYTES {
            return Err(PleRowsError::InvalidLease {
                reason: "coalesced read exceeds staging budget".to_string(),
            });
        }

        read_staging.resize(total, 0);
        self.source
            .read_at(first.shard, first_offset, read_staging)
            .map_err(PleRowsError::Source)?;
        if canceled.load(Ordering::Acquire) {
            return Err(PleRowsError::Canceled);
        }

        let mut state = self.state.lock().expect("PLE state mutex poisoned");
        let mut cursor = 0usize;
        for (&page, &length) in group.iter().zip(&lengths) {
            let page_bytes = &read_staging[cursor..cursor + length];
            if let Some(rows) = requested.get(&page) {
                copy_requested_rows(page, page_bytes, rows, output)?;
            }
            state.cache.insert(page, page_bytes.to_vec());
            cursor += length;
        }
        state.cache.reads += 1;
        state.cache.coalesced_reads += 1;
        state.cache.read_bytes += total as u64;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct RowCopy {
    output_offset: usize,
    page_offset: usize,
}

fn copy_requested_rows(
    page: PageKey,
    page_bytes: &[u8],
    rows: &[RowCopy],
    output: &mut [u8],
) -> Result<(), PleRowsError> {
    for row in rows {
        let page_end = row.page_offset.checked_add(PLE_ROW_BYTES).ok_or_else(|| {
            PleRowsError::InvalidLease {
                reason: format!("page {page:?} row offset overflow"),
            }
        })?;
        let output_end = row
            .output_offset
            .checked_add(PLE_ROW_BYTES)
            .ok_or_else(|| PleRowsError::InvalidLease {
                reason: "output row offset overflow".to_string(),
            })?;
        if page_end > page_bytes.len() || output_end > output.len() {
            return Err(PleRowsError::InvalidLease {
                reason: format!("requested row is outside page {page:?} or output"),
            });
        }
        output[row.output_offset..output_end]
            .copy_from_slice(&page_bytes[row.page_offset..page_end]);
    }
    Ok(())
}

struct ProcessedTicket {
    output: Option<Vec<u8>>,
    read_buffer: Vec<u8>,
    result: Result<(), PleRowsError>,
}

struct TicketState {
    epoch: u64,
    canceled: Arc<AtomicBool>,
    ids: Vec<[u64; PLE_ROWS_PER_TOKEN]>,
    locations: Vec<PleRowLocation>,
    pages: Vec<PageKey>,
    required_bytes: usize,
    output: Option<Vec<u8>>,
    status: TicketStatus,
}

enum TicketStatus {
    Pending,
    Completed(Result<(), PleRowsError>),
}

struct PleRowsState {
    queue: VecDeque<u64>,
    tickets: HashMap<u64, TicketState>,
    next_ticket: u64,
    accepting: bool,
    stop: bool,
    active_readers: usize,
    active_leases: usize,
    staging_pool: Vec<Option<Vec<u8>>>,
    staging_in_use: usize,
    staging_high_water: usize,
    cache: PageCache,
}

impl PleRowsState {
    fn new() -> Self {
        let staging_pool = (0..PLE_MAX_STAGING_BUFFERS)
            .map(|_| Some(Vec::with_capacity(PLE_STAGING_BYTES)))
            .collect();
        Self {
            queue: VecDeque::new(),
            tickets: HashMap::new(),
            next_ticket: 1,
            accepting: true,
            stop: false,
            active_readers: 0,
            active_leases: 0,
            staging_pool,
            staging_in_use: 0,

            staging_high_water: 0,
            cache: PageCache::new(PLE_PAGE_CACHE_BYTES),
        }
    }
}
fn invalidate_tickets_locked(inner: &PleRowsInner, state: &mut PleRowsState) {
    for ticket in state.tickets.values() {
        ticket.canceled.store(true, Ordering::Release);
    }
    let queued: Vec<u64> = state.queue.drain(..).collect();
    for id in queued {
        if let Some(ticket) = state.tickets.remove(&id) {
            if let Some(buffer) = ticket.output {
                inner.return_staging_locked(state, buffer);
            }
        }
    }
    let completed: Vec<u64> = state
        .tickets
        .iter()
        .filter_map(|(&id, ticket)| {
            matches!(ticket.status, TicketStatus::Completed(_)).then_some(id)
        })
        .collect();
    for id in completed {
        if let Some(ticket) = state.tickets.remove(&id) {
            if let Some(buffer) = ticket.output {
                inner.return_staging_locked(state, buffer);
            }
        }
    }
}

fn purge_tickets_locked(inner: &PleRowsInner, state: &mut PleRowsState) {
    let ids: Vec<u64> = state.tickets.keys().copied().collect();
    for id in ids {
        if let Some(ticket) = state.tickets.remove(&id) {
            if let Some(buffer) = ticket.output {
                inner.return_staging_locked(state, buffer);
            }
        }
    }
}

fn worker_loop(weak: Weak<PleRowsInner>) {
    loop {
        let Some(inner) = weak.upgrade() else {
            break;
        };
        let work = {
            let mut state = inner.state.lock().expect("PLE state mutex poisoned");
            loop {
                if state.stop {
                    break None;
                }
                let Some(id) = state.queue.front().copied() else {
                    state = inner.cv.wait(state).expect("PLE state mutex poisoned");
                    continue;
                };
                let Some(mut output) = inner.take_staging_locked(&mut state) else {
                    state = inner.cv.wait(state).expect("PLE state mutex poisoned");
                    continue;
                };
                let Some(read_buffer) = inner.take_staging_locked(&mut state) else {
                    inner.return_staging_locked(&mut state, output);
                    state = inner.cv.wait(state).expect("PLE state mutex poisoned");
                    continue;
                };
                state.queue.pop_front();
                let Some(ticket) = state.tickets.get_mut(&id) else {
                    inner.return_staging_locked(&mut state, output);
                    inner.return_staging_locked(&mut state, read_buffer);
                    continue;
                };
                output.resize(ticket.required_bytes, 0);
                ticket.output = Some(output);
                state.active_readers += 1;
                break Some((id, read_buffer));
            }
        };
        let Some((id, read_buffer)) = work else {
            break;
        };
        let ProcessedTicket {
            mut output,
            read_buffer,
            result,
        } = inner.process_ticket(id, read_buffer);
        let mut state = inner.state.lock().expect("PLE state mutex poisoned");
        state.active_readers = state.active_readers.saturating_sub(1);
        inner.return_staging_locked(&mut state, read_buffer);
        let Some((ticket_canceled, ticket_epoch)) = state
            .tickets
            .get(&id)
            .map(|ticket| (ticket.canceled.load(Ordering::Acquire), ticket.epoch))
        else {
            if let Some(buffer) = output.take() {
                inner.return_staging_locked(&mut state, buffer);
            }
            inner.cv.notify_all();
            continue;
        };
        let canceled =
            ticket_canceled || ticket_epoch != inner.current_epoch.load(Ordering::Acquire);
        if canceled {
            if let Some(ticket) = state.tickets.remove(&id) {
                if let Some(buffer) = ticket.output {
                    inner.return_staging_locked(&mut state, buffer);
                }
            }
            if let Some(buffer) = output.take() {
                inner.return_staging_locked(&mut state, buffer);
            }
            inner.cv.notify_all();
            continue;
        }
        let mut return_buffer = None;
        {
            let ticket = state.tickets.get_mut(&id).expect("ticket disappeared");
            match result {
                Ok(()) => {
                    ticket.output = output;
                    ticket.status = TicketStatus::Completed(Ok(()));
                }
                Err(error) => {
                    return_buffer = output.take();
                    ticket.output = None;
                    ticket.status = TicketStatus::Completed(Err(error));
                }
            }
        }
        if let Some(buffer) = return_buffer {
            inner.return_staging_locked(&mut state, buffer);
        }
        inner.cv.notify_all();
    }
}

trait PositionalRowSource: Send + Sync {
    fn read_at(&self, shard: usize, local_offset: u64, dst: &mut [u8]) -> Result<(), SourceError>;
}

struct DescriptorRowSource {
    descriptors: Arc<[SourceRangeDescriptor]>,
}

impl PositionalRowSource for DescriptorRowSource {
    fn read_at(&self, shard: usize, local_offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
        let descriptor = self
            .descriptors
            .get(shard)
            .ok_or_else(|| SourceError::InvalidSource {
                reason: format!("PLE shard {shard} is absent"),
            })?;
        let absolute =
            descriptor
                .offset
                .checked_add(local_offset)
                .ok_or(SourceError::Overflow {
                    offset: descriptor.offset,
                    length: local_offset,
                })?;
        descriptor.read_exact_at(absolute, dst)
    }
}

struct CachedPage {
    bytes: Vec<u8>,
    last_used: u64,
}

struct PageCache {
    capacity_bytes: usize,
    pages: HashMap<PageKey, CachedPage>,
    /// Recency index over [`CachedPage::last_used`]: stamp -> page.  A page is
    /// small, so the cache holds tens of thousands of entries and the least
    /// recently used page must be found without scanning the map.
    by_stamp: BTreeMap<u64, PageKey>,
    resident_bytes: usize,
    clock: u64,
    hits: u64,
    misses: u64,
    reads: u64,
    coalesced_reads: u64,
    read_bytes: u64,
    evictions: u64,
}

impl PageCache {
    fn new(capacity_bytes: usize) -> Self {
        Self {
            capacity_bytes,
            pages: HashMap::new(),
            by_stamp: BTreeMap::new(),
            resident_bytes: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            reads: 0,
            coalesced_reads: 0,
            read_bytes: 0,
            evictions: 0,
        }
    }

    fn touch(&mut self, key: PageKey) {
        self.clock = self.clock.wrapping_add(1);
        if let Some(page) = self.pages.get_mut(&key) {
            self.by_stamp.remove(&page.last_used);
            page.last_used = self.clock;
            self.by_stamp.insert(self.clock, key);
        }
    }

    fn insert(&mut self, key: PageKey, bytes: Vec<u8>) {
        if bytes.len() > self.capacity_bytes {
            return;
        }
        if let Some(previous) = self.pages.remove(&key) {
            self.by_stamp.remove(&previous.last_used);
            self.resident_bytes = self.resident_bytes.saturating_sub(previous.bytes.len());
        }
        while self.resident_bytes.saturating_add(bytes.len()) > self.capacity_bytes {
            let Some((&stamp, &victim)) = self.by_stamp.iter().next() else {
                break;
            };
            self.by_stamp.remove(&stamp);
            if let Some(page) = self.pages.remove(&victim) {
                self.resident_bytes = self.resident_bytes.saturating_sub(page.bytes.len());
                self.evictions += 1;
            }
        }
        self.clock = self.clock.wrapping_add(1);
        self.resident_bytes += bytes.len();
        self.by_stamp.insert(self.clock, key);
        self.pages.insert(
            key,
            CachedPage {
                bytes,
                last_used: self.clock,
            },
        );
    }

    fn clear(&mut self) -> usize {
        let dropped = self.resident_bytes;
        self.pages.clear();
        self.by_stamp.clear();
        self.resident_bytes = 0;
        dropped
    }
}

fn validate_descriptors(
    descriptors: &[SourceRangeDescriptor],
    metadata: &PleHashMetadata,
) -> Result<(), PleRowsError> {
    let first = descriptors
        .first()
        .ok_or_else(|| PleRowsError::Descriptor {
            index: 0,
            reason: "at least one physical shard is required".to_string(),
        })?;
    let shape = first.logical_shape();
    let rows = *shape.first().ok_or_else(|| PleRowsError::Descriptor {
        index: 0,
        reason: "logical shape is empty".to_string(),
    })?;
    if rows == 0 {
        return Err(PleRowsError::Descriptor {
            index: 0,
            reason: "physical shard has zero rows".to_string(),
        });
    }
    if shape.len() != 2 || shape[1] != PLE_ROW_WIDTH {
        return Err(PleRowsError::Descriptor {
            index: 0,
            reason: format!("expected [{rows}, {PLE_ROW_WIDTH}], got {shape:?}"),
        });
    }
    if descriptors.len() != PLE_SHARD_COUNT {
        return Err(PleRowsError::Descriptor {
            index: 0,
            reason: format!(
                "expected {PLE_SHARD_COUNT} physical PLE shards, got {}",
                descriptors.len()
            ),
        });
    }
    if rows != PLE_SHARD_ROWS {
        return Err(PleRowsError::Descriptor {
            index: 0,
            reason: format!("expected {PLE_SHARD_ROWS} rows per PLE shard, got {rows}"),
        });
    }
    let expected_length = (rows as u64)
        .checked_mul(PLE_ROW_BYTES as u64)
        .ok_or_else(|| PleRowsError::Descriptor {
            index: 0,
            reason: "descriptor length overflow".to_string(),
        })?;
    if first.length != expected_length {
        return Err(PleRowsError::Descriptor {
            index: 0,
            reason: format!("expected {expected_length} bytes, got {}", first.length),
        });
    }
    if !is_bf16(first.dtype()) {
        return Err(PleRowsError::Descriptor {
            index: 0,
            reason: format!("PLE rows must be BF16, got {}", first.dtype()),
        });
    }
    let identity = first.source_identity();
    for (index, descriptor) in descriptors.iter().enumerate() {
        if descriptor.source_identity() != identity {
            return Err(PleRowsError::Descriptor {
                index,
                reason: "descriptor source identity differs from shard 0".to_string(),
            });
        }
        let descriptor_shape = descriptor.logical_shape();
        if descriptor_shape.len() != 2
            || descriptor_shape[0] != rows
            || descriptor_shape[1] != PLE_ROW_WIDTH
        {
            return Err(PleRowsError::Descriptor {
                index,
                reason: format!("all shards must have shape [{rows}, {PLE_ROW_WIDTH}]"),
            });
        }
        if descriptor.length != expected_length {
            return Err(PleRowsError::Descriptor {
                index,
                reason: format!("all shards must have {expected_length} bytes"),
            });
        }
        if !is_bf16(descriptor.dtype()) {
            return Err(PleRowsError::Descriptor {
                index,
                reason: format!("PLE rows must be BF16, got {}", descriptor.dtype()),
            });
        }
    }
    let physical_rows = (rows as u64)
        .checked_mul(descriptors.len() as u64)
        .ok_or_else(|| PleRowsError::Descriptor {
            index: 0,
            reason: "physical row count overflow".to_string(),
        })?;
    if physical_rows != metadata.padded_rows() {
        return Err(PleRowsError::Descriptor {
            index: 0,
            reason: format!(
                "physical rows {physical_rows} do not equal metadata padded rows {}",
                metadata.padded_rows()
            ),
        });
    }
    Ok(())
}

fn is_bf16(dtype: &str) -> bool {
    matches!(dtype.to_ascii_lowercase().as_str(), "bf16" | "bfloat16")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;
    use std::sync::atomic::AtomicUsize;

    struct MemoryRowSource {
        shards: Vec<Vec<u8>>,
        reads: AtomicUsize,
        fail: Option<(usize, u64)>,
    }

    impl PositionalRowSource for MemoryRowSource {
        fn read_at(&self, shard: usize, offset: u64, dst: &mut [u8]) -> Result<(), SourceError> {
            self.reads.fetch_add(1, Ordering::AcqRel);
            if self.fail == Some((shard, offset)) {
                return Err(SourceError::Io {
                    offset,
                    source: io::Error::new(io::ErrorKind::Other, "fixture read failure"),
                });
            }
            let source = self
                .shards
                .get(shard)
                .ok_or_else(|| SourceError::InvalidSource {
                    reason: "fixture shard missing".to_string(),
                })?;
            let begin = usize::try_from(offset).map_err(|_| SourceError::Overflow {
                offset,
                length: dst.len() as u64,
            })?;
            let end = begin.checked_add(dst.len()).ok_or(SourceError::Overflow {
                offset,
                length: dst.len() as u64,
            })?;
            if end > source.len() {
                return Err(SourceError::ShortRead {
                    offset,
                    expected: dst.len(),
                    actual: source.len().saturating_sub(begin),
                });
            }
            dst.copy_from_slice(&source[begin..end]);
            Ok(())
        }
    }

    fn metadata_with_head_size(head_size: u64, padded_rows: u64) -> PleHashMetadata {
        let sizes = [head_size; PLE_HEAD_COUNT];
        let mut offsets = [0u64; PLE_HEAD_COUNT];
        for index in 1..PLE_HEAD_COUNT {
            offsets[index] = offsets[index - 1] + sizes[index - 1];
        }
        PleHashMetadata::from_stored_with_padding([1, 1, 1], sizes, offsets, padded_rows, 128)
            .unwrap_or_else(|error| panic!("metadata fixture: {error}"))
    }

    fn metadata(padded_rows: u64) -> PleHashMetadata {
        metadata_with_head_size(2, padded_rows)
    }

    fn rows_source(shard_count: usize, rows_per_shard: usize) -> Vec<Vec<u8>> {
        (0..shard_count)
            .map(|shard| {
                let mut bytes = vec![0u8; rows_per_shard * PLE_ROW_BYTES];
                for row in 0..rows_per_shard {
                    let global = shard * rows_per_shard + row;
                    let value = (global as u16).to_le_bytes();
                    for cell in
                        bytes[row * PLE_ROW_BYTES..(row + 1) * PLE_ROW_BYTES].chunks_exact_mut(2)
                    {
                        cell.copy_from_slice(&value);
                    }
                }
                bytes
            })
            .collect()
    }

    impl PleRows {
        fn from_test_source(
            metadata: PleHashMetadata,
            rows_per_shard: usize,
            source: Arc<MemoryRowSource>,
        ) -> Result<Self, PleRowsError> {
            Self::from_test_source_with_page_rows(
                metadata,
                rows_per_shard,
                PLE_ROWS_PER_PAGE,
                source,
            )
        }

        fn from_test_source_with_page_rows(
            metadata: PleHashMetadata,
            rows_per_shard: usize,
            rows_per_page: usize,
            source: Arc<MemoryRowSource>,
        ) -> Result<Self, PleRowsError> {
            if rows_per_shard == 0 || rows_per_page == 0 {
                return Err(PleRowsError::Descriptor {
                    index: 0,
                    reason: "invalid fixture shard geometry".to_string(),
                });
            }
            let shard_count = usize::try_from(metadata.padded_rows() / rows_per_shard as u64)
                .map_err(|_| PleRowsError::Descriptor {
                    index: 0,
                    reason: "test shard count overflow".to_string(),
                })?;
            if shard_count == 0 {
                return Err(PleRowsError::Descriptor {
                    index: 0,
                    reason: "invalid fixture shard geometry".to_string(),
                });
            }
            if (shard_count as u64) * rows_per_shard as u64 != metadata.padded_rows() {
                return Err(PleRowsError::Descriptor {
                    index: 0,
                    reason: "fixture rows do not cover padded metadata".to_string(),
                });
            }
            let inner = Arc::new(PleRowsInner {
                source,
                shard_count,
                _descriptors: Vec::new().into(),
                metadata,
                rows_per_shard,
                rows_per_page,
                current_epoch: AtomicU64::new(0),
                stopped: AtomicBool::new(false),
                epoch_started: AtomicBool::new(false),
                state: Mutex::new(PleRowsState::new()),
                cv: Condvar::new(),
            });
            let weak = Arc::downgrade(&inner);
            let worker = thread::Builder::new()
                .name("qwen4-ple-test-reader".to_string())
                .spawn(move || worker_loop(weak))
                .map_err(|error| PleRowsError::Descriptor {
                    index: 0,
                    reason: error.to_string(),
                })?;
            Ok(Self {
                inner,
                worker: Some(worker),
            })
        }
    }

    // The production constructor is covered by runtime source tests; these
    // tests exercise bounded cache and deterministic page planning.
    #[test]
    fn cache_is_bounded_and_evicts_lru() {
        let mut cache = PageCache::new(PLE_ROW_BYTES * 2);
        cache.insert(PageKey { shard: 0, page: 0 }, vec![1; PLE_ROW_BYTES]);
        cache.insert(PageKey { shard: 0, page: 1 }, vec![2; PLE_ROW_BYTES]);
        cache.touch(PageKey { shard: 0, page: 0 });
        cache.insert(PageKey { shard: 0, page: 2 }, vec![3; PLE_ROW_BYTES]);
        assert_eq!(cache.resident_bytes, PLE_ROW_BYTES * 2);
        assert_eq!(cache.pages.len(), 2);
        assert!(!cache.pages.contains_key(&PageKey { shard: 0, page: 1 }));
        assert_eq!(cache.evictions, 1);
    }

    #[test]
    fn location_rejects_padding_and_maps_shard_boundary() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = PleRows::from_test_source(metadata(128), 64, source).unwrap();
        assert_eq!(
            rows.locate_row(0).unwrap(),
            PleRowLocation {
                global_row: 0,
                shard: 0,
                local_row: 0,
                page: 0,
                page_byte_offset: 0,
            }
        );
        let error = rows.locate_row(32).unwrap_err();
        assert!(matches!(error, PleRowsError::PaddingRow { row: 32, .. }));
        let full = PleRows::from_test_source(
            metadata_with_head_size(8, 128),
            64,
            Arc::new(MemoryRowSource {
                shards: rows_source(2, 64),
                reads: AtomicUsize::new(0),
                fail: None,
            }),
        )
        .unwrap();
        assert_eq!(full.locate_row(63).unwrap().shard, 0);
        assert_eq!(full.locate_row(64).unwrap().shard, 1);
        assert!(full.unload().unwrap().is_clean());
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn prefetch_reads_once_and_preserves_token_head_order() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = PleRows::from_test_source(metadata(128), 64, source.clone()).unwrap();
        let ticket = rows.prefetch(0, PleHistory::new(99), &[0, 1]).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        assert_eq!(lease.token_count(), 2);
        assert_eq!(lease.row_ids(), expected_ids.as_slice());
        for token in 0..2 {
            for head in 0..PLE_ROWS_PER_TOKEN {
                let row = lease.row_bytes(token, head).unwrap();
                let value = u16::from_le_bytes([row[0], row[1]]) as u64;
                assert_eq!(value, lease.row_ids()[token][head]);
            }
        }
        let mut copied = vec![0; lease.as_bytes().unwrap().len()];
        lease.stage_into(&mut copied).unwrap();
        assert_eq!(copied, lease.as_bytes().unwrap());
        drop(lease);
        // Every distinct physical page is read exactly once for one ticket;
        // adjacent pages may coalesce into one call.
        let mut distinct = std::collections::HashSet::new();
        let mut expected_bytes = 0usize;
        for &row in expected_ids.iter().flatten() {
            let key = rows.locate_row(row).unwrap().page_key();
            if distinct.insert(key) {
                expected_bytes += rows.inner.page_len(key).unwrap();
            }
        }
        let stats = rows.cache_stats();
        assert_eq!(stats.read_bytes as usize, expected_bytes);
        assert!(stats.reads as usize <= distinct.len());
        assert_eq!(source.reads.load(Ordering::Acquire) as u64, stats.reads);
        let second = rows.prefetch(0, PleHistory::new(99), &[0, 1]).unwrap();
        let lease = rows.wait_completed_lease(&second).unwrap();
        drop(lease);
        assert_eq!(rows.cache_stats().read_bytes, stats.read_bytes);
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn source_error_and_epoch_invalidation_are_explicit() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: Some((0, 0)),
        });
        let rows = PleRows::from_test_source(metadata(128), 64, source).unwrap();
        let ticket = rows.prefetch(0, PleHistory::new(99), &[0]).unwrap();
        let error = rows.wait_completed_lease(&ticket).unwrap_err();
        assert!(matches!(
            error,
            PleRowsError::Source(SourceError::Io { .. })
        ));
        // This is the forward-attempt abort boundary: the consumed ticket
        // may already report a source error, but reset_epoch must still drain
        // the exact epoch before a caller returns to its request loop.
        assert_eq!(rows.reset_epoch(Duration::from_secs(5)).unwrap(), 1);
        let stats = rows.cache_stats();
        assert_eq!(stats.queue_depth, 0);
        assert_eq!(stats.outstanding_readers, 0);
        assert_eq!(stats.outstanding_leases, 0);
        assert_eq!(stats.staging_in_use, 0);
        assert!(rows.unload().unwrap().is_clean());

        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = PleRows::from_test_source(metadata(128), 64, source).unwrap();
        let ticket = rows.prefetch(0, PleHistory::new(99), &[0]).unwrap();
        rows.begin_epoch(1).unwrap();
        let error = rows.wait_completed_lease(&ticket).unwrap_err();
        assert!(matches!(
            error,
            PleRowsError::Canceled | PleRowsError::EpochMismatch { .. }
        ));
        // A canceled read follows the same abort path and must leave the
        // reader reusable at a later epoch rather than leaking its ticket.
        assert_eq!(rows.reset_epoch(Duration::from_secs(5)).unwrap(), 2);
        let retry = rows.prefetch(2, PleHistory::new(99), &[0]).unwrap();
        let lease = rows.wait_completed_lease(&retry).unwrap();
        drop(lease);
        let stats = rows.cache_stats();
        assert_eq!(stats.queue_depth, 0);
        assert_eq!(stats.outstanding_readers, 0);
        assert_eq!(stats.outstanding_leases, 0);
        assert_eq!(stats.staging_in_use, 0);
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn coalesced_page_plan_is_ordered_and_deduplicated() {
        let pages = vec![
            PageKey { shard: 1, page: 2 },
            PageKey { shard: 0, page: 1 },
            PageKey { shard: 0, page: 0 },
            PageKey { shard: 0, page: 1 },
        ];
        let mut pages = pages;
        pages.sort_unstable();
        pages.dedup();
        assert_eq!(
            pages,
            vec![
                PageKey { shard: 0, page: 0 },
                PageKey { shard: 0, page: 1 },
                PageKey { shard: 1, page: 2 },
            ]
        );
    }
    /// One requested 320-byte row must not drag a multi-megabyte window out of
    /// the source.  The page unit is also the read unit, so this pins the
    /// per-row amplification that the lease wait depends on: with a 2 MiB
    /// window a 291-token chunk read ~9.4 GB for ~1.5 MB of rows, which cost
    /// 492 ms of page-cache copy with the GPU idle.
    #[test]
    fn read_window_stays_within_the_row_page() {
        // Big enough that hashed rows land in distinct windows the way the
        // production table spreads them.
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 32768),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = PleRows::from_test_source_with_page_rows(
            metadata_with_head_size(4096, 65536),
            32768,
            PLE_ROWS_PER_PAGE,
            source.clone(),
        )
        .unwrap();
        let tokens: Vec<u32> = (0..4).collect();
        let ticket = rows.prefetch(0, PleHistory::new(7), &tokens).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        for (token, ids) in expected_ids.iter().enumerate() {
            for (head, &row_id) in ids.iter().enumerate() {
                let row = lease.row_bytes(token, head).unwrap();
                assert_eq!(u16::from_le_bytes([row[0], row[1]]) as u64, row_id);
            }
        }
        drop(lease);

        let mut distinct = std::collections::HashSet::new();
        let mut expected_bytes = 0usize;
        for &row in expected_ids.iter().flatten() {
            let key = rows.locate_row(row).unwrap().page_key();
            if distinct.insert(key) {
                expected_bytes += rows.inner.page_len(key).unwrap();
            }
        }
        let stats = rows.cache_stats();
        assert_eq!(stats.read_bytes as usize, expected_bytes);
        assert!(stats.reads as usize <= distinct.len());
        assert_eq!(source.reads.load(Ordering::Acquire) as u64, stats.reads);

        let needed = expected_ids.iter().flatten().count() * PLE_ROW_BYTES;
        assert!(
            expected_bytes <= needed * 16,
            "requested {needed} bytes and read {expected_bytes} bytes"
        );
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn direct_fill_handles_partial_pages_duplicates_and_cache_pressure() {
        assert_eq!(PLE_PAGE_BYTES % PLE_ROW_BYTES, 0);
        assert_eq!(PLE_STAGING_BYTES % PLE_ROW_BYTES, 0);

        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = PleRows::from_test_source_with_page_rows(
            metadata_with_head_size(8, 128),
            64,
            5,
            source.clone(),
        )
        .unwrap();
        {
            let mut state = rows
                .inner
                .state
                .lock()
                .expect("PLE test state mutex poisoned");
            state.cache.capacity_bytes = PLE_ROW_BYTES * 10;
        }

        let ticket = rows.prefetch(0, PleHistory::new(999), &[0, 0]).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        assert_eq!(lease.row_ids(), expected_ids.as_slice());
        for token in 0..2 {
            for head in 0..PLE_ROWS_PER_TOKEN {
                let row = lease.row_bytes(token, head).unwrap();
                let value = u16::from_le_bytes([row[0], row[1]]) as u64;
                assert_eq!(value, lease.row_ids()[token][head]);
            }
        }
        let first_stats = rows.cache_stats();
        assert!(first_stats.reads > 1);
        assert_eq!(
            source.reads.load(Ordering::Acquire) as u64,
            first_stats.reads
        );
        assert!(first_stats.resident_bytes <= PLE_ROW_BYTES * 10);
        drop(lease);

        let ticket = rows.prefetch(0, PleHistory::new(999), &[1]).unwrap();
        let expected_ids = ticket.row_ids().unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();
        for (head, &row_id) in lease.row_ids()[0].iter().enumerate() {
            let row = lease.row_bytes(0, head).unwrap();
            let value = u16::from_le_bytes([row[0], row[1]]) as u64;
            assert_eq!(value, row_id);
        }
        assert_eq!(lease.row_ids(), expected_ids.as_slice());
        drop(lease);

        let stats = rows.cache_stats();
        assert!(stats.reads > first_stats.reads);
        assert_eq!(source.reads.load(Ordering::Acquire) as u64, stats.reads);
        assert!(stats.cache_hits > 0);
        assert!(stats.cache_misses > 0);
        assert!(stats.evictions > 0);
        assert!(stats.resident_bytes <= PLE_ROW_BYTES * 10);
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn epoch_reset_stales_lease_and_enforces_monotonic_epochs() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = PleRows::from_test_source(metadata(128), 64, source).unwrap();
        rows.begin_epoch(0).unwrap();
        let ticket = rows.prefetch(0, PleHistory::new(99), &[0]).unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();

        std::thread::scope(|scope| {
            let resetter = scope.spawn(|| rows.reset_epoch(Duration::from_secs(1)));
            while rows.current_epoch() != 1 {
                std::thread::yield_now();
            }
            assert!(matches!(
                lease.validate(),
                Err(PleRowsError::LeaseStale { .. })
            ));
            assert!(matches!(
                lease.as_bytes(),
                Err(PleRowsError::LeaseStale { .. })
            ));
            let mut destination =
                vec![0u8; lease.token_count() * PLE_ROWS_PER_TOKEN * PLE_ROW_BYTES];
            assert!(matches!(
                lease.stage_into(&mut destination),
                Err(PleRowsError::LeaseStale { .. })
            ));
            assert!(matches!(
                lease.validate_after_upload(),
                Err(PleRowsError::LeaseStale { .. })
            ));
            drop(lease);
            assert_eq!(resetter.join().unwrap().unwrap(), 1);
        });

        assert!(matches!(
            rows.begin_epoch(1),
            Err(PleRowsError::EpochNotMonotonic {
                requested: 1,
                current: 1
            })
        ));
        rows.begin_epoch(2).unwrap();
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn cancel_releases_queued_ticket_slot_while_lease_holds_staging() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = PleRows::from_test_source(metadata(128), 64, source).unwrap();
        let first = rows.prefetch(0, PleHistory::new(99), &[0]).unwrap();
        let lease = rows.wait_completed_lease(&first).unwrap();

        let mut queued = Vec::with_capacity(PLE_READER_QUEUE_CAPACITY);
        for _ in 0..PLE_READER_QUEUE_CAPACITY {
            queued.push(rows.prefetch(0, PleHistory::new(99), &[0]).unwrap());
        }
        assert!(matches!(
            rows.prefetch(0, PleHistory::new(99), &[0]),
            Err(PleRowsError::QueueFull)
        ));
        rows.cancel(&queued[0]).unwrap();
        let replacement = rows.prefetch(0, PleHistory::new(99), &[0]).unwrap();

        drop(queued);
        drop(replacement);
        drop(lease);
        assert!(rows.unload().unwrap().is_clean());
    }

    #[test]
    fn unload_waits_for_a_live_lease_before_releasing_the_reader() {
        let source = Arc::new(MemoryRowSource {
            shards: rows_source(2, 64),
            reads: AtomicUsize::new(0),
            fail: None,
        });
        let rows = PleRows::from_test_source(metadata(128), 64, source).unwrap();
        let ticket = rows.prefetch(0, PleHistory::new(99), &[0]).unwrap();
        let lease = rows.wait_completed_lease(&ticket).unwrap();

        let unload = std::thread::spawn(move || rows.unload());
        for _ in 0..128 {
            assert!(!unload.is_finished());
            std::thread::yield_now();
        }
        drop(lease);
        assert!(unload.join().unwrap().unwrap().is_clean());
    }
}
