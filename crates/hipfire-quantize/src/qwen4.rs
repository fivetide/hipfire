// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// Copyright (c) 2026 Nick Woolmer
// hipfire — see LICENSE and NOTICE in the project root.

//! Streaming producer for the native Qwen4/Qwen3.8-Flash-Next artifact.
//!
//! This path is deliberately separate from the legacy model recipes.  It reads
//! safetensors payloads with positional, bounded reads, quantizes complete
//! logical rows, and writes the HFQM index before any large payload is copied.
//! No source tensor (and in particular no stacked expert tensor or PLE shard) is
//! ever collected into one `Vec<u8>`.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Map, Value};

use crate::quant_fwht::{gen_fwht_signs, quantize_mq4g256v2};
use crate::quant_q4::quantize_q8f16;
use hipfire_quantize::float16::bf16_to_f32;

/// Native Qwen4 architecture ID reserved by the runtime registry.
pub(crate) const QWEN4_ARCH_ID: u32 = 16;

const QWEN4_PLE_VERSION: u32 = 1;
const PLE_SHARD_COUNT: usize = 128;
const PLE_ROWS_PER_SHARD: u64 = 2_500_012;
const PLE_ROW_WIDTH: u64 = 160;
const PLE_HEAD_COUNT: usize = 16;
const PLE_NGRAM_SIZE: u32 = 3;
const ROUTED_EXPERTS: u64 = 512;
const ROUTER_TOP_K: u32 = 10;
const HIDDEN_WIDTH: u64 = 2_560;
const GATE_UP_INTERMEDIATE: u64 = 1_280;
const DOWN_INTERMEDIATE: u64 = 640;
const MQ4_GROUP_SIZE: u64 = 256;
const Q8_GROUP_SIZE: u64 = 32;
const Q8_GROUP_BYTES: u64 = 34;
const MQ4_GROUP_BYTES: u64 = 136;
/// Raw signed-I64 records are a distinct HFQ type.  TidI32 is not a valid
/// representation for Qwen4's hash metadata.
const QWEN4_I64_QUANT_TYPE: u8 = 52;
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_CONFIG_BYTES: u64 = 16 * 1024 * 1024;
const MAX_REOPEN_REGION_BYTES: u64 = 64 * 1024 * 1024;
/// Default bounded source row chunk.  A gate/up chunk is about 2.5 MiB BF16
/// and 5 MiB F32; the quantizer's output is smaller still.
pub(crate) const DEFAULT_ROW_CHUNK: usize = 256;
const MAX_CHUNK_BYTES: u64 = 32 * 1024 * 1024;
/// Do not let an accidental CLI value turn a row stream into a tensor buffer.
const MAX_ROW_CHUNK: usize = 1_024;

#[derive(Debug)]
pub(crate) struct Qwen4Options<'a> {
    pub(crate) input: &'a Path,
    pub(crate) output: &'a Path,
    pub(crate) row_chunk: usize,
}

impl<'a> Qwen4Options<'a> {
    pub(crate) fn new(input: &'a Path, output: &'a Path) -> Self {
        Self {
            input,
            output,
            row_chunk: DEFAULT_ROW_CHUNK,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Qwen4Summary {
    pub(crate) entries: usize,
    pub(crate) resident_entries: usize,
    pub(crate) expert_entries: usize,
    pub(crate) ple_shards: usize,
    pub(crate) resident_bytes: u64,
    pub(crate) external_ple_bytes: u64,
}

#[derive(Debug)]
pub(crate) enum Qwen4Error {
    Io {
        context: String,
        source: io::Error,
    },
    Json {
        context: String,
        source: serde_json::Error,
    },
    Invalid(String),
}

impl Qwen4Error {
    fn io(context: impl Into<String>, source: io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    fn json(context: impl Into<String>, source: serde_json::Error) -> Self {
        Self::Json {
            context: context.into(),
            source,
        }
    }

    fn into_io(self) -> io::Error {
        let message = self.to_string();
        match self {
            Self::Io { source, .. } => source,
            Self::Json { .. } | Self::Invalid(_) => {
                io::Error::new(io::ErrorKind::InvalidData, message)
            }
        }
    }
}

impl std::fmt::Display for Qwen4Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { context, source } => write!(f, "{context}: {source}"),
            Self::Json { context, source } => write!(f, "{context}: {source}"),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Qwen4Error {}

/// CLI-facing producer entry point.  The existing quantizer routes do not call
/// this function unless `--qwen4-flash-next` is explicitly selected.
pub(crate) fn run_cli(input: &Path, output: &Path) -> Result<Qwen4Summary, Qwen4Error> {
    write_qwen4_artifact(&Qwen4Options::new(input, output))
}

/// Build one transactional Qwen4 HFQM artifact.
///
/// The writer's index is planned entirely from safetensors headers first.  A
/// temporary file is written and reopened with the bounded plan reader before
/// the final rename, so a malformed source or short callback never publishes a
/// partially written candidate over an existing artifact.
pub(crate) fn write_qwen4_artifact(options: &Qwen4Options<'_>) -> Result<Qwen4Summary, Qwen4Error> {
    if options.row_chunk == 0 || options.row_chunk > MAX_ROW_CHUNK {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 row chunk must be in 1..={MAX_ROW_CHUNK}, got {}",
            options.row_chunk
        )));
    }
    let source = source_paths(options.input)?;
    let tensors = load_inventory(&source)?;
    let config = load_optional_config(&source)?;
    validate_config(config.as_ref())?;

    let plan = plan_entries(tensors, options.row_chunk)?;
    let ple_metadata = plan.ple_metadata.as_ref().ok_or_else(|| {
        Qwen4Error::Invalid("Qwen4 plan did not produce PLE metadata".to_string())
    })?;
    let metadata_json = build_metadata(config.as_ref(), &plan, ple_metadata)?;

    if plan.entries.len() > u32::MAX as usize {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 artifact has too many entries: {}",
            plan.entries.len()
        )));
    }
    let stream_entries: Vec<hipfire_runtime::hfq::HfqStreamEntry> = plan
        .entries
        .iter()
        .map(|entry| hipfire_runtime::hfq::HfqStreamEntry {
            name: entry.name.clone(),
            quant_type: entry.quant_type,
            shape: entry.shape.clone(),
            group_size: entry.group_size,
            data_len: entry.data_len,
        })
        .collect();

    let temporary = temporary_output_path(options.output);
    let _ = fs::remove_file(&temporary);
    let signs1 = gen_fwht_signs(42, 256);
    let signs2 = gen_fwht_signs(1042, 256);
    let write_result = hipfire_runtime::hfq::write_hfqm_package_streaming(
        &temporary,
        QWEN4_ARCH_ID,
        &metadata_json,
        &stream_entries,
        |index, writer| {
            stream_entry(
                &plan.entries[index],
                options.row_chunk,
                &signs1,
                &signs2,
                writer,
            )
            .map_err(Qwen4Error::into_io)
        },
    );
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(Qwen4Error::io(
            format!("write Qwen4 artifact {}", options.output.display()),
            error,
        ));
    }

    // Reopen without mmap'ing the large payload.  The plan reader checks the
    // header/index extents and qwen4_ple schema while retaining only metadata
    // and index-sized allocations.
    if let Err(error) = Qwen4ReopenPlan::open(&temporary) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    if let Err(error) = fs::rename(&temporary, options.output) {
        let _ = fs::remove_file(&temporary);
        return Err(Qwen4Error::io(
            format!("publish Qwen4 artifact {}", options.output.display()),
            error,
        ));
    }

    Ok(Qwen4Summary {
        entries: plan.entries.len(),
        resident_entries: plan.resident_entries,
        expert_entries: plan.expert_entries,
        ple_shards: PLE_SHARD_COUNT,
        resident_bytes: plan.resident_bytes,
        external_ple_bytes: plan.external_ple_bytes,
    })
}

fn temporary_output_path(output: &Path) -> PathBuf {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("qwen4.hfq");
    parent.join(format!(".{file_name}.qwen4-tmp-{}", std::process::id()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RemoteSpec {
    owner: String,
    repo: String,
    revision: String,
}

/// Parse the only remote source syntax accepted by the Qwen4 producer.
///
/// A revision is deliberately required to be a full commit ID.  Resolving
/// `main`, a tag, or a short SHA would make an artifact depend on mutable
/// repository state and is therefore never allowed here.
fn parse_remote_spec(input: &str) -> Result<Option<RemoteSpec>, Qwen4Error> {
    let Some(rest) = input.strip_prefix("hf://") else {
        return Ok(None);
    };
    let (repository, revision) = rest.split_once('@').ok_or_else(|| {
        Qwen4Error::Invalid(
            "Qwen4 remote input must be hf://OWNER/REPO@40_HEX_REVISION".to_string(),
        )
    })?;
    if repository.is_empty() || revision.is_empty() || revision.contains('@') {
        return Err(Qwen4Error::Invalid(
            "Qwen4 remote input must be hf://OWNER/REPO@40_HEX_REVISION".to_string(),
        ));
    }
    let mut components = repository.split('/');
    let owner = components.next().unwrap_or_default();
    let repo = components.next().unwrap_or_default();
    if components.next().is_some() || !valid_hf_component(owner) || !valid_hf_component(repo) {
        return Err(Qwen4Error::Invalid(format!(
            "invalid Qwen4 Hugging Face repository {repository:?}; expected OWNER/REPO"
        )));
    }
    if revision.len() != 40 || !revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 Hugging Face revision must be exactly 40 hexadecimal characters, got {revision:?}"
        )));
    }
    Ok(Some(RemoteSpec {
        owner: owner.to_string(),
        repo: repo.to_string(),
        revision: revision.to_string(),
    }))
}

fn valid_hf_component(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

struct RemoteSource {
    spec: RemoteSpec,
    base_url: String,
    agent: ureq::Agent,
    authorization: Option<String>,
}

impl std::fmt::Debug for RemoteSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteSource")
            .field("spec", &self.spec)
            .field("base_url", &self.base_url)
            .field("authenticated", &self.authorization.is_some())
            .finish()
    }
}

impl RemoteSource {
    fn new(spec: RemoteSpec) -> Result<Self, Qwen4Error> {
        let base_url = env::var("HIPFIRE_HF_BASE")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                env::var("HF_ENDPOINT")
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            })
            .unwrap_or_else(|| "https://huggingface.co".to_string())
            .trim_end_matches('/')
            .to_string();
        if !base_url.starts_with("https://") && !base_url.starts_with("http://") {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 Hugging Face endpoint must use http:// or https://, got {base_url:?}"
            )));
        }
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(24 * 60 * 60)))
            .timeout_recv_body(Some(Duration::from_secs(24 * 60 * 60)))
            .http_status_as_error(false)
            .build()
            .into();
        let authorization = env::var_os("HF_TOKEN")
            .and_then(|value| {
                let value = value.to_string_lossy().into_owned();
                (!value.is_empty()).then(|| format!("Bearer {value}"))
            })
            .or_else(|| {
                env::var_os("HUGGING_FACE_HUB_TOKEN").and_then(|value| {
                    let value = value.to_string_lossy().into_owned();
                    (!value.is_empty()).then(|| format!("Bearer {value}"))
                })
            });
        Ok(Self {
            spec,
            base_url,
            agent,
            authorization,
        })
    }

    fn url_for(&self, path: &str) -> String {
        format!(
            "{}/{}/{}/resolve/{}/{}",
            self.base_url, self.spec.owner, self.spec.repo, self.spec.revision, path
        )
    }

    fn read_json(&self, path: &str, max_bytes: u64) -> Result<Value, Qwen4Error> {
        validate_remote_path(path, None)?;
        let bytes = self.read_bounded(path, max_bytes)?;
        serde_json::from_slice(&bytes)
            .map_err(|error| Qwen4Error::json(format!("parse remote {path}"), error))
    }

    /// Read a bounded object with a Range request.  Even small JSON objects
    /// use 206 so a proxy cannot silently turn an unbounded download into a
    /// metadata request.
    fn read_bounded(&self, path: &str, max_bytes: u64) -> Result<Vec<u8>, Qwen4Error> {
        if max_bytes == 0 {
            return Err(Qwen4Error::Invalid(
                "Qwen4 bounded remote read must have non-zero capacity".to_string(),
            ));
        }
        let mut response = self.range_request(path, 0, max_bytes)?;
        let content_range = response
            .headers()
            .get("content-range")
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| {
                Qwen4Error::Invalid(format!(
                    "remote {path} response is missing a valid Content-Range"
                ))
            })?;
        let (start, end, total) = parse_content_range(content_range)?;
        let body_len = end
            .checked_sub(start)
            .and_then(|length| length.checked_add(1))
            .ok_or_else(|| {
                Qwen4Error::Invalid(format!("remote {path} Content-Range length overflows"))
            })?;
        if start != 0 || body_len > max_bytes || total != body_len {
            return Err(Qwen4Error::Invalid(format!(
                "remote {path} bounded range {content_range:?} is not the complete object within {} bytes",
                max_bytes
            )));
        }
        let announced = content_length(response.headers(), path)?;
        if announced != body_len {
            return Err(Qwen4Error::Invalid(format!(
                "remote {path} Content-Length {announced} disagrees with Content-Range length {body_len}"
            )));
        }
        let length = usize::try_from(body_len).map_err(|_| {
            Qwen4Error::Invalid(format!("remote {path} response does not fit usize"))
        })?;
        let mut bytes = vec![0u8; length];
        read_response_exact(&mut response, &mut bytes, path)?;
        Ok(bytes)
    }

    /// Read exactly `dst.len()` bytes from one immutable remote shard range.
    /// `expected_total` is omitted only for the initial eight-byte read used
    /// to discover a safetensors shard's total length.
    fn read_range(
        &self,
        path: &str,
        offset: u64,
        dst: &mut [u8],
        expected_total: Option<u64>,
    ) -> Result<u64, Qwen4Error> {
        if dst.is_empty() {
            return Err(Qwen4Error::Invalid(format!(
                "remote {path} requested an empty range"
            )));
        }
        validate_remote_path(path, Some("safetensors"))?;
        let length = u64::try_from(dst.len())
            .map_err(|_| Qwen4Error::Invalid(format!("remote {path} range is too large")))?;
        let mut response = self.range_request(path, offset, length)?;
        let content_range = response
            .headers()
            .get("content-range")
            .and_then(|value| value.to_str().ok())
            .ok_or_else(|| {
                Qwen4Error::Invalid(format!(
                    "remote {path} response is missing a valid Content-Range"
                ))
            })?;
        let (start, end, total) = parse_content_range(content_range)?;
        let body_len = end
            .checked_sub(start)
            .and_then(|length| length.checked_add(1))
            .ok_or_else(|| {
                Qwen4Error::Invalid(format!("remote {path} Content-Range length overflows"))
            })?;
        if start != offset
            || body_len != length
            || expected_total.is_some_and(|expected| total != expected)
        {
            let expected_total = expected_total
                .map(|value| format!(" / {value}"))
                .unwrap_or_default();
            return Err(Qwen4Error::Invalid(format!(
                "remote {path} returned range {content_range:?}, expected bytes {offset}-{}{expected_total}",
                offset.saturating_add(length).saturating_sub(1)
            )));
        }
        let announced = content_length(response.headers(), path)?;
        if announced != length {
            return Err(Qwen4Error::Invalid(format!(
                "remote {path} Content-Length {announced} disagrees with requested range length {length}"
            )));
        }
        read_response_exact(&mut response, dst, path)?;
        Ok(total)
    }

    fn range_request(
        &self,
        path: &str,
        offset: u64,
        length: u64,
    ) -> Result<ureq::http::Response<ureq::Body>, Qwen4Error> {
        if length == 0 {
            return Err(Qwen4Error::Invalid(format!(
                "remote {path} requested an empty range"
            )));
        }
        let end = offset
            .checked_add(length - 1)
            .ok_or_else(|| Qwen4Error::Invalid(format!("remote {path} range overflows")))?;
        let url = self.url_for(path);
        let mut request = self
            .agent
            .get(&url)
            .header("Range", &format!("bytes={offset}-{end}"));
        if let Some(authorization) = &self.authorization {
            request = request.header("Authorization", authorization);
        }
        let response = request
            .call()
            .map_err(|error| Qwen4Error::Invalid(format!("remote GET {url} failed: {error}")))?;
        if response.status().as_u16() != 206 {
            return Err(Qwen4Error::Invalid(format!(
                "remote GET {url} returned HTTP {}, expected 206 Partial Content",
                response.status().as_u16()
            )));
        }
        Ok(response)
    }

    fn enumerate_shards(&self) -> Result<Vec<String>, Qwen4Error> {
        let index = self.read_json("model.safetensors.index.json", MAX_CONFIG_BYTES)?;
        let weight_map = index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                Qwen4Error::Invalid(
                    "remote model.safetensors.index.json is missing an object weight_map"
                        .to_string(),
                )
            })?;
        let mut paths = BTreeSet::new();
        for shard in weight_map.values() {
            let shard = shard.as_str().ok_or_else(|| {
                Qwen4Error::Invalid(
                    "remote safetensors index weight_map contains a non-string shard".to_string(),
                )
            })?;
            if !shard.ends_with(".safetensors") {
                continue;
            }
            validate_remote_path(shard, Some("safetensors"))?;
            paths.insert(shard.to_string());
        }
        if paths.is_empty() {
            return Err(Qwen4Error::Invalid(
                "remote safetensors index references no .safetensors shards".to_string(),
            ));
        }
        Ok(paths.into_iter().collect())
    }
}

fn validate_remote_path(path: &str, required_suffix: Option<&str>) -> Result<(), Qwen4Error> {
    if path.is_empty()
        || path.starts_with('/')
        || path.contains('\\')
        || path.split('/').any(|part| {
            part.is_empty()
                || part == "."
                || part == ".."
                || !part
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        })
    {
        return Err(Qwen4Error::Invalid(format!(
            "remote Hugging Face path {path:?} is not a safe relative path"
        )));
    }
    if let Some(suffix) = required_suffix {
        if !path.ends_with(suffix) {
            return Err(Qwen4Error::Invalid(format!(
                "remote path {path:?} does not end with .{suffix}"
            )));
        }
    }
    Ok(())
}

fn parse_content_range(value: &str) -> Result<(u64, u64, u64), Qwen4Error> {
    let (unit, range) = value.split_once(' ').ok_or_else(|| {
        Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; expected bytes START-END/TOTAL"
        ))
    })?;
    if unit != "bytes" || range.contains(' ') {
        return Err(Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; expected bytes START-END/TOTAL"
        )));
    }
    let (offsets, total) = range.split_once('/').ok_or_else(|| {
        Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; expected bytes START-END/TOTAL"
        ))
    })?;
    if total == "*" || total.is_empty() || total.contains('/') {
        return Err(Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; total length must be explicit"
        )));
    }
    let (start, end) = offsets.split_once('-').ok_or_else(|| {
        Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; expected bytes START-END/TOTAL"
        ))
    })?;
    if start.is_empty() || end.is_empty() || end.contains('-') {
        return Err(Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; expected bytes START-END/TOTAL"
        )));
    }
    let start = start.parse::<u64>().map_err(|_| {
        Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; start is not an integer"
        ))
    })?;
    let end = end.parse::<u64>().map_err(|_| {
        Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; end is not an integer"
        ))
    })?;
    let total = total.parse::<u64>().map_err(|_| {
        Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; total is not an integer"
        ))
    })?;
    if start > end || total <= end {
        return Err(Qwen4Error::Invalid(format!(
            "invalid Content-Range {value:?}; require START <= END < TOTAL"
        )));
    }
    Ok((start, end, total))
}

fn content_length(headers: &ureq::http::HeaderMap, path: &str) -> Result<u64, Qwen4Error> {
    let value = headers
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            Qwen4Error::Invalid(format!("remote {path} response is missing Content-Length"))
        })?;
    value.parse::<u64>().map_err(|_| {
        Qwen4Error::Invalid(format!(
            "remote {path} response has invalid Content-Length {value:?}"
        ))
    })
}

fn read_response_exact(
    response: &mut ureq::http::Response<ureq::Body>,
    dst: &mut [u8],
    path: &str,
) -> Result<(), Qwen4Error> {
    let mut reader = response.body_mut().as_reader();
    let mut filled = 0usize;
    while filled < dst.len() {
        let count = reader
            .read(&mut dst[filled..])
            .map_err(|error| Qwen4Error::io(format!("read remote {path} body"), error))?;
        if count == 0 {
            return Err(Qwen4Error::Invalid(format!(
                "remote {path} body ended after {filled} bytes, expected {}",
                dst.len()
            )));
        }
        filled += count;
    }
    let mut extra = [0u8; 1];
    let count = reader
        .read(&mut extra)
        .map_err(|error| Qwen4Error::io(format!("read remote {path} body"), error))?;
    if count != 0 {
        return Err(Qwen4Error::Invalid(format!(
            "remote {path} body exceeded the announced {} bytes",
            dst.len()
        )));
    }
    Ok(())
}

enum SourceSet {
    Local {
        paths: Vec<PathBuf>,
    },
    Remote {
        source: Arc<RemoteSource>,
        paths: Vec<String>,
    },
}

enum SourceKind {
    Local {
        file: Arc<File>,
    },
    Remote {
        source: Arc<RemoteSource>,
        path: String,
    },
}

struct SourceShard {
    path: PathBuf,
    kind: SourceKind,
    file_len: u64,
}

impl Clone for SourceShard {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            kind: match &self.kind {
                SourceKind::Local { file } => SourceKind::Local {
                    file: Arc::clone(file),
                },
                SourceKind::Remote { source, path } => SourceKind::Remote {
                    source: Arc::clone(source),
                    path: path.clone(),
                },
            },
            file_len: self.file_len,
        }
    }
}

impl std::fmt::Debug for SourceShard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SourceShard")
            .field("path", &self.path)
            .field("file_len", &self.file_len)
            .finish()
    }
}

impl SourceShard {
    fn read_exact_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), Qwen4Error> {
        let end = offset.checked_add(dst.len() as u64).ok_or_else(|| {
            Qwen4Error::Invalid(format!("{} source range overflows", self.path.display()))
        })?;
        if end > self.file_len {
            return Err(Qwen4Error::Invalid(format!(
                "{} source range [{offset}, {end}) exceeds {} bytes",
                self.path.display(),
                self.file_len
            )));
        }
        match &self.kind {
            SourceKind::Local { file } => read_exact_at(file, offset, dst)
                .map_err(|error| Qwen4Error::io(format!("read {}", self.path.display()), error)),
            SourceKind::Remote { source, path } => {
                source.read_range(path, offset, dst, Some(self.file_len))?;
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug)]
struct SourceTensor {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    data_start: u64,
    data_end: u64,
    shard: Arc<SourceShard>,
}

impl SourceTensor {
    fn data_len(&self) -> u64 {
        self.data_end - self.data_start
    }

    fn read_range(&self, relative: u64, dst: &mut [u8]) -> Result<(), Qwen4Error> {
        let len = dst.len() as u64;
        let end = relative
            .checked_add(len)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{} range overflows", self.name)))?;
        if end > self.data_len() {
            return Err(Qwen4Error::Invalid(format!(
                "{} range [{relative}, {end}) exceeds tensor payload {}",
                self.name,
                self.data_len()
            )));
        }
        let offset = self
            .data_start
            .checked_add(relative)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{} file offset overflows", self.name)))?;
        self.shard.read_exact_at(offset, dst)
    }
}

fn source_paths(input: &Path) -> Result<SourceSet, Qwen4Error> {
    if let Some(raw) = input.to_str() {
        if let Some(spec) = parse_remote_spec(raw)? {
            let source = Arc::new(RemoteSource::new(spec)?);
            let paths = source.enumerate_shards()?;
            return Ok(SourceSet::Remote { source, paths });
        }
    }
    let metadata =
        fs::metadata(input).map_err(|error| Qwen4Error::io(input.display().to_string(), error))?;
    if metadata.is_file() {
        if input.extension().and_then(|ext| ext.to_str()) != Some("safetensors") {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 input file must have .safetensors extension: {}",
                input.display()
            )));
        }
        return Ok(SourceSet::Local {
            paths: vec![input.to_path_buf()],
        });
    }
    if !metadata.is_dir() {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 input is neither a directory nor a safetensors file: {}",
            input.display()
        )));
    }
    let mut paths = Vec::new();
    let entries = fs::read_dir(input)
        .map_err(|error| Qwen4Error::io(format!("read Qwen4 input {}", input.display()), error))?;
    for entry in entries {
        let entry = entry.map_err(|error| Qwen4Error::io("read Qwen4 directory entry", error))?;
        let path = entry.path();
        if entry
            .file_type()
            .map_err(|error| Qwen4Error::io(format!("stat {}", path.display()), error))?
            .is_file()
            && path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
        {
            paths.push(path);
        }
    }
    paths.sort_by(|left, right| left.as_os_str().cmp(right.as_os_str()));
    if paths.is_empty() {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 input directory contains no .safetensors shards: {}",
            input.display()
        )));
    }
    Ok(SourceSet::Local { paths })
}

fn load_inventory(source: &SourceSet) -> Result<Vec<SourceTensor>, Qwen4Error> {
    let mut tensors = HashMap::<String, SourceTensor>::new();
    match source {
        SourceSet::Local { paths } => {
            for path in paths {
                for tensor in parse_safetensors_header(path)? {
                    if tensors.insert(tensor.name.clone(), tensor).is_some() {
                        return Err(Qwen4Error::Invalid(format!(
                            "duplicate safetensors tensor name in Qwen4 input: {}",
                            path.display()
                        )));
                    }
                }
            }
        }
        SourceSet::Remote { source, paths } => {
            for path in paths {
                for tensor in parse_remote_safetensors_header(Arc::clone(source), path)? {
                    if tensors.insert(tensor.name.clone(), tensor).is_some() {
                        return Err(Qwen4Error::Invalid(format!(
                            "duplicate safetensors tensor name in remote shard {path}"
                        )));
                    }
                }
            }
        }
    }
    let mut tensors: Vec<_> = tensors.into_values().collect();
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(tensors)
}

fn parse_safetensors_header(path: &Path) -> Result<Vec<SourceTensor>, Qwen4Error> {
    let file =
        File::open(path).map_err(|error| Qwen4Error::io(path.display().to_string(), error))?;
    let file_len = file
        .metadata()
        .map_err(|error| Qwen4Error::io(format!("stat {}", path.display()), error))?
        .len();
    let mut length_bytes = [0u8; 8];
    read_exact_at(&file, 0, &mut length_bytes).map_err(|error| {
        Qwen4Error::io(
            format!("read safetensors header length {}", path.display()),
            error,
        )
    })?;
    let shard = Arc::new(SourceShard {
        path: path.to_path_buf(),
        kind: SourceKind::Local {
            file: Arc::new(file),
        },
        file_len,
    });
    parse_safetensors_header_from_shard(shard, length_bytes)
}

fn parse_remote_safetensors_header(
    source: Arc<RemoteSource>,
    path: &str,
) -> Result<Vec<SourceTensor>, Qwen4Error> {
    validate_remote_path(path, Some("safetensors"))?;
    let mut length_bytes = [0u8; 8];
    let file_len = source.read_range(path, 0, &mut length_bytes, None)?;
    let shard = Arc::new(SourceShard {
        path: PathBuf::from(source.url_for(path)),
        kind: SourceKind::Remote {
            source: Arc::clone(&source),
            path: path.to_string(),
        },
        file_len,
    });
    parse_safetensors_header_from_shard(shard, length_bytes)
}

fn parse_safetensors_header_from_shard(
    shard: Arc<SourceShard>,
    length_bytes: [u8; 8],
) -> Result<Vec<SourceTensor>, Qwen4Error> {
    let path = shard.path.display().to_string();
    let file_len = shard.file_len;
    let header_len = u64::from_le_bytes(length_bytes);
    if header_len == 0 || header_len > MAX_HEADER_BYTES {
        return Err(Qwen4Error::Invalid(format!(
            "{path} safetensors header length {header_len} is outside 1..={MAX_HEADER_BYTES}"
        )));
    }
    let header_end = 8u64
        .checked_add(header_len)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{path} header offset overflows")))?;
    if header_end > file_len {
        return Err(Qwen4Error::Invalid(format!(
            "{path} safetensors header is truncated: end {header_end}, file {file_len}"
        )));
    }
    let header_len_usize = usize::try_from(header_len)
        .map_err(|_| Qwen4Error::Invalid(format!("{path} header is too large")))?;
    let mut header_bytes = vec![0u8; header_len_usize];
    shard.read_exact_at(8, &mut header_bytes)?;
    let header: Value = serde_json::from_slice(&header_bytes)
        .map_err(|error| Qwen4Error::json(format!("parse safetensors header {path}"), error))?;
    let object = header.as_object().ok_or_else(|| {
        Qwen4Error::Invalid(format!("{path} safetensors header is not an object"))
    })?;
    let mut tensors = Vec::with_capacity(object.len().saturating_sub(1));
    for (name, descriptor) in object {
        if name == "__metadata__" {
            continue;
        }
        let descriptor = descriptor.as_object().ok_or_else(|| {
            Qwen4Error::Invalid(format!("{name} descriptor in {path} is not an object"))
        })?;
        let dtype = descriptor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} is missing a dtype")))?
            .to_string();
        let shape = descriptor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} is missing a shape")))?
            .iter()
            .map(|dimension| {
                dimension.as_u64().ok_or_else(|| {
                    Qwen4Error::Invalid(format!("{name} has a non-integer shape dimension"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let offsets = descriptor
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} is missing data_offsets")))?;
        if offsets.len() != 2 {
            return Err(Qwen4Error::Invalid(format!(
                "{name} data_offsets must have exactly two elements"
            )));
        }
        let relative_start = offsets[0]
            .as_u64()
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} has a non-integer start offset")))?;
        let relative_end = offsets[1]
            .as_u64()
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} has a non-integer end offset")))?;
        if relative_start > relative_end {
            return Err(Qwen4Error::Invalid(format!(
                "{name} data_offsets are reversed"
            )));
        }
        let data_start = header_end
            .checked_add(relative_start)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} data start overflows")))?;
        let data_end = header_end
            .checked_add(relative_end)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} data end overflows")))?;
        if data_end > file_len {
            return Err(Qwen4Error::Invalid(format!(
                "{name} payload [{data_start}, {data_end}) exceeds {file_len} bytes"
            )));
        }
        tensors.push(SourceTensor {
            name: name.clone(),
            dtype,
            shape,
            data_start,
            data_end,
            shard: Arc::clone(&shard),
        });
    }
    Ok(tensors)
}

fn load_optional_config(source: &SourceSet) -> Result<Option<Value>, Qwen4Error> {
    match source {
        SourceSet::Remote { source, .. } => {
            Ok(Some(source.read_json("config.json", MAX_CONFIG_BYTES)?))
        }
        SourceSet::Local { paths } => {
            let first = paths.first().ok_or_else(|| {
                Qwen4Error::Invalid("Qwen4 local source has no safetensors shards".to_string())
            })?;
            let path = first
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("config.json");
            let metadata = match fs::metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
                Err(error) => {
                    return Err(Qwen4Error::io(format!("stat {}", path.display()), error))
                }
            };
            if metadata.len() > MAX_CONFIG_BYTES {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 config {} is larger than {MAX_CONFIG_BYTES} bytes",
                    path.display()
                )));
            }
            let mut file = File::open(&path)
                .map_err(|error| Qwen4Error::io(path.display().to_string(), error))?;
            let mut bytes = Vec::with_capacity(metadata.len() as usize);
            file.read_to_end(&mut bytes)
                .map_err(|error| Qwen4Error::io(format!("read {}", path.display()), error))?;
            let value = serde_json::from_slice(&bytes)
                .map_err(|error| Qwen4Error::json(format!("parse {}", path.display()), error))?;
            Ok(Some(value))
        }
    }
}

fn validate_config(config: Option<&Value>) -> Result<(), Qwen4Error> {
    let Some(config) = config else { return Ok(()) };
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .or_else(|| {
            config
                .get("text_config")
                .and_then(Value::as_object)
                .and_then(|text| text.get("model_type"))
                .and_then(Value::as_str)
        });
    if let Some(model_type) = model_type {
        if model_type != "qwen4_exp" && model_type != "qwen4_exp_text" {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 producer received model_type {model_type:?}, expected qwen4_exp"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum I64Role {
    Multipliers,
    VocabSizes,
    Offsets,
}

fn i64_role(name: &str) -> Option<I64Role> {
    if name.ends_with(".layer_multipliers") {
        Some(I64Role::Multipliers)
    } else if name.ends_with(".ngram_heads_vocab_sizes") {
        Some(I64Role::VocabSizes)
    } else if name.ends_with(".ngram_heads_offsets") {
        Some(I64Role::Offsets)
    } else {
        None
    }
}

fn ple_shard_index(name: &str) -> Option<usize> {
    let marker = ".ngram_embedding.shard_";
    let start = name.find(marker)? + marker.len();
    let rest = &name[start..];
    let digits_end = rest.find(".weight")?;
    if digits_end == 0 || &rest[digits_end..] != ".weight" {
        return None;
    }
    let digits = &rest[..digits_end];
    if !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn is_ple_shard_name(name: &str) -> bool {
    name.contains(".ngram_embedding.shard_")
}

fn is_vision_tensor(name: &str) -> bool {
    [
        "model.visual.",
        "model.vision_tower.",
        "model.vision_projection.",
        "model.multi_modal_projector.",
        "vision_tower.",
        "visual.",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpertKind {
    GateUp,
    Down,
}

fn expert_kind(name: &str) -> Option<ExpertKind> {
    if name.ends_with(".mlp.experts.gate_up_proj")
        || name.ends_with(".mlp.experts.gate_up_proj.weight")
    {
        Some(ExpertKind::GateUp)
    } else if name.ends_with(".mlp.experts.down_proj")
        || name.ends_with(".mlp.experts.down_proj.weight")
    {
        Some(ExpertKind::Down)
    } else {
        None
    }
}

fn checked_product(values: &[u64], what: &str) -> Result<u64, Qwen4Error> {
    values.iter().try_fold(1u64, |product, &value| {
        product
            .checked_mul(value)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{what} dimensions overflow u64")))
    })
}

fn checked_bf16_bytes(shape: &[u64], what: &str) -> Result<u64, Qwen4Error> {
    checked_product(shape, what)?
        .checked_mul(2)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{what} BF16 byte length overflows u64")))
}

fn shape_u32(shape: &[u64], name: &str) -> Result<Vec<u32>, Qwen4Error> {
    if shape.len() > u8::MAX as usize {
        return Err(Qwen4Error::Invalid(format!(
            "{name} has {} dimensions, HFQM supports at most {}",
            shape.len(),
            u8::MAX
        )));
    }
    shape
        .iter()
        .map(|&dimension| {
            u32::try_from(dimension).map_err(|_| {
                Qwen4Error::Invalid(format!("{name} dimension {dimension} does not fit u32"))
            })
        })
        .collect()
}

fn validate_expert_shape(
    tensor: &SourceTensor,
    kind: ExpertKind,
) -> Result<(u64, u64), Qwen4Error> {
    let expected = match kind {
        ExpertKind::GateUp => [ROUTED_EXPERTS, GATE_UP_INTERMEDIATE, HIDDEN_WIDTH],
        ExpertKind::Down => [ROUTED_EXPERTS, HIDDEN_WIDTH, DOWN_INTERMEDIATE],
    };
    if tensor.shape.as_slice() != expected {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {:?}, expected {:?}; Qwen4 never pads expert K",
            tensor.name, tensor.shape, expected
        )));
    }
    let rows = tensor.shape[0]
        .checked_mul(tensor.shape[1])
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} row count overflows", tensor.name)))?;
    let k = tensor.shape[2];
    let expected_bytes = rows
        .checked_mul(k)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} byte length overflows", tensor.name)))?;
    if tensor.data_len() != expected_bytes {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {} payload bytes, expected {}",
            tensor.name,
            tensor.data_len(),
            expected_bytes
        )));
    }
    Ok((rows, k))
}

fn quantized_data_len(kind: ExpertKind, rows: u64, k: u64) -> Result<u64, Qwen4Error> {
    let (group_size, group_bytes) = match kind {
        ExpertKind::GateUp => (MQ4_GROUP_SIZE, MQ4_GROUP_BYTES),
        ExpertKind::Down => (Q8_GROUP_SIZE, Q8_GROUP_BYTES),
    };
    if k % group_size != 0 {
        return Err(Qwen4Error::Invalid(format!(
            "expert K={k} is not aligned to quantizer group {group_size}"
        )));
    }
    rows.checked_mul(k / group_size)
        .and_then(|groups| groups.checked_mul(group_bytes))
        .ok_or_else(|| {
            Qwen4Error::Invalid("quantized expert byte length overflows u64".to_string())
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EntryKind {
    Bf16,
    GateUp,
    Down,
    Ple,
    I64(I64Role),
}

#[derive(Clone, Debug)]
struct PlannedEntry {
    source: SourceTensor,
    name: String,
    quant_type: u8,
    shape: Vec<u32>,
    group_size: u32,
    data_len: u64,
    kind: EntryKind,
}
#[derive(Clone, Debug)]
struct PleMetadata {
    multipliers: Vec<i64>,
    vocab_sizes: Vec<i64>,
    prefix_offsets: Vec<i64>,
    source_names: Vec<String>,
    source_shapes: Vec<Vec<u64>>,
    metadata_names: Vec<String>,
    metadata_shapes: Vec<Vec<u64>>,
}

#[derive(Debug)]
struct EntryPlan {
    entries: Vec<PlannedEntry>,
    ple_metadata: Option<PleMetadata>,
    row_chunk: usize,
    resident_entries: usize,
    expert_entries: usize,
    resident_bytes: u64,
    external_ple_bytes: u64,
}

fn read_i64_array(tensor: &SourceTensor, expected_len: usize) -> Result<Vec<i64>, Qwen4Error> {
    if tensor.dtype != "I64" {
        return Err(Qwen4Error::Invalid(format!(
            "{} has dtype {}, expected I64 metadata",
            tensor.name, tensor.dtype
        )));
    }
    if tensor.shape != [expected_len as u64] {
        return Err(Qwen4Error::Invalid(format!(
            "{} has shape {:?}, expected [{expected_len}]",
            tensor.name, tensor.shape
        )));
    }
    let expected_bytes = (expected_len as u64)
        .checked_mul(8)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} I64 byte length overflows", tensor.name)))?;
    if tensor.data_len() != expected_bytes {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {} payload bytes, expected {}",
            tensor.name,
            tensor.data_len(),
            expected_bytes
        )));
    }
    let mut bytes = vec![0u8; expected_bytes as usize];
    tensor.read_range(0, &mut bytes)?;
    Ok(bytes
        .chunks_exact(8)
        .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("chunks_exact(8)")))
        .collect())
}

fn is_prime(value: i64) -> bool {
    if value < 2 {
        return false;
    }
    if value % 2 == 0 {
        return value == 2;
    }
    let mut divisor = 3i64;
    while divisor <= value / divisor {
        if value % divisor == 0 {
            return false;
        }
        divisor += 2;
    }
    true
}

fn validate_ple_metadata(
    multipliers: &[i64],
    vocab_sizes: &[i64],
    prefix_offsets: &[i64],
) -> Result<u64, Qwen4Error> {
    if multipliers.len() != 3 {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 PLE multipliers must have length 3, got {}",
            multipliers.len()
        )));
    }
    if vocab_sizes.len() != PLE_HEAD_COUNT || prefix_offsets.len() != PLE_HEAD_COUNT {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 PLE metadata requires {PLE_HEAD_COUNT} vocab sizes and offsets"
        )));
    }
    let mut valid_rows = 0u64;
    for (index, &vocab_size) in vocab_sizes.iter().enumerate() {
        if vocab_size <= 0 || !is_prime(vocab_size) {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 PLE vocabulary size at head {index} is not a positive prime: {vocab_size}"
            )));
        }
        valid_rows = valid_rows.checked_add(vocab_size as u64).ok_or_else(|| {
            Qwen4Error::Invalid("Qwen4 PLE valid row count overflows".to_string())
        })?;
        let expected_offset = if index == 0 {
            0
        } else {
            prefix_offsets[index - 1]
                .checked_add(vocab_sizes[index - 1])
                .ok_or_else(|| {
                    Qwen4Error::Invalid("Qwen4 PLE prefix offset overflows i64".to_string())
                })?
        };
        if prefix_offsets[index] != expected_offset {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 PLE offset {index} is {}, expected {expected_offset}",
                prefix_offsets[index]
            )));
        }
    }
    if prefix_offsets[0] != 0 {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 PLE first prefix offset is {}, expected 0",
            prefix_offsets[0]
        )));
    }
    let final_end = prefix_offsets[PLE_HEAD_COUNT - 1]
        .checked_add(vocab_sizes[PLE_HEAD_COUNT - 1])
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 PLE final offset overflows i64".to_string()))?;
    if final_end < 0 || final_end as u64 != valid_rows {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 PLE final offset {final_end} does not equal valid rows {valid_rows}"
        )));
    }
    let physical_rows = (PLE_SHARD_COUNT as u64)
        .checked_mul(PLE_ROWS_PER_SHARD)
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 PLE physical row count overflows".to_string()))?;
    if valid_rows > physical_rows {
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 PLE valid rows {valid_rows} exceed physical rows {physical_rows}"
        )));
    }
    Ok(valid_rows)
}

fn plan_entries(tensors: Vec<SourceTensor>, row_chunk: usize) -> Result<EntryPlan, Qwen4Error> {
    let mut metadata_sources: BTreeMap<I64Role, SourceTensor> = BTreeMap::new();
    let mut ple_sources: BTreeMap<usize, SourceTensor> = BTreeMap::new();
    let mut metadata_entries = Vec::new();
    let mut resident = Vec::new();
    let mut expert_entries = 0usize;
    let mut gate_up_count = 0usize;
    let mut down_count = 0usize;
    let mut mtp_gate_up = false;
    let mut mtp_down = false;

    for tensor in tensors {
        if let Some(role) = i64_role(&tensor.name) {
            if tensor.dtype != "I64" {
                return Err(Qwen4Error::Invalid(format!(
                    "{} has dtype {}, expected I64 metadata",
                    tensor.name, tensor.dtype
                )));
            }
            if metadata_sources.insert(role, tensor.clone()).is_some() {
                return Err(Qwen4Error::Invalid(format!(
                    "duplicate Qwen4 I64 metadata role {role:?}"
                )));
            }
            continue;
        }
        if tensor.dtype == "I64" {
            return Err(Qwen4Error::Invalid(format!(
                "unknown Qwen4 I64 tensor {}; refusing TidI32 or implicit casts",
                tensor.name
            )));
        }
        if is_ple_shard_name(&tensor.name) {
            let index = ple_shard_index(&tensor.name).ok_or_else(|| {
                Qwen4Error::Invalid(format!("malformed Qwen4 PLE shard name {}", tensor.name))
            })?;
            if index >= PLE_SHARD_COUNT {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 PLE shard index {index} is outside 0..{}",
                    PLE_SHARD_COUNT - 1
                )));
            }
            if tensor.dtype != "BF16" || tensor.shape != [PLE_ROWS_PER_SHARD, PLE_ROW_WIDTH] {
                return Err(Qwen4Error::Invalid(format!(
                    "{} must be BF16 [{PLE_ROWS_PER_SHARD},{PLE_ROW_WIDTH}], got {} {:?}",
                    tensor.name, tensor.dtype, tensor.shape
                )));
            }
            let expected_bytes = checked_bf16_bytes(&tensor.shape, &tensor.name)?;
            if tensor.data_len() != expected_bytes {
                return Err(Qwen4Error::Invalid(format!(
                    "{} has {} payload bytes, expected {}",
                    tensor.name,
                    tensor.data_len(),
                    expected_bytes
                )));
            }
            if ple_sources.insert(index, tensor).is_some() {
                return Err(Qwen4Error::Invalid(format!(
                    "duplicate Qwen4 PLE shard index {index}"
                )));
            }
            continue;
        }
        if is_vision_tensor(&tensor.name) {
            // The Qwen4 text artifact intentionally omits vision tensors.  Do
            // not let a vision BF16 tensor become an accidental text weight.
            continue;
        }
        if tensor.dtype != "BF16" {
            return Err(Qwen4Error::Invalid(format!(
                "{} has dtype {}, expected BF16 or one of the typed Qwen4 I64 metadata arrays",
                tensor.name, tensor.dtype
            )));
        }
        let shape = shape_u32(&tensor.shape, &tensor.name)?;
        if let Some(expert_kind) = expert_kind(&tensor.name) {
            let (rows, k) = validate_expert_shape(&tensor, expert_kind)?;
            let expected_len = quantized_data_len(expert_kind, rows, k)?;
            let is_mtp = tensor.name.starts_with("mtp.") || tensor.name.contains(".mtp.");
            match expert_kind {
                ExpertKind::GateUp => {
                    gate_up_count += 1;
                    mtp_gate_up |= is_mtp;
                }
                ExpertKind::Down => {
                    down_count += 1;
                    mtp_down |= is_mtp;
                }
            }
            expert_entries += 1;
            resident.push(PlannedEntry {
                source: tensor,
                name: String::new(),
                quant_type: match expert_kind {
                    ExpertKind::GateUp => 44,
                    ExpertKind::Down => 3,
                },
                shape,
                group_size: match expert_kind {
                    ExpertKind::GateUp => MQ4_GROUP_SIZE as u32,
                    ExpertKind::Down => Q8_GROUP_SIZE as u32,
                },
                data_len: expected_len,
                kind: match expert_kind {
                    ExpertKind::GateUp => EntryKind::GateUp,
                    ExpertKind::Down => EntryKind::Down,
                },
            });
            continue;
        }
        let expected_len = checked_bf16_bytes(&tensor.shape, &tensor.name)?;
        if tensor.data_len() != expected_len {
            return Err(Qwen4Error::Invalid(format!(
                "{} has {} payload bytes, expected {}",
                tensor.name,
                tensor.data_len(),
                expected_len
            )));
        }
        resident.push(PlannedEntry {
            source: tensor,
            name: String::new(),
            quant_type: 16,
            shape,
            group_size: 0,
            data_len: expected_len,
            kind: EntryKind::Bf16,
        });
    }

    let multipliers = metadata_sources
        .get(&I64Role::Multipliers)
        .ok_or_else(|| {
            Qwen4Error::Invalid("missing Qwen4 layer_multipliers I64 tensor".to_string())
        })
        .and_then(|tensor| read_i64_array(tensor, 3))?;
    let vocab_sizes = metadata_sources
        .get(&I64Role::VocabSizes)
        .ok_or_else(|| {
            Qwen4Error::Invalid("missing Qwen4 ngram_heads_vocab_sizes I64 tensor".to_string())
        })
        .and_then(|tensor| read_i64_array(tensor, PLE_HEAD_COUNT))?;
    let prefix_offsets = metadata_sources
        .get(&I64Role::Offsets)
        .ok_or_else(|| {
            Qwen4Error::Invalid("missing Qwen4 ngram_heads_offsets I64 tensor".to_string())
        })
        .and_then(|tensor| read_i64_array(tensor, PLE_HEAD_COUNT))?;
    let _valid_rows = validate_ple_metadata(&multipliers, &vocab_sizes, &prefix_offsets)?;
    for role in [I64Role::Multipliers, I64Role::VocabSizes, I64Role::Offsets] {
        let source = metadata_sources
            .get(&role)
            .expect("metadata role presence was validated")
            .clone();
        let shape = shape_u32(&source.shape, &source.name)?;
        let data_len = (checked_product(&source.shape, &source.name)?)
            .checked_mul(8)
            .ok_or_else(|| {
                Qwen4Error::Invalid(format!("{} I64 payload length overflows", source.name))
            })?;
        if source.data_len() != data_len {
            return Err(Qwen4Error::Invalid(format!(
                "{} has {} payload bytes, expected {}",
                source.name,
                source.data_len(),
                data_len
            )));
        }
        metadata_entries.push(PlannedEntry {
            name: source.name.clone(),
            source,
            quant_type: QWEN4_I64_QUANT_TYPE,
            shape,
            group_size: 0,
            data_len,
            kind: EntryKind::I64(role),
        });
    }

    if ple_sources.len() != PLE_SHARD_COUNT {
        let missing: Vec<usize> = (0..PLE_SHARD_COUNT)
            .filter(|index| !ple_sources.contains_key(index))
            .collect();
        return Err(Qwen4Error::Invalid(format!(
            "Qwen4 requires all {PLE_SHARD_COUNT} PLE shards; found {}, missing {:?}",
            ple_sources.len(),
            missing
        )));
    }
    let physical_rows = (PLE_SHARD_COUNT as u64)
        .checked_mul(PLE_ROWS_PER_SHARD)
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 PLE physical row count overflows".to_string()))?;
    let external_ple_bytes = physical_rows
        .checked_mul(PLE_ROW_WIDTH)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 PLE payload byte count overflows".to_string()))?;
    let source_ple_bytes = ple_sources.values().try_fold(0u64, |sum, source| {
        sum.checked_add(source.data_len())
            .ok_or_else(|| Qwen4Error::Invalid("Qwen4 PLE source byte sum overflows".to_string()))
    })?;
    if external_ple_bytes != source_ple_bytes {
        return Err(Qwen4Error::Invalid(
            "Qwen4 PLE shard payload lengths do not sum to the physical table".to_string(),
        ));
    }
    if gate_up_count == 0 || down_count == 0 {
        return Err(Qwen4Error::Invalid(
            "Qwen4 input is missing routed gate/up or down experts".to_string(),
        ));
    }
    if !mtp_gate_up || !mtp_down {
        return Err(Qwen4Error::Invalid(
            "Qwen4 input is missing native MTP gate/up and down experts".to_string(),
        ));
    }
    if row_chunk == 0 {
        return Err(Qwen4Error::Invalid(
            "Qwen4 row chunk cannot be zero".to_string(),
        ));
    }

    for entry in &mut resident {
        entry.name = entry.source.name.clone();
    }
    resident.sort_by(|left, right| left.name.cmp(&right.name));

    let source_names: Vec<String> = (0..PLE_SHARD_COUNT)
        .map(|index| ple_sources[&index].name.clone())
        .collect();
    let source_shapes: Vec<Vec<u64>> = (0..PLE_SHARD_COUNT)
        .map(|index| ple_sources[&index].shape.clone())
        .collect();
    let metadata_names: Vec<String> = [I64Role::Multipliers, I64Role::VocabSizes, I64Role::Offsets]
        .iter()
        .map(|role| metadata_sources[role].name.clone())
        .collect();
    let metadata_shapes: Vec<Vec<u64>> =
        [I64Role::Multipliers, I64Role::VocabSizes, I64Role::Offsets]
            .iter()
            .map(|role| metadata_sources[role].shape.clone())
            .collect();
    let ple_metadata = PleMetadata {
        multipliers,
        vocab_sizes,
        prefix_offsets,
        source_names,
        source_shapes,
        metadata_names,
        metadata_shapes,
    };
    let mut ple_entries = Vec::with_capacity(PLE_SHARD_COUNT);
    for index in 0..PLE_SHARD_COUNT {
        let source = ple_sources.remove(&index).expect("validated PLE index set");
        let shape = shape_u32(&source.shape, &source.name)?;
        ple_entries.push(PlannedEntry {
            name: source.name.clone(),
            source,
            quant_type: 16,
            shape,
            group_size: 0,
            data_len: PLE_ROWS_PER_SHARD
                .checked_mul(PLE_ROW_WIDTH)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(|| {
                    Qwen4Error::Invalid("Qwen4 PLE entry length overflows".to_string())
                })?,
            kind: EntryKind::Ple,
        });
    }

    let resident_bytes = resident.iter().try_fold(0u64, |sum, entry| {
        sum.checked_add(entry.data_len)
            .ok_or_else(|| Qwen4Error::Invalid("Qwen4 resident byte count overflows".to_string()))
    })?;
    let resident_entries = resident.len();
    let mut entries = metadata_entries;
    entries.extend(resident);
    entries.extend(ple_entries);
    Ok(EntryPlan {
        entries,
        ple_metadata: Some(ple_metadata),
        row_chunk,
        resident_entries,
        expert_entries,
        resident_bytes,
        external_ple_bytes,
    })
}

fn build_metadata(
    config: Option<&Value>,
    plan: &EntryPlan,
    ple: &PleMetadata,
) -> Result<String, Qwen4Error> {
    let valid_rows =
        validate_ple_metadata(&ple.multipliers, &ple.vocab_sizes, &ple.prefix_offsets)?;
    let physical_rows = (PLE_SHARD_COUNT as u64)
        .checked_mul(PLE_ROWS_PER_SHARD)
        .ok_or_else(|| Qwen4Error::Invalid("Qwen4 physical PLE rows overflow".to_string()))?;
    let trailing_padding = physical_rows - valid_rows;
    let shard_records: Vec<Value> = ple
        .source_names
        .iter()
        .zip(ple.source_shapes.iter())
        .enumerate()
        .map(|(index, (name, shape))| {
            let shard_start = (index as u64) * PLE_ROWS_PER_SHARD;
            let valid_in_shard = valid_rows
                .saturating_sub(shard_start)
                .min(PLE_ROWS_PER_SHARD);
            json!({
                "index": index,
                "source_name": name,
                "record_name": name,
                "source_dtype": "BF16",
                "source_shape": shape,
                "residency": "external_rows",
                "valid_rows": valid_in_shard,
                "physical_rows": PLE_ROWS_PER_SHARD,
            })
        })
        .collect();

    let mut root = Map::new();
    root.insert("format".to_string(), Value::String("hfqm".to_string()));
    root.insert("format_version".to_string(), json!(1));
    root.insert("arch_id".to_string(), json!(QWEN4_ARCH_ID));
    root.insert(
        "model_type".to_string(),
        Value::String("qwen4_exp".to_string()),
    );
    root.insert(
        "config".to_string(),
        config
            .cloned()
            .unwrap_or_else(|| json!({"model_type": "qwen4_exp"})),
    );
    root.insert(
        "qwen4_recipe".to_string(),
        json!({
            "version": 1,
            "stacked_experts": true,
            "routing": {"num_experts": ROUTED_EXPERTS, "top_k": ROUTER_TOP_K},
            "gate_up": {"quant_type": 44, "format": "MQ4G256V2", "group_size": MQ4_GROUP_SIZE, "k": HIDDEN_WIDTH},
            "down": {"quant_type": 3, "format": "Q8F16", "group_size": Q8_GROUP_SIZE, "k": DOWN_INTERMEDIATE},
            "nonexpert": {"quant_type": 16, "format": "BF16", "byte_preserving": true},
            "mtp_experts": "same_as_trunk"
        }),
    );
    root.insert(
        "qwen4_ple".to_string(),
        json!({
            "version": QWEN4_PLE_VERSION,
            "schema": "qwen4_ple",
            "metadata_dtype": "I64",
            "metadata_encoding": "little_endian_signed_i64",
            "heads": PLE_HEAD_COUNT,
            "ngram_size": PLE_NGRAM_SIZE,
            "multipliers": ple.multipliers.clone(),
            "prime_vocab_sizes": ple.vocab_sizes.clone(),
            "vocab_sizes": ple.vocab_sizes.clone(),
            "prefix_offsets": ple.prefix_offsets.clone(),
            "offsets": ple.prefix_offsets.clone(),
            "valid_rows": valid_rows,
            "padded_physical_rows": physical_rows,
            "trailing_padding_rows": trailing_padding,
            "physical_rows_per_shard": PLE_ROWS_PER_SHARD,
            "row_width": PLE_ROW_WIDTH,
            "dtype": "BF16",
            "quant_type": 16,
            "metadata_quant_type": QWEN4_I64_QUANT_TYPE,
            "metadata_names": ple.metadata_names.clone(),
            "metadata_shapes": ple.metadata_shapes.clone(),
            "residency": "external_rows",
            "source_names": ple.source_names.clone(),
            "source_shapes": ple.source_shapes.clone(),
            "shards": shard_records,
        }),
    );
    root.insert(
        "qwen4_streaming".to_string(),
        json!({
            "version": 1,
            "row_chunk": plan.row_chunk,
            "resident_entries": plan.resident_entries,
            "expert_entries": plan.expert_entries,
            "external_ple_entries": PLE_SHARD_COUNT,
        }),
    );
    serde_json::to_string(&Value::Object(root))
        .map_err(|error| Qwen4Error::json("serialize Qwen4 metadata", error))
}

fn decode_bf16(raw: &[u8], values: &mut Vec<f32>) -> Result<(), Qwen4Error> {
    if raw.len() % 2 != 0 {
        return Err(Qwen4Error::Invalid(format!(
            "BF16 source chunk has odd byte length {}",
            raw.len()
        )));
    }
    values.clear();
    values.reserve(raw.len() / 2);
    for bytes in raw.chunks_exact(2) {
        values.push(bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])));
    }
    Ok(())
}

fn stream_raw_rows(
    tensor: &SourceTensor,
    row_width: u64,
    element_bytes: u64,
    row_chunk: usize,
    writer: &mut dyn Write,
) -> Result<(), Qwen4Error> {
    let rows = if tensor.shape.len() <= 1 {
        1
    } else {
        tensor.shape[0]
    };
    let expected_elements = if tensor.shape.len() <= 1 {
        checked_product(&tensor.shape, &tensor.name)?
    } else {
        checked_product(&tensor.shape[1..], &tensor.name)?
    };
    if expected_elements != row_width {
        return Err(Qwen4Error::Invalid(format!(
            "{} row width {row_width} does not match shape {:?}",
            tensor.name, tensor.shape
        )));
    }
    let row_bytes = row_width
        .checked_mul(element_bytes)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} row bytes overflow", tensor.name)))?;
    let max_chunk_bytes = (row_chunk as u64)
        .checked_mul(row_bytes)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} chunk length overflows", tensor.name)))?;
    if max_chunk_bytes > MAX_CHUNK_BYTES {
        return Err(Qwen4Error::Invalid(format!(
            "{} logical-row chunk is {max_chunk_bytes} bytes, exceeds bounded {MAX_CHUNK_BYTES}-byte staging",
            tensor.name
        )));
    }
    let expected_len = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} payload length overflow", tensor.name)))?;
    if expected_len != tensor.data_len() {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {} bytes, expected {}",
            tensor.name,
            tensor.data_len(),
            expected_len
        )));
    }
    let mut raw = Vec::new();
    let mut row = 0u64;
    while row < rows {
        let count = (rows - row).min(row_chunk as u64);
        let bytes = count.checked_mul(row_bytes).ok_or_else(|| {
            Qwen4Error::Invalid(format!("{} chunk length overflows", tensor.name))
        })?;
        let bytes_usize = usize::try_from(bytes).map_err(|_| {
            Qwen4Error::Invalid(format!("{} chunk does not fit usize", tensor.name))
        })?;
        raw.resize(bytes_usize, 0);
        let offset = row
            .checked_mul(row_bytes)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{} row offset overflows", tensor.name)))?;
        tensor.read_range(offset, &mut raw)?;
        writer
            .write_all(&raw)
            .map_err(|error| Qwen4Error::io(format!("write {}", tensor.name), error))?;
        row += count;
    }
    Ok(())
}

fn stream_quantized_rows(
    tensor: &SourceTensor,
    kind: ExpertKind,
    rows: u64,
    k: u64,
    row_chunk: usize,
    signs1: &[f32],
    signs2: &[f32],
    writer: &mut dyn Write,
) -> Result<(), Qwen4Error> {
    let row_bytes = k
        .checked_mul(2)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} row bytes overflow", tensor.name)))?;
    let max_chunk_bytes = (row_chunk as u64)
        .checked_mul(row_bytes)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} chunk length overflows", tensor.name)))?;
    if max_chunk_bytes > MAX_CHUNK_BYTES {
        return Err(Qwen4Error::Invalid(format!(
            "{} logical-row chunk is {max_chunk_bytes} bytes, exceeds bounded {MAX_CHUNK_BYTES}-byte staging",
            tensor.name
        )));
    }
    let expected_source = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| Qwen4Error::Invalid(format!("{} source length overflow", tensor.name)))?;
    if expected_source != tensor.data_len() {
        return Err(Qwen4Error::Invalid(format!(
            "{} has {} bytes, expected {}",
            tensor.name,
            tensor.data_len(),
            expected_source
        )));
    }
    let mut raw = Vec::new();
    let mut values = Vec::new();
    let mut row = 0u64;
    let k_usize = usize::try_from(k)
        .map_err(|_| Qwen4Error::Invalid(format!("{} K does not fit usize", tensor.name)))?;
    while row < rows {
        let count = (rows - row).min(row_chunk as u64);
        let bytes = count.checked_mul(row_bytes).ok_or_else(|| {
            Qwen4Error::Invalid(format!("{} chunk length overflows", tensor.name))
        })?;
        let bytes_usize = usize::try_from(bytes).map_err(|_| {
            Qwen4Error::Invalid(format!("{} chunk does not fit usize", tensor.name))
        })?;
        raw.resize(bytes_usize, 0);
        let offset = row
            .checked_mul(row_bytes)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{} row offset overflows", tensor.name)))?;
        tensor.read_range(offset, &mut raw)?;
        decode_bf16(&raw, &mut values)?;
        let count_usize = usize::try_from(count).map_err(|_| {
            Qwen4Error::Invalid(format!("{} row count does not fit usize", tensor.name))
        })?;
        let quantized = match kind {
            ExpertKind::GateUp => quantize_mq4g256v2(&values, count_usize, k_usize, signs1, signs2),
            ExpertKind::Down => quantize_q8f16(&values),
        };
        writer
            .write_all(&quantized)
            .map_err(|error| Qwen4Error::io(format!("write {}", tensor.name), error))?;
        row += count;
    }
    Ok(())
}

fn stream_entry(
    entry: &PlannedEntry,
    row_chunk: usize,
    signs1: &[f32],
    signs2: &[f32],
    writer: &mut dyn Write,
) -> Result<(), Qwen4Error> {
    match entry.kind {
        EntryKind::Bf16 => {
            let row_width = if entry.source.shape.len() <= 1 {
                checked_product(&entry.source.shape, &entry.source.name)?
            } else {
                checked_product(&entry.source.shape[1..], &entry.source.name)?
            };
            stream_raw_rows(&entry.source, row_width, 2, row_chunk, writer)
        }
        EntryKind::Ple => stream_raw_rows(&entry.source, PLE_ROW_WIDTH, 2, row_chunk, writer),
        EntryKind::I64(_) => {
            let row_width = if entry.source.shape.len() <= 1 {
                checked_product(&entry.source.shape, &entry.source.name)?
            } else {
                checked_product(&entry.source.shape[1..], &entry.source.name)?
            };
            stream_raw_rows(&entry.source, row_width, 8, row_chunk, writer)
        }
        EntryKind::GateUp | EntryKind::Down => {
            let kind = match entry.kind {
                EntryKind::GateUp => ExpertKind::GateUp,
                EntryKind::Down => ExpertKind::Down,
                EntryKind::Bf16 | EntryKind::Ple | EntryKind::I64(_) => unreachable!(),
            };
            let rows = entry.source.shape[0]
                .checked_mul(entry.source.shape[1])
                .ok_or_else(|| Qwen4Error::Invalid(format!("{} rows overflow", entry.name)))?;
            stream_quantized_rows(
                &entry.source,
                kind,
                rows,
                entry.source.shape[2],
                row_chunk,
                signs1,
                signs2,
                writer,
            )
        }
    }
}

/// Bounded, mmap-free reopen plan used both by the producer and by fixture
/// tests.  It is intentionally a plan rather than a payload reader: loading
/// the 100+ GiB PLE records is the runtime's external-row responsibility.
#[derive(Debug)]
pub(crate) struct Qwen4ReopenPlan {
    pub(crate) arch_id: u32,
    pub(crate) metadata_json: String,
    pub(crate) entries: Vec<Qwen4PlanEntry>,
}

#[derive(Debug, Clone)]
pub(crate) struct Qwen4PlanEntry {
    pub(crate) name: String,
    pub(crate) quant_type: u8,
    pub(crate) shape: Vec<u32>,
    pub(crate) group_size: u32,
    pub(crate) data_offset: u64,
    pub(crate) data_len: u64,
}

impl Qwen4ReopenPlan {
    pub(crate) fn open(path: &Path) -> Result<Self, Qwen4Error> {
        let file =
            File::open(path).map_err(|error| Qwen4Error::io(path.display().to_string(), error))?;
        let file_len = file
            .metadata()
            .map_err(|error| Qwen4Error::io(format!("stat {}", path.display()), error))?
            .len();
        if file_len < 32 {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 artifact {} is shorter than its 32-byte header",
                path.display()
            )));
        }
        let mut header = [0u8; 32];
        read_exact_at(&file, 0, &mut header).map_err(|error| {
            Qwen4Error::io(
                format!("read Qwen4 artifact header {}", path.display()),
                error,
            )
        })?;
        if &header[..4] != b"HFQM" {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 artifact {} has wrong HFQM magic",
                path.display()
            )));
        }
        let version = u32::from_le_bytes(header[4..8].try_into().expect("header slice"));
        if version != 1 {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 artifact {} has unsupported version {version}",
                path.display()
            )));
        }
        let arch_id = u32::from_le_bytes(header[8..12].try_into().expect("header slice"));
        if arch_id != QWEN4_ARCH_ID {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 artifact has arch id {arch_id}, expected {QWEN4_ARCH_ID}"
            )));
        }
        let tensor_count =
            u32::from_le_bytes(header[12..16].try_into().expect("header slice")) as usize;
        let metadata_offset = u64::from_le_bytes(header[16..24].try_into().expect("header slice"));
        let data_offset = u64::from_le_bytes(header[24..32].try_into().expect("header slice"));
        if metadata_offset < 32 || metadata_offset > data_offset || data_offset > file_len {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 artifact has invalid metadata/data offsets {metadata_offset}/{data_offset} for file {file_len}"
            )));
        }
        let region_len = data_offset - metadata_offset;
        if region_len > MAX_REOPEN_REGION_BYTES {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 metadata/index region is {region_len} bytes, exceeds {MAX_REOPEN_REGION_BYTES}"
            )));
        }
        let region_len_usize = usize::try_from(region_len).map_err(|_| {
            Qwen4Error::Invalid("Qwen4 metadata/index region does not fit usize".to_string())
        })?;
        let mut region = vec![0u8; region_len_usize];
        read_exact_at(&file, metadata_offset, &mut region).map_err(|error| {
            Qwen4Error::io(
                format!("read Qwen4 metadata/index {}", path.display()),
                error,
            )
        })?;
        let json_end = json_blob_end(&region).ok_or_else(|| {
            Qwen4Error::Invalid("Qwen4 metadata JSON is not terminated".to_string())
        })?;
        let metadata_json = String::from_utf8(region[..json_end].to_vec()).map_err(|error| {
            Qwen4Error::Invalid(format!("Qwen4 metadata is not UTF-8: {error}"))
        })?;
        let metadata: Value = serde_json::from_str(&metadata_json)
            .map_err(|error| Qwen4Error::json("parse reopened Qwen4 metadata", error))?;
        validate_reopened_metadata(&metadata)?;

        let mut pos = json_end;
        let read_u8 = |region: &[u8], pos: &mut usize, what: &str| -> Result<u8, Qwen4Error> {
            if *pos >= region.len() {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 index truncated while reading {what}"
                )));
            }
            let value = region[*pos];
            *pos += 1;
            Ok(value)
        };
        let read_u16 = |region: &[u8], pos: &mut usize, what: &str| -> Result<u16, Qwen4Error> {
            if region.len().saturating_sub(*pos) < 2 {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 index truncated while reading {what}"
                )));
            }
            let value = u16::from_le_bytes(region[*pos..*pos + 2].try_into().expect("index slice"));
            *pos += 2;
            Ok(value)
        };
        let read_u32 = |region: &[u8], pos: &mut usize, what: &str| -> Result<u32, Qwen4Error> {
            if region.len().saturating_sub(*pos) < 4 {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 index truncated while reading {what}"
                )));
            }
            let value = u32::from_le_bytes(region[*pos..*pos + 4].try_into().expect("index slice"));
            *pos += 4;
            Ok(value)
        };
        let read_u64 = |region: &[u8], pos: &mut usize, what: &str| -> Result<u64, Qwen4Error> {
            if region.len().saturating_sub(*pos) < 8 {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 index truncated while reading {what}"
                )));
            }
            let value = u64::from_le_bytes(region[*pos..*pos + 8].try_into().expect("index slice"));
            *pos += 8;
            Ok(value)
        };
        let index_count = read_u32(&region, &mut pos, "tensor count")? as usize;
        if index_count != tensor_count {
            return Err(Qwen4Error::Invalid(format!(
                "Qwen4 header has {tensor_count} tensors but index has {index_count}"
            )));
        }
        let mut entries = Vec::with_capacity(index_count);
        let mut names = HashMap::with_capacity(index_count);
        let mut cumulative_offset = data_offset;
        for _ in 0..index_count {
            let name_len = read_u16(&region, &mut pos, "name length")? as usize;
            if region.len().saturating_sub(pos) < name_len {
                return Err(Qwen4Error::Invalid(
                    "Qwen4 index truncated while reading tensor name".to_string(),
                ));
            }
            let name =
                String::from_utf8(region[pos..pos + name_len].to_vec()).map_err(|error| {
                    Qwen4Error::Invalid(format!("Qwen4 tensor name is not UTF-8: {error}"))
                })?;
            pos += name_len;
            let quant_type = read_u8(&region, &mut pos, "quant type")?;
            if !matches!(quant_type, 3 | 16 | 44 | QWEN4_I64_QUANT_TYPE) {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 artifact tensor {name} uses unknown quant type {quant_type}"
                )));
            }
            let n_dims = read_u8(&region, &mut pos, "dimension count")? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(read_u32(&region, &mut pos, "shape dimension")?);
            }
            let group_size = read_u32(&region, &mut pos, "group size")?;
            let data_len = read_u64(&region, &mut pos, "data length")?;
            let end = cumulative_offset.checked_add(data_len).ok_or_else(|| {
                Qwen4Error::Invalid(format!("Qwen4 tensor {name} payload overflows"))
            })?;
            if end > file_len {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 tensor {name} payload [{cumulative_offset},{end}) exceeds file {file_len}"
                )));
            }
            if names.insert(name.clone(), ()).is_some() {
                return Err(Qwen4Error::Invalid(format!(
                    "duplicate Qwen4 output tensor {name}"
                )));
            }
            validate_output_entry_len(&name, quant_type, &shape, data_len)?;
            entries.push(Qwen4PlanEntry {
                name,
                quant_type,
                shape,
                group_size,
                data_offset: cumulative_offset,
                data_len,
            });
            cumulative_offset = end;
        }
        if pos > region.len() {
            return Err(Qwen4Error::Invalid(
                "Qwen4 index exceeds metadata/data region".to_string(),
            ));
        }
        Ok(Self {
            arch_id,
            metadata_json,
            entries,
        })
    }
}

fn validate_output_entry_len(
    name: &str,
    quant_type: u8,
    shape: &[u32],
    data_len: u64,
) -> Result<(), Qwen4Error> {
    match quant_type {
        QWEN4_I64_QUANT_TYPE => {
            if i64_role(name).is_none() {
                return Err(Qwen4Error::Invalid(format!(
                    "Qwen4 raw-I64 output {name} is not a recognized PLE metadata tensor"
                )));
            }
            let expected_shape: &[u32] = match i64_role(name).expect("role checked above") {
                I64Role::Multipliers => &[3],
                I64Role::VocabSizes | I64Role::Offsets => &[16],
            };
            if shape != expected_shape {
                return Err(Qwen4Error::Invalid(format!(
                    "{name} raw-I64 shape {:?} is not {:?}",
                    shape, expected_shape
                )));
            }
            let expected = checked_product(
                &shape.iter().map(|&value| value as u64).collect::<Vec<_>>(),
                name,
            )?
            .checked_mul(8)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} I64 length overflows")))?;
            if expected != data_len {
                return Err(Qwen4Error::Invalid(format!(
                    "{name} I64 payload is {data_len} bytes, expected {expected}"
                )));
            }
        }
        16 => {
            let expected = checked_product(
                &shape.iter().map(|&value| value as u64).collect::<Vec<_>>(),
                name,
            )?
            .checked_mul(2)
            .ok_or_else(|| Qwen4Error::Invalid(format!("{name} BF16 length overflows")))?;
            if expected != data_len {
                return Err(Qwen4Error::Invalid(format!(
                    "{name} BF16 payload is {data_len} bytes, expected {expected}"
                )));
            }
        }
        44 => {
            if shape != [512, 1280, 2560] {
                return Err(Qwen4Error::Invalid(format!(
                    "{name} MQ4 output shape {:?} is not [512,1280,2560]",
                    shape
                )));
            }
            let expected = quantized_data_len(ExpertKind::GateUp, 512 * 1280, 2560)?;
            if expected != data_len {
                return Err(Qwen4Error::Invalid(format!(
                    "{name} MQ4 payload is {data_len} bytes, expected {expected}"
                )));
            }
        }
        3 => {
            if shape != [512, 2560, 640] {
                return Err(Qwen4Error::Invalid(format!(
                    "{name} Q8 output shape {:?} is not [512,2560,640]",
                    shape
                )));
            }
            let expected = quantized_data_len(ExpertKind::Down, 512 * 2560, 640)?;
            if expected != data_len {
                return Err(Qwen4Error::Invalid(format!(
                    "{name} Q8 payload is {data_len} bytes, expected {expected}"
                )));
            }
        }
        other => {
            return Err(Qwen4Error::Invalid(format!(
                "{name} uses unsupported Qwen4 output quant_type {other}"
            )));
        }
    }
    Ok(())
}

fn validate_reopened_metadata(metadata: &Value) -> Result<(), Qwen4Error> {
    let object = metadata.as_object().ok_or_else(|| {
        Qwen4Error::Invalid("reopened Qwen4 metadata is not an object".to_string())
    })?;
    if object.get("arch_id").and_then(Value::as_u64) != Some(QWEN4_ARCH_ID as u64) {
        return Err(Qwen4Error::Invalid(
            "reopened Qwen4 metadata arch_id is not 16".to_string(),
        ));
    }
    let ple = object
        .get("qwen4_ple")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            Qwen4Error::Invalid("reopened Qwen4 metadata is missing qwen4_ple".to_string())
        })?;
    if ple.get("version").and_then(Value::as_u64) != Some(QWEN4_PLE_VERSION as u64)
        || ple.get("metadata_dtype").and_then(Value::as_str) != Some("I64")
        || ple.get("residency").and_then(Value::as_str) != Some("external_rows")
    {
        return Err(Qwen4Error::Invalid(
            "reopened qwen4_ple metadata has an unsupported version, dtype, or residency"
                .to_string(),
        ));
    }
    let source_names = ple
        .get("source_names")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            Qwen4Error::Invalid("reopened qwen4_ple metadata is missing source_names".to_string())
        })?;
    if source_names.len() != PLE_SHARD_COUNT {
        return Err(Qwen4Error::Invalid(format!(
            "reopened qwen4_ple metadata has {} source names, expected {PLE_SHARD_COUNT}",
            source_names.len()
        )));
    }
    Ok(())
}

fn json_blob_end(bytes: &[u8]) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (index, &byte) in bytes.iter().enumerate() {
        if escape {
            escape = false;
            continue;
        }
        if in_string && byte == b'\\' {
            escape = true;
            continue;
        }
        if byte == b'"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match byte {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index + 1);
                }
                if depth < 0 {
                    return None;
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(unix)]
fn read_exact_at(file: &File, offset: u64, dst: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(dst, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, offset: u64, dst: &mut [u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < dst.len() {
        let read = file.seek_read(&mut dst[done..], offset + done as u64)?;
        if read == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short positional read",
            ));
        }
        done += read;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn read_exact_at(file: &File, offset: u64, dst: &mut [u8]) -> io::Result<()> {
    use std::io::{Seek, SeekFrom};
    let mut file = file.try_clone()?;
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(dst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read as _, Write as _};
    use std::net::{TcpListener, TcpStream};
    use std::thread::{self, JoinHandle};
    use tempfile::NamedTempFile;

    fn test_remote_source(base_url: &str) -> RemoteSource {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(5)))
            .http_status_as_error(false)
            .build()
            .into();
        RemoteSource {
            spec: RemoteSpec {
                owner: "owner".to_string(),
                repo: "repo".to_string(),
                revision: "0123456789abcdef0123456789abcdef01234567".to_string(),
            },
            base_url: base_url.trim_end_matches('/').to_string(),
            agent,
            authorization: None,
        }
    }

    fn spawn_http_response(
        status: u16,
        content_range: Option<&str>,
        body: &[u8],
        announced_length: Option<usize>,
        assert_range: Option<&str>,
    ) -> (String, JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let address = listener.local_addr().expect("mock server address");
        let content_range = content_range.map(str::to_string);
        let body = body.to_vec();
        let announced = announced_length.unwrap_or(body.len());
        let assert_range = assert_range.map(str::to_string);
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept mock request");
            let mut request = [0u8; 4096];
            let count = stream.read(&mut request).expect("read mock request");
            let request = String::from_utf8_lossy(&request[..count]);
            if let Some(expected) = assert_range {
                assert!(
                    request
                        .lines()
                        .any(|line| line.eq_ignore_ascii_case(&format!("Range: {expected}"))),
                    "request did not contain expected Range header: {request}"
                );
            }
            let reason = match status {
                206 => "Partial Content",
                200 => "OK",
                _ => "Response",
            };
            let mut response =
                format!("HTTP/1.1 {status} {reason}\r\nContent-Length: {announced}\r\n");
            if let Some(content_range) = content_range {
                response.push_str(&format!("Content-Range: {content_range}\r\n"));
            }
            response.push_str("Connection: close\r\n\r\n");
            stream
                .write_all(response.as_bytes())
                .expect("write mock headers");
            stream.write_all(&body).expect("write mock body");
        });
        (format!("http://{address}"), handle)
    }

    #[test]
    fn remote_spec_requires_full_immutable_revision() {
        for input in [
            "hf://owner/repo",
            "hf://owner/repo@main",
            "hf://owner/repo@0123456789abcdef",
            "hf://owner/repo@0123456789abcdef0123456789abcdef0123456z",
            "hf://owner/repo@0123456789abcdef0123456789abcdef01234567@main",
        ] {
            assert!(
                parse_remote_spec(input).is_err(),
                "accepted invalid spec {input}"
            );
        }
        let spec = parse_remote_spec("hf://owner/repo@0123456789ABCDEF0123456789abcdef01234567")
            .expect("parse valid remote spec")
            .expect("remote spec");
        assert_eq!(spec.owner, "owner");
        assert_eq!(spec.repo, "repo");
        assert_eq!(spec.revision, "0123456789ABCDEF0123456789abcdef01234567");
        assert!(parse_remote_spec("weights/model.safetensors")
            .expect("local path is not remote")
            .is_none());
    }

    #[test]
    fn remote_url_targets_exact_repo_revision_and_object() {
        let source = test_remote_source("https://huggingface.co/");
        assert_eq!(
            source.url_for("weights/model.safetensors"),
            "https://huggingface.co/owner/repo/resolve/0123456789abcdef0123456789abcdef01234567/weights/model.safetensors"
        );
    }

    #[test]
    fn content_range_parser_requires_explicit_valid_bounds() {
        assert_eq!(parse_content_range("bytes 2-5/8").unwrap(), (2, 5, 8));
        for value in [
            "bytes 2-5/*",
            "bytes 5-2/8",
            "bytes 0-8/8",
            "bytes 0-3/8 extra",
            "items 0-3/8",
            "bytes 0-3",
            "bytes 0-a/8",
        ] {
            assert!(parse_content_range(value).is_err(), "accepted {value}");
        }
    }

    #[test]
    fn remote_range_reader_accepts_exact_partial_response() {
        let (base, handle) =
            spawn_http_response(206, Some("bytes 2-5/8"), b"cdef", None, Some("bytes=2-5"));
        let source = test_remote_source(&base);
        let mut bytes = [0u8; 4];
        assert_eq!(
            source
                .read_range("model.safetensors", 2, &mut bytes, Some(8))
                .unwrap(),
            8
        );
        assert_eq!(&bytes, b"cdef");
        handle.join().unwrap();
    }

    #[test]
    fn remote_range_reader_rejects_full_response() {
        let (base, handle) = spawn_http_response(200, None, b"cdef", None, None);
        let source = test_remote_source(&base);
        let mut bytes = [0u8; 4];
        assert!(source
            .read_range("model.safetensors", 2, &mut bytes, Some(8))
            .is_err());
        handle.join().unwrap();
    }

    #[test]
    fn remote_range_reader_rejects_mismatched_content_range() {
        let (base, handle) =
            spawn_http_response(206, Some("bytes 0-3/8"), b"cdef", None, Some("bytes=2-5"));
        let source = test_remote_source(&base);
        let mut bytes = [0u8; 4];
        assert!(source
            .read_range("model.safetensors", 2, &mut bytes, Some(8))
            .is_err());
        handle.join().unwrap();
    }

    #[test]
    fn remote_range_reader_rejects_short_body() {
        let (base, handle) =
            spawn_http_response(206, Some("bytes 2-5/8"), b"cde", Some(4), Some("bytes=2-5"));
        let source = test_remote_source(&base);
        let mut bytes = [0u8; 4];
        assert!(source
            .read_range("model.safetensors", 2, &mut bytes, Some(8))
            .is_err());
        handle.join().unwrap();
    }
    fn source_tensor(bytes: &[u8], shape: Vec<u64>, dtype: &str) -> (NamedTempFile, SourceTensor) {
        let mut file = NamedTempFile::new().expect("temp source");
        file.write_all(bytes).expect("source bytes");
        file.as_file().sync_all().expect("sync source");
        let path = file.path().to_path_buf();
        let source = Arc::new(SourceShard {
            path,
            kind: SourceKind::Local {
                file: Arc::new(file.reopen().expect("reopen source")),
            },
            file_len: bytes.len() as u64,
        });
        (
            file,
            SourceTensor {
                name: "fixture.tensor".to_string(),
                dtype: dtype.to_string(),
                shape,
                data_start: 0,
                data_end: bytes.len() as u64,
                shard: source,
            },
        )
    }

    #[test]
    fn qwen4_recipe_geometry_has_no_padding_fallback() {
        assert_eq!(
            quantized_data_len(ExpertKind::GateUp, 512 * 1280, 2560).unwrap(),
            891_289_600
        );
        assert_eq!(
            quantized_data_len(ExpertKind::Down, 512 * 2560, 640).unwrap(),
            891_289_600
        );
        assert!(quantized_data_len(ExpertKind::GateUp, 1, 640).is_err());
        assert!(validate_expert_shape_stub(&[512, 1280, 2559], ExpertKind::GateUp).is_err());
    }

    fn validate_expert_shape_stub(shape: &[u64], kind: ExpertKind) -> Result<(), Qwen4Error> {
        let expected = match kind {
            ExpertKind::GateUp => [ROUTED_EXPERTS, GATE_UP_INTERMEDIATE, HIDDEN_WIDTH],
            ExpertKind::Down => [ROUTED_EXPERTS, HIDDEN_WIDTH, DOWN_INTERMEDIATE],
        };
        if shape != expected {
            return Err(Qwen4Error::Invalid("shape mismatch".to_string()));
        }
        Ok(())
    }

    #[test]
    fn ple_suffix_sort_is_numeric_not_lexical() {
        let mut names = [
            "model.ple.ngram_embedding.shard_10.weight",
            "model.ple.ngram_embedding.shard_2.weight",
            "model.ple.ngram_embedding.shard_0.weight",
        ];
        names.sort_by_key(|name| ple_shard_index(name).unwrap());
        assert_eq!(names[0], "model.ple.ngram_embedding.shard_0.weight");
        assert_eq!(names[1], "model.ple.ngram_embedding.shard_2.weight");
        assert_eq!(names[2], "model.ple.ngram_embedding.shard_10.weight");
    }

    #[test]
    fn bf16_stream_preserves_source_bytes() {
        let bytes = [0x00, 0x3f, 0x80, 0xbf, 0x34, 0x12, 0xff, 0x7f];
        let (_file, tensor) = source_tensor(&bytes, vec![2, 2], "BF16");
        let mut output = Vec::new();
        stream_raw_rows(&tensor, 2, 2, 1, &mut output).unwrap();
        assert_eq!(output, bytes);
    }

    #[test]
    fn i64_metadata_keeps_signed_values_exactly() {
        let values = [i64::MIN + 7, -19, i64::MAX - 11];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let (_file, tensor) = source_tensor(&bytes, vec![3], "I64");
        assert_eq!(read_i64_array(&tensor, 3).unwrap(), values);
        assert_eq!(i64_role("x.layer_multipliers"), Some(I64Role::Multipliers));
        assert_eq!(QWEN4_I64_QUANT_TYPE, 52);
        assert!(
            validate_output_entry_len("x.layer_multipliers", QWEN4_I64_QUANT_TYPE, &[3], 24,)
                .is_ok()
        );
        assert!(validate_output_entry_len("x.layer_multipliers", 22, &[3], 12).is_err());
    }

    #[test]
    fn reopen_plan_reads_small_fixture_without_mapping_payload() {
        let temp = NamedTempFile::new().unwrap();
        let metadata = json!({
            "arch_id": 16,
            "qwen4_ple": {
                "version": 1,
                "metadata_dtype": "I64",
                "residency": "external_rows",
                "source_names": (0..128).map(|i| format!("shard_{i}")).collect::<Vec<_>>()
            }
        })
        .to_string();
        let payload = [0x00u8, 0x3f, 0x80, 0xbf];
        let entry = hipfire_runtime::hfq::HfqStreamEntry {
            name: "fixture.weight".to_string(),
            quant_type: 16,
            shape: vec![2],
            group_size: 0,
            data_len: payload.len() as u64,
        };
        hipfire_runtime::hfq::write_hfqm_package_streaming(
            temp.path(),
            16,
            &metadata,
            &[entry],
            |_, writer| writer.write_all(&payload),
        )
        .unwrap();
        let plan = Qwen4ReopenPlan::open(temp.path()).unwrap();
        assert_eq!(plan.arch_id, 16);
        assert_eq!(plan.entries.len(), 1);
        assert_eq!(plan.entries[0].data_len, payload.len() as u64);
    }
}
