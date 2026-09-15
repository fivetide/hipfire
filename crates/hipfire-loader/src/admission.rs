// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt

//! Source-aware admission (device-mesh G2).
//!
//! Classifies a retained source once and decides one effective topology BEFORE
//! any destructive side effect (prior-model teardown, VMM init, remap, GPU
//! allocation, carrier entry, collective creation). The load route consumes the
//! [`SourceAdmission`]'s already-open [`ModelSource`] — never re-opening or
//! re-classifying the path. A refusal here leaves whatever model is currently
//! loaded untouched: no teardown, no allocation, no carrier entry, no cache
//! mutation.

use crate::Carrier;
use hipfire_arch_qwen4::config::{Qwen4Config, ARCH_ID as QWEN4_ARCH_ID};
use hipfire_arch_qwen4::{InputModality, Qwen4Capabilities};
use hipfire_runtime::kv_backend::KvBackend;
use hipfire_runtime::loader_api::{ModelSource, SpecLoadCfg};
fn qwen4_vision_tensor_name(name: &str) -> bool {
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

/// The one effective topology admitted for a load. `tp>1` (expert-parallel) and
/// `pp>1` (pipeline-parallel) are mutually exclusive; both default to 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveTopology {
    Single,
    Pipeline(usize),
    Expert(usize),
}

/// Optional load-time controls that must be rejected at Qwen4 admission
/// before any previous model teardown or GPU allocation.
#[derive(Clone, Copy, Default)]
pub struct SourceAdmissionOptions {
    pub spec: SpecLoadCfg,
    pub gemma4_drafter: bool,
    pub cask: bool,
    pub state_quant: bool,
    pub non_single_compute: bool,
    pub pflash: bool,
}

/// Native Qwen4 MTP is an explicit load intent. A missing field and an
/// explicit disable both keep the carrier on its ordinary AR-only shape.
pub(crate) const fn qwen4_native_mtp_requested(spec: SpecLoadCfg) -> bool {
    matches!(spec.mtp, Some(true))
}

const QWEN4_DDTREE_DEFAULT_BUDGET: usize = 0;
const QWEN4_DDTREE_DEFAULT_TOPK: usize = 4;

/// Return whether a Qwen4 load carries an active or non-default DDTree
/// request. The CLI resolves schema defaults before serializing load params,
/// so an ordinary AR load arrives as `Some(0)`/`Some(4)` rather than `None`.
/// A non-default top-K remains unsupported even when the budget is zero.
pub(crate) const fn qwen4_ddtree_requested(spec: SpecLoadCfg) -> bool {
    match (spec.ddtree_budget, spec.ddtree_topk) {
        (Some(budget), _) if budget != QWEN4_DDTREE_DEFAULT_BUDGET => true,
        (_, Some(topk)) if topk != QWEN4_DDTREE_DEFAULT_TOPK => true,
        _ => false,
    }
}

/// The source-only portion of Qwen4 admission. The validated config and
/// inventory are reused by the executable Single carrier without reopening
/// or reclassifying the HFQM path.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4SourceAdmission {
    pub config: Qwen4Config,
    pub effective_mesh: hipfire_arch_qwen4::EffectiveMesh,
    pub modality: InputModality,
    pub native_mtp: bool,
    pub manifest_source_tensors: usize,
}

/// Refuse unsupported Qwen4 request dimensions before touching source records.
/// In particular, image/video requests cannot make the boundary inspect vision
/// payloads and then silently fall back to text.
pub fn qwen4_request_admission(
    effective_mesh: hipfire_arch_qwen4::EffectiveMesh,
    modality: InputModality,
    native_mtp: bool,
) -> Result<(), String> {
    let capabilities = if native_mtp {
        Qwen4Capabilities::text_mtp()
    } else {
        Qwen4Capabilities::text_ar()
    };
    if !capabilities.supports_modality(modality) {
        return Err(format!("qwen4: unsupported modality {modality:?}"));
    }
    if native_mtp && !capabilities.native_mtp {
        return Err("qwen4: native MTP is not supported by this source".into());
    }
    if !effective_mesh.is_single() {
        return Err(format!(
            "qwen4: only Single topology is admitted (pp={}, tp={}, ep={})",
            effective_mesh.pp, effective_mesh.tp, effective_mesh.ep
        ));
    }
    Ok(())
}
/// Validate a reserved arch-16 HFQM source without allocating weights.
///
/// The architecture crate owns the complete source/index contract.  Loader
/// admission only adds request/topology policy and retains the receipt's
/// value-only config/count for the carrier handoff.
pub fn admit_qwen4_source(
    source: &ModelSource,
    effective_mesh: hipfire_arch_qwen4::EffectiveMesh,
    modality: InputModality,
    native_mtp: bool,
) -> Result<Qwen4SourceAdmission, String> {
    qwen4_request_admission(effective_mesh, modality, native_mtp)?;
    let ModelSource::Hfq(hfq) = source else {
        return Err(
            "qwen4: safetensors is a conversion input, not an executable HFQ artifact".into(),
        );
    };
    let receipt =
        hipfire_arch_qwen4::admit_hfqm_artifact(hfq).map_err(|error| error.to_string())?;
    Ok(Qwen4SourceAdmission {
        config: receipt.config,
        effective_mesh,
        modality,
        native_mtp,
        manifest_source_tensors: receipt.source_tensor_count,
    })
}

/// A source admitted before any destructive side effect.
pub struct SourceAdmission {
    /// The already-open source. The single/pp route consumes it (no second
    /// open); the EP route re-opens `path` per rank and drops this handle.
    pub source: ModelSource,
    pub arch_id: u32,
    pub is_dir: bool,
    /// Tower-tensor presence decides text-vs-VL; config metadata alone never
    /// does (remediation contract `179a20d7f`).
    pub has_vision: bool,
    pub topology: EffectiveTopology,
    pub kv_backend: KvBackend,
    /// The resolved carrier (single/pp path). `None` for expert-parallel, which
    /// dispatches on `arch_id` directly rather than through the registry.
    pub carrier: Option<&'static dyn Carrier>,
    /// Validated vision-tower sidecar (`params.vision` / `HIPFIRE_VISION_SIDECAR`).
    /// `Some` only when the sidecar opened, carries arch_id 5|6, and holds the
    /// tower probe tensor; the single/pp route threads it into `LoadCtx`.
    /// `None` = trunk-only (or explicit opt-out via empty string).
    pub vision_path: Option<std::path::PathBuf>,
}

/// Pure text-vs-VL decision. The vision tower tensor decides; configuration
/// metadata alone never does. Every Qwen3.5-family HF config embeds
/// `vision_config` even for text-only quantized artifacts, so a config marker
/// without the tower is the text backbone, not a refusal.
///
/// LFM2 is the one exception that *refuses*: a tower tensor with no parseable
/// `vision_config` metadata is malformed (carriers.rs:1553-1558) and fails
/// closed rather than silently loading as text.
///
/// Contract provenance: remediation commit `179a20d7f` ("classify Qwen3.5/LFM2
/// sources by vision tower tensor, not config markers").
pub fn classify_vision(
    arch_id: u32,
    has_vision_tensor: bool,
    has_vision_config: bool,
) -> Result<bool, String> {
    match arch_id {
        // Qwen3.5 dense (5) / MoE (6): the tower tensor alone decides.
        5 | 6 => Ok(has_vision_tensor),
        // LFM2 (11): tower + config both required; tower-without-config refuses.
        11 => {
            if has_vision_tensor && !has_vision_config {
                return Err(
                    "lfm2moe: artifact carries vision tensors but no vision_config \
                     metadata — requantize with --include-vision"
                        .into(),
                );
            }
            Ok(has_vision_tensor)
        }
        // Qwen4's conditional-generation name does not imply a vision route.
        // A text artifact is capable when no tower records are present; a
        // tower-bearing artifact is refused rather than probed as an image
        // request or silently downgraded to text.
        QWEN4_ARCH_ID => {
            if has_vision_tensor {
                Err("qwen4: vision tensors are unsupported by the text-only route".into())
            } else {
                Ok(false)
            }
        }
        _ => Ok(false),
    }
}

/// Read the vision-tower probes out of an already-open source and fold them
/// through [`classify_vision`]. Read-only: probes the HFQ tensor index and (for
/// LFM2) parses `vision_config` metadata; touches no GPU state.
fn probe_vision(src: &ModelSource, arch_id: u32) -> Result<bool, String> {
    let ModelSource::Hfq(hfq) = src else {
        return classify_vision(arch_id, false, false);
    };
    let (has_tensor, has_config) = match arch_id {
        5 | 6 => (
            hfq.tensor_data("model.visual.patch_embed.proj.weight")
                .is_some(),
            // Qwen3.5 does not use the config in the decision; config parse is
            // soft (carriers.rs:527-544). A dummy `false` is never read.
            false,
        ),
        11 => (
            hfq.tensor_data("model.vision_tower.vision_model.embeddings.patch_embedding.weight")
                .is_some(),
            hipfire_arch_lfm2_vl::vision_config_from_hfq(hfq).is_some(),
        ),
        QWEN4_ARCH_ID => (
            hfq.tensors()
                .iter()
                .any(|tensor| qwen4_vision_tensor_name(&tensor.name)),
            false,
        ),
        _ => (false, false),
    };
    classify_vision(arch_id, has_tensor, has_config)
}

/// Tower probe tensor shared by the trunk and the vision sidecar.
const VISION_PROBE_TENSOR: &str = "model.visual.patch_embed.proj.weight";

/// Validate the optional vision-tower sidecar and return its resolved path.
/// Fail-closed: an unopenable, non-5|6, or tower-less sidecar refuses with a
/// message naming the remedy. The sidecar opens read-only as a SEPARATE
/// `HfqFile` (never `attach_overlay` — REAP rejects additive tensor names,
/// hfq.rs:391-427). Empty string counts as unset (explicit opt-out, same
/// semantics as `HIPFIRE_DFLASH_DRAFT`).
fn resolve_vision_sidecar(
    vision: Option<&str>,
    arch_id: u32,
    is_dir: bool,
) -> Result<Option<std::path::PathBuf>, String> {
    let path = match vision.filter(|s| !s.is_empty()) {
        Some(p) => p,
        None => return Ok(None),
    };
    if is_dir {
        return Err(format!(
            "vision sidecar '{path}' requires an HFQ trunk: safetensors directory \
             sources cannot carry a sidecar tower"
        ));
    }
    if !matches!(arch_id, 5 | 6) {
        return Err(format!(
            "vision sidecar '{path}' requested for arch_id={arch_id}: vision sidecars \
             only serve Qwen3.5-VL trunks (arch_id 5|6)"
        ));
    }
    let sidecar = hipfire_runtime::hfq::HfqFile::open(std::path::Path::new(path))
        .map_err(|e| format!("vision sidecar '{path}': open failed: {e}"))?;
    if !matches!(sidecar.arch_id, 5 | 6) {
        return Err(format!(
            "vision sidecar '{path}' has arch_id={} (expected 5|6): pack the tower \
             with `hipfire-quantize <hf-dir> --include-vision --include-prefix model.visual.`",
            sidecar.arch_id
        ));
    }
    if sidecar.tensor_data(VISION_PROBE_TENSOR).is_none() {
        return Err(format!(
            "vision sidecar '{path}' carries no vision tower tensor \
             '{VISION_PROBE_TENSOR}': pack the tower with `hipfire-quantize <hf-dir> \
             --include-vision --include-prefix model.visual.`"
        ));
    }
    Ok(Some(std::path::PathBuf::from(path)))
}

/// Maple head-overlay arch id (`hipfire-quantize --head-only` carriers).
const MAPLE_ARCH_ID: u32 = 15;

/// Validate a `--head` overlay against the already-open base and attach it so
/// the retained source IS the effective base+head: loading consumes it with
/// no second open. Every refusal fires here, before prior-model teardown.
/// Empty string counts as unset (explicit opt-out, same as vision/draft).
/// Refusals, never silent fallbacks: serving the base head when an overlay
/// was requested hands back a model the operator did not ask for.
fn admit_head_overlay(
    head: Option<&str>,
    base: &mut ModelSource,
    arch_id: u32,
    topology: EffectiveTopology,
) -> Result<(), String> {
    let path = match head.filter(|s| !s.is_empty()) {
        Some(p) => p,
        None => return Ok(()),
    };
    if base.is_dir() {
        return Err(format!(
            "--head '{path}' requires an HFQ trunk: safetensors directory \
             sources cannot carry a head overlay"
        ));
    }
    if topology != EffectiveTopology::Single {
        return Err(format!(
            "--head '{path}' requires a single-device load (tp=1, pp=1): \
             expert/pipeline-parallel loads use the head baked into the model file"
        ));
    }
    if arch_id != MAPLE_ARCH_ID {
        return Err(format!(
            "--head '{path}' requested for arch_id={arch_id}: head overlays only \
             serve Maple (arch_id 15) — refusing rather than silently serving \
             the base head"
        ));
    }
    let ModelSource::Hfq(hfq) = &mut *base else {
        return Err(format!("--head '{path}' requires an HFQ trunk"));
    };
    // The overlay slot holds at most one file: a REAP splice already
    // installed there would be silently discarded by the head attach.
    // Conservatively refuse the combination instead of stacking overlays.
    if hfq.has_overlay() {
        return Err(format!(
            "--head '{path}' cannot combine with an active REAP overlay \
             (single overlay slot): disable one of them"
        ));
    }
    let ov = hipfire_runtime::hfq::HfqFile::open_at_offset(std::path::Path::new(path), 0)
        .map_err(|e| format!("head overlay '{path}': open failed: {e}"))?;
    hfq.attach_opened_head(ov, std::path::Path::new(path))?;
    Ok(())
}

/// Resolve the single carrier that claims a source, refusing no-carrier and
/// ambiguous-carrier sources exactly as the load entries do.
fn resolve_carrier(src: &ModelSource) -> Result<&'static dyn Carrier, String> {
    let mut matches = crate::REGISTRY.iter().copied().filter(|c| c.probe(src));
    let carrier = matches
        .next()
        .ok_or_else(|| format!("no carrier for {}", src.describe()))?;
    if let Some(other) = matches.next() {
        return Err(format!(
            "ambiguous carrier dispatch for {}: '{}' and '{}' both claim it",
            src.describe(),
            carrier.name(),
            other.name()
        ));
    }
    Ok(carrier)
}

/// Read-only DFlash lm-head quant refusal: a draft is attached but the target's
/// lm_head/embed quant type is not admitted for the batched GEMM verify paths.
/// Mirrors the gemma4-entry pre-allocation check (lib.rs) so the refusal fires
/// at admission instead of after prior-model teardown.
fn df_lash_lm_head_admission(
    hfq: &hipfire_runtime::hfq::HfqFile,
    draft_path: Option<&str>,
    gpu_arch: &str,
) -> Result<(), String> {
    if draft_path.is_none() {
        return Ok(());
    }
    let lm_qt = hfq
        .tensor_data("lm_head.weight")
        .or_else(|| hfq.tensor_data("model.language_model.lm_head.weight"))
        .or_else(|| hfq.tensor_data("model.language_model.embed_tokens.weight"))
        .or_else(|| hfq.tensor_data("model.embed_tokens.weight"))
        .map(|(info, _)| info.quant_type);
    if !crate::dflash_lm_head_quant_supported(lm_qt, gpu_arch) {
        let qt_desc = match lm_qt {
            Some(qt) => format!("quant_type={qt}"),
            None => "no lm_head/embed_tokens tensor found".to_string(),
        };
        return Err(format!(
            "DFlash draft requested but target lm_head {qt_desc} is not supported \
             on gfx11+gfx12 WMMA ({gpu_arch})."
        ));
    }
    Ok(())
}

/// Expert-parallel VMM refusal, mirroring `load_model_ep_with_kv_mode`'s
/// per-arch dispatch: VMM is single-device, so the EP arches whose loaders
/// have no VMM path (Qwen3.5 5|6, MiniMax 10) refuse it. DeepSeek V4 (9) is
/// the one EP arch that serves vmm by design and stays vmm-capable here.
fn ep_vmm_refusal(arch_id: u32, kv_backend: KvBackend) -> Option<String> {
    (kv_backend == KvBackend::Vmm && matches!(arch_id, 5 | 6 | 10))
        .then(|| format!("KV backend '{}' requires tp=1", kv_backend.as_str()))
}

/// FLUX/Klein image-gen arch refusal: the trunk GEMM (`gemm_wmma_lds256`)
/// and `attention_flux_vtk/v2_wmma` use the gfx11
/// `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` intrinsic, which hipcc
/// rejects on gfx12 ("needs target feature wmma-256b-insts,wavefrontsize32").
/// Admits exactly the `has_wmma_w32` set (`arch_caps.rs:143-145`: `is_rdna3`,
/// NOT the `has_wmma_w32_gfx12` gfx12 variant) — pure on the `gpu_arch`
/// string because admission is read-only and never inits a GPU. `None` for
/// non-diffusion archs and for gfx11; `Some(reason)` otherwise, so the load
/// refuses before any allocation with the prior model still loaded.
fn flux_arch_refusal(arch_id: u32, gpu_arch: &str) -> Option<String> {
    if !matches!(arch_id, 40 | 45) {
        return None;
    }
    let gfx11_wmma_w32 = matches!(
        gpu_arch,
        "gfx1100" | "gfx1101" | "gfx1102" | "gfx1103" | "gfx1150" | "gfx1151" | "gfx1152"
    );
    (!gfx11_wmma_w32).then(|| {
        format!(
            "image generation (arch 40/45) requires RDNA3/3.5 (gfx11 wave32 WMMA); \
             detected {gpu_arch}. See docs/IMAGEGEN.md §1."
        )
    })
}

/// Read-only source admission: open the source, classify `arch_id` + vision,
/// decide the effective topology, and refuse every unsupported/contradictory
/// combination — without touching GPU state, VMM, or any prior model.
///
/// Refusals mirror the current-master daemon/loader refusals so no
/// currently-served route changes; they simply fire before destructive work.
/// Compatibility wrapper for callers that do not expose the optional
/// load-time controls.  Qwen4 still receives the conservative defaults.
pub fn admit_source(
    path: &str,
    tp: usize,
    pp: usize,
    kv_backend_override: Option<&str>,
    draft_path: Option<&str>,
    gpu_arch: &str,
    vision: Option<&str>,
    head: Option<&str>,
    max_seq: usize,
) -> Result<SourceAdmission, String> {
    admit_source_with_options(
        path,
        tp,
        pp,
        kv_backend_override,
        draft_path,
        gpu_arch,
        vision,
        head,
        max_seq,
        SourceAdmissionOptions::default(),
    )
}

pub fn admit_source_with_options(
    path: &str,
    tp: usize,
    pp: usize,
    kv_backend_override: Option<&str>,
    draft_path: Option<&str>,
    gpu_arch: &str,
    vision: Option<&str>,
    head: Option<&str>,
    max_seq: usize,
    options: SourceAdmissionOptions,
) -> Result<SourceAdmission, String> {
    let mut source = ModelSource::from_path(path)?;
    let arch_id = source
        .arch_id()
        .ok_or_else(|| format!("unrecognized source: {}", source.describe()))?;
    let is_dir = source.is_dir();
    let kv_backend: KvBackend = kv_backend_override
        .unwrap_or("contiguous")
        .parse()
        .map_err(|err| format!("{err}"))?;

    // FLUX/Klein need gfx11 wave32 WMMA (the trunk GEMM and vtk/v2 use the
    // gfx11 WMMA intrinsic hipcc rejects on gfx12). Refuse here — before the
    // topology branch and any allocation — so the prior model stays loaded.
    if let Some(refusal) = flux_arch_refusal(arch_id, gpu_arch) {
        return Err(refusal);
    }
    // Arch 16 is an executable local-path carrier, but only after its complete
    // source-only boundary succeeds.  Keep this before vision/head handling so
    // every refusal remains pre-allocation.
    let (topology, carrier) = if arch_id == QWEN4_ARCH_ID {
        if max_seq != 2048 {
            return Err(format!(
                "qwen4: max_seq must be exactly 2048 (got {max_seq})"
            ));
        }
        // The executable HFQM manifest always carries the validated one-layer
        // MTP head. Native execution is opt-in: only `Some(true)` expresses
        // the request to attach it; `None` and `Some(false)` remain AR-only.
        let native_mtp = qwen4_native_mtp_requested(options.spec);
        if draft_path.is_some()
            || options.gemma4_drafter
            || options.cask
            || options.state_quant
            || options.non_single_compute
            || options.pflash
            || options.spec.dflash.is_some_and(|enabled| enabled)
            || options.spec.dspark.is_some_and(|enabled| enabled)
            || options.spec.ngram_draft.is_some_and(|enabled| enabled)
            || qwen4_ddtree_requested(options.spec)
        {
            return Err(
                "qwen4: requested DFlash, DSpark, n-gram, DDTree, EAGLE, CASK, state-quant, PFlash, or non-Single option is unsupported"
                    .into(),
            );
        }
        if native_mtp
            && hipfire_runtime::config::retained_redline_default(
                gpu_arch, "qwen4", path, pp, tp, true,
            )
        {
            return Err(
                "qwen4: native MTP cannot be admitted with retained Redline; load the non-MQ4R HFQM artifact or disable MTP"
                    .into(),
            );
        }
        if kv_backend != KvBackend::Contiguous {
            return Err(format!(
                "qwen4: KV backend '{}' is unsupported; only contiguous is admitted",
                kv_backend.as_str()
            ));
        }
        admit_qwen4_source(
            &source,
            hipfire_arch_qwen4::EffectiveMesh::new(pp, tp, 1),
            InputModality::Text,
            native_mtp,
        )?;
        let carrier = resolve_carrier(&source)?;
        carrier.admit_topology(arch_id, is_dir, pp, kv_backend)?;
        (EffectiveTopology::Single, Some(carrier))
    } else {
        let (topology, carrier) = if tp > 1 {
            // Expert-parallel admission (HFQ-only). Mirrors
            // `load_model_ep_with_kv_mode`'s arch_id dispatch + per-arch VMM
            // refusal: DeepSeek V4 (9) serves vmm by design; Qwen3.5 (5|6) and
            // MiniMax (10) refuse it (single-device backend, no EP VMM path).
            if is_dir {
                return Err(
                "EP not supported for safetensors directory sources (load as a single HFQ file)"
                    .into(),
            );
            }
            if !matches!(arch_id, 5 | 6 | 9 | 10) {
                return Err(format!(
                "EP not supported for arch_id={arch_id} (expected 5|6 for Qwen3.5, 9 for DeepSeek V4 or 10 for MiniMax)"
            ));
            }
            if let Some(refusal) = ep_vmm_refusal(arch_id, kv_backend) {
                return Err(refusal);
            }
            // Qwen3.5 MoE has no EP serve path. Parse the retained source's
            // actual config before any caller can tear down its active model or
            // enter `Gpus::init_ep`; this must reuse the loader's established
            // refusal predicate and exact error text.
            if matches!(arch_id, 5 | 6) {
                let ModelSource::Hfq(hfq) = &source else {
                    return Err("EP qwen35 requires an HFQ source".to_string());
                };
                let config = hipfire_arch_qwen35::qwen35::config_from_hfq(hfq)
                    .map_err(|e| format!("qwen35 config: {e}"))?;
                if let Some(refusal) = crate::qwen35_ep_moe_refusal(arch_id, config.num_experts) {
                    return Err(refusal);
                }
            }
            (EffectiveTopology::Expert(tp), None)
        } else {
            // Single / pipeline-parallel via the carrier registry.
            let carrier = resolve_carrier(&source)?;
            if kv_backend == KvBackend::Vmm
                && !matches!(carrier.name(), "qwen35" | "deepseek4" | "muse_glimmer")
            {
                return Err(format!(
                "KV backend 'vmm' currently supports qwen3.5, deepseek4, and Muse Glimmer only (selected carrier: {})",
                carrier.name()
            ));
            }
            if kv_backend == KvBackend::Vmm && pp > 1 {
                return Err(
                "KV backend 'vmm' is single-device and does not support pipeline parallelism (pp>1); \
                 use a different kv_cache backend or load with pp=1"
                    .to_string(),
            );
            }
            carrier.admit_topology(arch_id, is_dir, pp, kv_backend)?;
            let topology = if pp > 1 {
                EffectiveTopology::Pipeline(pp)
            } else {
                EffectiveTopology::Single
            };
            (topology, Some(carrier))
        };
        (topology, carrier)
    };
    let mut has_vision = probe_vision(&source, arch_id)?;
    // Shared tower sidecar (registry `vision` slot / `params.vision` /
    // `HIPFIRE_VISION_SIDECAR`), validated fail-closed; a tower-bearing
    // sidecar promotes a tower-less trunk to VL.
    let vision_path = resolve_vision_sidecar(vision, arch_id, is_dir)?;
    if vision_path.is_some() {
        has_vision = true;
    }
    if let ModelSource::Hfq(hfq) = &source {
        df_lash_lm_head_admission(hfq, draft_path, gpu_arch)?;
        // Gemma 4 lowered min-context: refuse before teardown/alloc so a
        // small max_seq leaves the prior model serving. Eager stays exempt.
        if matches!(arch_id, 13 | 22) {
            let use_lowered = hipfire_arch_gemma4::gemma4_source_uses_lowered(hfq, false);
            hipfire_arch_gemma4::gemma4_context_admission(max_seq, use_lowered)?;
        }
    }
    // Head overlay (`params.head`): validated AND attached to the retained
    // base here, so the admitted source is already effective and loading
    // consumes it with no second open. Any refusal leaves the prior model
    // loaded — the overlay is in-memory only; no GPU state is touched.
    admit_head_overlay(head, &mut source, arch_id, topology)?;

    Ok(SourceAdmission {
        source,
        arch_id,
        is_dir,
        has_vision,
        topology,
        kv_backend,
        carrier,
        vision_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every Qwen3.5-family HF config embeds `vision_config` even for text-only
    /// quantized artifacts (the 27B/A3B production files all do). The tower
    /// tensor decides; config markers alone classify as the text backbone,
    /// never refuse. This is the contract from remediation `179a20d7f`.
    #[test]
    fn qwen35_config_marker_without_tower_is_text() {
        assert_eq!(classify_vision(5, false, true).unwrap(), false); // dense
        assert_eq!(classify_vision(6, false, true).unwrap(), false); // MoE
    }

    #[test]
    fn qwen35_tower_tensor_decides_vl() {
        assert_eq!(classify_vision(5, true, false).unwrap(), true);
        assert_eq!(classify_vision(6, true, false).unwrap(), true);
    }

    #[test]
    fn lfm2_tower_without_config_refuses() {
        assert!(classify_vision(11, true, false).is_err());
    }

    #[test]
    fn lfm2_config_without_tower_is_text() {
        assert_eq!(classify_vision(11, false, true).unwrap(), false);
    }

    #[test]
    fn non_vision_archs_are_never_vl() {
        for arch in [0u32, 1, 7, 9, 10, 22] {
            assert_eq!(classify_vision(arch, true, true).unwrap(), false);
        }
    }

    #[test]
    fn qwen4_conditional_generation_without_vision_is_text_capable() {
        assert_eq!(classify_vision(QWEN4_ARCH_ID, false, true).unwrap(), false);
        assert!(classify_vision(QWEN4_ARCH_ID, true, false).is_err());
    }

    #[test]
    fn qwen4_native_mtp_request_is_explicit() {
        let mut spec = SpecLoadCfg::default();
        assert!(!qwen4_native_mtp_requested(spec));

        spec.mtp = Some(false);
        assert!(!qwen4_native_mtp_requested(spec));

        spec.mtp = Some(true);
        assert!(qwen4_native_mtp_requested(spec));
    }

    mod qwen4_ddtree_admission {
        use super::super::*;
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

        fn write_probe(name: &str) -> std::path::PathBuf {
            let dir = std::env::temp_dir().join(format!(
                "hipfire-qwen4-admit-{}-{}",
                std::process::id(),
                name
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("{name}.hfq"));
            write_hfqm_package_mem(
                &path,
                QWEN4_ARCH_ID,
                "{}",
                &[HfqMemTensor {
                    name: "model.embed_tokens.weight".into(),
                    quant_type: 1,
                    shape: vec![4, 4],
                    group_size: 0,
                    data: vec![0u8; 32],
                }],
            )
            .unwrap();
            path
        }

        fn cleanup(path: &std::path::Path) {
            let _ = std::fs::remove_file(path);
            let _ = std::fs::remove_dir(path.parent().unwrap());
        }

        fn admit(path: &std::path::Path, spec: SpecLoadCfg) -> Result<SourceAdmission, String> {
            admit_source_with_options(
                path.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1151",
                None,
                None,
                2048,
                SourceAdmissionOptions {
                    spec,
                    ..Default::default()
                },
            )
        }

        #[test]
        fn resolved_defaults_pass_source_gate_but_active_overrides_refuse() {
            let path = write_probe("ddtree");
            let defaults = SpecLoadCfg {
                ddtree_budget: Some(0),
                ddtree_topk: Some(4),
                ..SpecLoadCfg::default()
            };
            let err = admit(&path, defaults)
                .map(|_| ())
                .expect_err("minimal probe must fail later on invalid Qwen4 config");
            assert!(
                !err.contains("requested DFlash, DSpark, n-gram, DDTree"),
                "resolved schema defaults must pass the DDTree source gate: {err}"
            );
            assert!(
                err.contains("config admission failed"),
                "defaults must reach Qwen4 config validation: {err}"
            );

            let active = SpecLoadCfg {
                ddtree_budget: Some(1),
                ddtree_topk: Some(4),
                ..defaults
            };
            let err = admit(&path, active)
                .map(|_| ())
                .expect_err("active DDTree must refuse");
            assert!(
                err.contains("requested DFlash, DSpark, n-gram, DDTree"),
                "active DDTree must be refused at source admission: {err}"
            );

            let nondefault_topk = SpecLoadCfg {
                ddtree_budget: Some(0),
                ddtree_topk: Some(5),
                ..defaults
            };
            let err = admit(&path, nondefault_topk)
                .map(|_| ())
                .expect_err("non-default top-K must refuse");
            assert!(
                err.contains("requested DFlash, DSpark, n-gram, DDTree"),
                "non-default DDTree top-K must be refused at source admission: {err}"
            );
            cleanup(&path);
        }
    }
    #[test]
    fn qwen4_request_refuses_image_video_and_non_single_before_source_access() {
        let single = hipfire_arch_qwen4::EffectiveMesh::single();
        assert!(qwen4_request_admission(single, InputModality::Text, true).is_ok());
        assert!(qwen4_request_admission(single, InputModality::Text, false).is_ok());
        assert!(qwen4_request_admission(single, InputModality::Image, true).is_err());
        assert!(qwen4_request_admission(single, InputModality::Video, true).is_err());
        assert!(qwen4_request_admission(
            hipfire_arch_qwen4::EffectiveMesh::new(2, 1, 1),
            InputModality::Text,
            true
        )
        .is_err());
    }

    /// The EP VMM refusal is per-arch, mirroring `load_model_ep_with_kv_mode`:
    /// Qwen3.5 (5|6) and MiniMax (10) refuse `vmm`; DeepSeek V4 (9) serves it.
    /// A blanket gate here would refuse the DS4 EP + vmm load master serves.
    #[test]
    fn ep_vmm_refusal_is_per_arch() {
        assert!(ep_vmm_refusal(5, KvBackend::Vmm).is_some());
        assert!(ep_vmm_refusal(6, KvBackend::Vmm).is_some());
        assert!(ep_vmm_refusal(10, KvBackend::Vmm).is_some());
        // DeepSeek V4 is vmm-capable.
        assert!(ep_vmm_refusal(9, KvBackend::Vmm).is_none());
        // Non-vmm backends are never refused.
        assert!(ep_vmm_refusal(5, KvBackend::Contiguous).is_none());
        assert!(ep_vmm_refusal(9, KvBackend::Contiguous).is_none());
    }

    /// FLUX/Klein admit exactly the gfx11 wave32 WMMA set: gfx1201 (and any
    /// non-gfx11 arch) refuses with the RDNA3/3.5 remedy before any
    /// allocation; gfx1100/gfx1151 admit; non-diffusion archs are untouched.
    #[test]
    fn flux_arch_refusal_is_gfx11_only() {
        for arch_id in [40u32, 45] {
            let err = flux_arch_refusal(arch_id, "gfx1201")
                .expect("gfx1201 FLUX/Klein must refuse at admission");
            assert!(err.contains("RDNA3/3.5"), "reason names remedy: {err}");
            assert!(err.contains("gfx11 wave32 WMMA"), "reason: {err}");
            assert!(err.contains("gfx1201"), "reason names detected arch: {err}");
            assert!(err.contains("IMAGEGEN.md"), "reason: {err}");
            assert!(flux_arch_refusal(arch_id, "gfx1200").is_some());
            assert_eq!(flux_arch_refusal(arch_id, "gfx1100"), None);
            assert_eq!(flux_arch_refusal(arch_id, "gfx1151"), None);
        }
        // Non-diffusion archs never hit this gate, on any arch string.
        assert_eq!(flux_arch_refusal(5, "gfx1201"), None);
        assert_eq!(flux_arch_refusal(9, "gfx1201"), None);
    }

    /// Vision-sidecar fixtures: minimal HFQ files via the in-memory writer.
    /// The trunk is a tower-less arch-5 text pack; the sidecar carries just
    /// the probe tensor. Admission is read-only — no GPU needed.
    mod vision_sidecar {
        use super::super::*;
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqMemTensor};

        fn write_hfq(name: &str, arch_id: u32, with_tower: bool) -> std::path::PathBuf {
            let dir = std::env::temp_dir().join(format!(
                "hipfire-vision-admit-{}-{}",
                std::process::id(),
                name
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("{name}.hfq"));
            let mut tensors = vec![HfqMemTensor {
                name: "model.embed_tokens.weight".into(),
                quant_type: 1,
                shape: vec![4, 4],
                group_size: 0,
                data: vec![0u8; 32],
            }];
            if with_tower {
                tensors.push(HfqMemTensor {
                    name: super::super::VISION_PROBE_TENSOR.into(),
                    quant_type: 1,
                    shape: vec![4, 4],
                    group_size: 0,
                    data: vec![0u8; 32],
                });
            }
            write_hfqm_package_mem(&path, arch_id, "{}", &tensors).unwrap();
            path
        }

        fn cleanup(path: &std::path::Path) {
            let _ = std::fs::remove_file(path);
            let _ = std::fs::remove_dir(path.parent().unwrap());
        }

        /// Tower-less trunk + tower-bearing sidecar admits as VL on qwen35.
        #[test]
        fn sidecar_promotes_tower_less_trunk_to_vl() {
            let trunk = write_hfq("a-trunk", 5, false);
            let sidecar = write_hfq("a-sidecar", 5, true);
            let admitted = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1100",
                Some(sidecar.to_str().unwrap()),
                None,
                4096,
            )
            .expect("tower sidecar must admit");
            assert!(admitted.has_vision, "sidecar tower promotes trunk to VL");
            assert_eq!(admitted.vision_path, Some(sidecar.clone()));
            assert!(
                admitted.carrier.is_some_and(|c| c.name() == "qwen35"),
                "trunk still routes to qwen35"
            );
            cleanup(&trunk);
            cleanup(&sidecar);
        }

        /// A sidecar without the tower tensor refuses with the pack remedy.
        #[test]
        fn sidecar_without_tower_refuses() {
            let trunk = write_hfq("b-trunk", 5, false);
            let sidecar = write_hfq("b-sidecar", 5, false);
            let err = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1100",
                Some(sidecar.to_str().unwrap()),
                None,
                4096,
            )
            .map(|_| ())
            .expect_err("tower-less sidecar must refuse");
            assert!(err.contains("no vision tower tensor"), "remedy: {err}");
            cleanup(&trunk);
            cleanup(&sidecar);
        }

        /// A sidecar stamped with the wrong arch refuses.
        #[test]
        fn sidecar_with_wrong_arch_refuses() {
            let trunk = write_hfq("c-trunk", 5, false);
            let sidecar = write_hfq("c-sidecar", 9, true);
            let err = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1100",
                Some(sidecar.to_str().unwrap()),
                None,
                4096,
            )
            .map(|_| ())
            .expect_err("wrong-arch sidecar must refuse");
            assert!(err.contains("arch_id=9"), "names the sidecar arch: {err}");
            cleanup(&trunk);
            cleanup(&sidecar);
        }
    }
    mod head_overlay {
        use super::super::*;
        use hipfire_runtime::hfq::{write_hfqm_package_mem, HfqFile, HfqMemTensor};

        fn write_tensors(
            name: &str,
            arch_id: u32,
            specs: &[(&str, u8, Vec<u32>, Vec<u8>)],
        ) -> std::path::PathBuf {
            let dir = std::env::temp_dir().join(format!(
                "hipfire-head-admit-{}-{}",
                std::process::id(),
                name
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("{name}.hfq"));
            let tensors: Vec<HfqMemTensor> = specs
                .iter()
                .map(|(n, qt, shape, data)| HfqMemTensor {
                    name: (*n).into(),
                    quant_type: *qt,
                    shape: shape.clone(),
                    group_size: 0,
                    data: data.clone(),
                })
                .collect();
            write_hfqm_package_mem(&path, arch_id, "{}", &tensors).unwrap();
            path
        }

        fn maple_trunk(name: &str) -> std::path::PathBuf {
            write_tensors(
                name,
                15,
                &[
                    ("model.embed_tokens.weight", 1, vec![4, 4], vec![0u8; 32]),
                    ("lm_head.weight", 3, vec![2, 4], vec![1u8; 32]),
                ],
            )
        }

        fn maple_head(name: &str, byte: u8) -> std::path::PathBuf {
            write_tensors(
                name,
                15,
                &[("lm_head.weight", 13, vec![2, 4], vec![byte; 32])],
            )
        }

        fn cleanup(paths: &[std::path::PathBuf]) {
            for path in paths {
                let _ = std::fs::remove_file(path);
                let _ = std::fs::remove_dir(path.parent().unwrap());
            }
        }

        fn head_bytes(admitted: &SourceAdmission) -> Vec<u8> {
            let ModelSource::Hfq(hfq) = &admitted.source else {
                panic!("expected HFQ source");
            };
            hfq.tensor_data("lm_head.weight")
                .map(|(_, d)| d.to_vec())
                .expect(" admitted source must serve lm_head.weight")
        }

        /// A valid maple head admits and shadows the base head in the
        /// retained source — loading consumes this, never reopens the file.
        #[test]
        fn head_on_maple_admits_and_shadows_base() {
            let trunk = maple_trunk("d-trunk");
            let head = maple_head("d-head", 7);
            let admitted = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1151",
                None,
                Some(head.to_str().unwrap()),
                4096,
            )
            .expect("valid maple head must admit");
            assert!(
                admitted.carrier.is_some_and(|c| c.name() == "maple"),
                "trunk still routes to maple"
            );
            assert_eq!(
                head_bytes(&admitted),
                vec![7u8; 32],
                "retained source serves the overlay head, not the base"
            );
            cleanup(&[trunk, head]);
        }

        /// Empty head string opts out exactly like vision/draft.
        #[test]
        fn empty_head_string_is_unset() {
            let trunk = maple_trunk("e-trunk");
            let admitted = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1151",
                None,
                Some(""),
                4096,
            )
            .expect("empty head must admit as unset");
            assert_eq!(
                head_bytes(&admitted),
                vec![1u8; 32],
                "unset head serves the baked base head"
            );
            cleanup(&[trunk]);
        }

        /// A head on a non-Maple trunk refuses instead of being ignored.
        #[test]
        fn head_on_non_maple_refuses() {
            let trunk = write_tensors(
                "f-trunk",
                5,
                &[
                    ("model.embed_tokens.weight", 1, vec![4, 4], vec![0u8; 32]),
                    ("lm_head.weight", 3, vec![2, 4], vec![1u8; 32]),
                ],
            );
            let head = maple_head("f-head", 7);
            let err = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1151",
                None,
                Some(head.to_str().unwrap()),
                4096,
            )
            .map(|_| ())
            .expect_err("non-maple head must refuse");
            assert!(err.contains("only serve Maple"), "refusal: {err}");
            cleanup(&[trunk, head]);
        }

        /// A head with expert-parallel topology refuses in preflight.
        #[test]
        fn head_on_ep_topology_refuses() {
            let trunk = maple_trunk("g-trunk");
            let mut base = ModelSource::from_path(trunk.to_str().unwrap()).expect("open trunk");
            let err = super::super::admit_head_overlay(
                Some("g-head"),
                &mut base,
                15,
                EffectiveTopology::Expert(2),
            )
            .expect_err("EP head must refuse");
            assert!(err.contains("single-device"), "refusal: {err}");
            cleanup(&[trunk]);
        }

        /// A head cannot stack on an installed REAP overlay (single slot).
        #[test]
        fn head_with_reap_overlay_refuses() {
            let trunk = maple_trunk("h-trunk");
            let plan = std::env::temp_dir()
                .join(format!("hipfire-head-admit-{}-h-plan", std::process::id()));
            std::fs::create_dir_all(&plan).unwrap();
            // Install a REAP overlay through the injected plan (deterministic:
            // no process-config snapshot involved).
            let plan_file = plan.join("overlay.hfq");
            let staged = write_tensors(
                "h-ov",
                15,
                &[("lm_head.weight", 8, vec![2, 4], vec![9u8; 32])],
            );
            std::fs::rename(&staged, &plan_file).unwrap();
            let head = maple_head("h-head", 7);
            let base = HfqFile::open_with_reap_plan(&trunk, Some(&plan)).expect("open trunk");
            assert!(base.has_overlay(), "REAP overlay must install");
            let mut source = ModelSource::Hfq(base);
            let err = super::super::admit_head_overlay(
                Some(head.to_str().unwrap()),
                &mut source,
                15,
                EffectiveTopology::Single,
            )
            .expect_err("REAP+head must refuse");
            assert!(err.contains("REAP"), "refusal: {err}");
            cleanup(&[trunk, head, plan_file, staged]);
            let _ = std::fs::remove_dir(&plan);
        }

        /// A truncated head (valid header/index, short payload) refuses.
        #[test]
        fn truncated_head_refuses() {
            let trunk = maple_trunk("i-trunk");
            let head = maple_head("i-head", 7);
            let len = std::fs::metadata(&head).unwrap().len();
            std::fs::OpenOptions::new()
                .write(true)
                .open(&head)
                .unwrap()
                .set_len(len - 10)
                .unwrap();
            let err = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1151",
                None,
                Some(head.to_str().unwrap()),
                4096,
            )
            .map(|_| ())
            .expect_err("truncated head must refuse");
            assert!(err.contains("truncated"), "refusal: {err}");
            cleanup(&[trunk, head]);
        }

        /// A head stamped for another arch refuses.
        #[test]
        fn head_with_wrong_arch_refuses() {
            let trunk = maple_trunk("j-trunk");
            let head = write_tensors(
                "j-head",
                9,
                &[("lm_head.weight", 13, vec![2, 4], vec![7u8; 32])],
            );
            let err = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1151",
                None,
                Some(head.to_str().unwrap()),
                4096,
            )
            .map(|_| ())
            .expect_err("wrong-arch head must refuse");
            assert!(err.contains("arch_id"), "refusal: {err}");
            cleanup(&[trunk, head]);
        }

        /// A full model passed as --head refuses (single-tensor guard).
        #[test]
        fn full_model_as_head_refuses() {
            let trunk = maple_trunk("k-trunk");
            let head = write_tensors(
                "k-head",
                15,
                &[
                    ("model.embed_tokens.weight", 1, vec![4, 4], vec![0u8; 32]),
                    ("lm_head.weight", 13, vec![2, 4], vec![7u8; 32]),
                ],
            );
            let err = admit_source(
                trunk.to_str().unwrap(),
                1,
                1,
                None,
                None,
                "gfx1151",
                None,
                Some(head.to_str().unwrap()),
                4096,
            )
            .map(|_| ())
            .expect_err("full model as head must refuse");
            assert!(err.contains("expected only"), "refusal: {err}");
            cleanup(&[trunk, head]);
        }
    }
}
