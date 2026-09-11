// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Pure Qwen4 capability and artifact admission.
//!
//! Admission is intentionally a value-only operation.  It validates source
//! shape, request modality, effective mesh, container format, and quantization
//! geometry before a loader is allowed to allocate weights or start a reader.

use crate::config::{Qwen4Config, ARCH_ID};
use std::fmt;

/// Request modality understood by the Qwen4 carrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputModality {
    Text,
    Image,
    Video,
}

/// Short alias for callers that use request rather than input terminology.
pub type RequestKind = InputModality;

/// Capability bits exposed by this first Qwen4 implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Qwen4Capability {
    Text,
    NativeMtp,
}

/// Qwen4 is text-only at the initial capability boundary.  Image and video
/// are represented as unsupported rather than silently routed to text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Qwen4Capabilities {
    pub text: bool,
    pub native_mtp: bool,
    pub image: bool,
    pub video: bool,
}

impl Qwen4Capabilities {
    pub const fn text_mtp() -> Self {
        Self {
            text: true,
            native_mtp: true,
            image: false,
            video: false,
        }
    }

    pub const fn supports_modality(self, modality: InputModality) -> bool {
        match modality {
            InputModality::Text => self.text,
            InputModality::Image => self.image,
            InputModality::Video => self.video,
        }
    }

    pub const fn supports(self, capability: Qwen4Capability) -> bool {
        match capability {
            Qwen4Capability::Text => self.text,
            Qwen4Capability::NativeMtp => self.native_mtp,
        }
    }
}

impl Default for Qwen4Capabilities {
    fn default() -> Self {
        Self::text_mtp()
    }
}

/// Effective (post-policy) mesh sizes.  A size of zero is retained so a
/// malformed plan is refused rather than normalized into a single-device
/// request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EffectiveMesh {
    pub pp: usize,
    pub tp: usize,
    pub ep: usize,
}

/// Alias used by mesh planners and tests.
pub type MeshShape = EffectiveMesh;

impl EffectiveMesh {
    pub const fn single() -> Self {
        Self {
            pp: 1,
            tp: 1,
            ep: 1,
        }
    }

    pub const fn new(pp: usize, tp: usize, ep: usize) -> Self {
        Self { pp, tp, ep }
    }

    pub const fn is_single(self) -> bool {
        self.pp == 1 && self.tp == 1 && self.ep == 1
    }
}

impl Default for EffectiveMesh {
    fn default() -> Self {
        Self::single()
    }
}

/// Wire/storage format admitted by the runtime Qwen4 artifact boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Qwen4ArtifactFormat {
    /// Versioned HFQM package containing all resident and external-row records.
    Hfqm,
    /// Safetensors is a source/conversion input, not a serving artifact.
    Safetensors,
    /// Explicitly retain unknown formats so they fail closed with a useful
    /// diagnostic instead of being mistaken for HFQM.
    Other(String),
}

/// Short format alias for callers sharing an artifact admission interface.
pub type ArtifactFormat = Qwen4ArtifactFormat;

impl Qwen4ArtifactFormat {
    pub const fn is_hfqm(&self) -> bool {
        matches!(self, Self::Hfqm)
    }
}

/// Quantized tensor format relevant to the Qwen4 mixed artifact recipe.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    /// MQ4G256V2, HFQM quant_type 44.
    Mq4G256V2,
    /// Q8F16/Q8_0, HFQM quant_type 3.
    Q8F16,
    /// Native bfloat16 record, HFQM quant_type 16.
    BF16,
    Other(String),
}

/// Shape/format identity for a routed expert projection.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct QuantTensorGeometry {
    pub format: QuantFormat,
    /// HFQM quant_type byte (44 for MQ4G256V2, 3 for Q8F16).
    pub quant_type: u8,
    /// Logical input width K of one projection row.
    pub k: usize,
    /// Quantization block width.  Q8F16 uses the Q8_0 32-element block.
    pub block_size: usize,
}

impl QuantTensorGeometry {
    pub const fn mq4g256v2(k: usize) -> Self {
        Self {
            format: QuantFormat::Mq4G256V2,
            quant_type: 44,
            k,
            block_size: 256,
        }
    }

    pub const fn q8f16(k: usize) -> Self {
        Self {
            format: QuantFormat::Q8F16,
            quant_type: 3,
            k,
            block_size: 32,
        }
    }
}

/// Exact per-projection and nonexpert storage recipe.  This is deliberately
/// more specific than a format name: a valid format with the wrong K must be
/// refused before dispatch.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Qwen4QuantGeometry {
    pub routed_gate_up: QuantTensorGeometry,
    pub routed_down: QuantTensorGeometry,
    pub nonexpert_dtype: QuantFormat,
    pub ple_row_dtype: QuantFormat,
}

/// Compatibility spelling for callers that call expert projections simply
/// gate/up and down.
pub type QuantGeometry = Qwen4QuantGeometry;

impl Qwen4QuantGeometry {
    /// Selected Halo candidate recipe: routed gate/up MQ4G256V2, routed down
    /// Q8F16, and byte-preserving BF16 nonexpert/PLE records.
    pub const fn halo_mixed() -> Self {
        Self {
            routed_gate_up: QuantTensorGeometry::mq4g256v2(2560),
            routed_down: QuantTensorGeometry::q8f16(640),
            nonexpert_dtype: QuantFormat::BF16,
            ple_row_dtype: QuantFormat::BF16,
        }
    }

    pub const fn expected() -> Self {
        Self::halo_mixed()
    }

    /// Validate the exact geometry selected for the initial Qwen4 artifact.
    pub fn validate(&self) -> Result<(), QuantAdmissionError> {
        let expected = Self::expected();
        if self.routed_gate_up != expected.routed_gate_up {
            return Err(QuantAdmissionError::WrongGeometry {
                component: "routed gate/up",
                expected: expected.routed_gate_up,
                got: self.routed_gate_up.clone(),
            });
        }
        if self.routed_down != expected.routed_down {
            return Err(QuantAdmissionError::WrongGeometry {
                component: "routed down",
                expected: expected.routed_down,
                got: self.routed_down.clone(),
            });
        }
        if self.nonexpert_dtype != expected.nonexpert_dtype {
            return Err(QuantAdmissionError::WrongDtype {
                component: "nonexpert",
                expected: expected.nonexpert_dtype,
                got: self.nonexpert_dtype.clone(),
            });
        }
        if self.ple_row_dtype != expected.ple_row_dtype {
            return Err(QuantAdmissionError::WrongDtype {
                component: "PLE rows",
                expected: expected.ple_row_dtype,
                got: self.ple_row_dtype.clone(),
            });
        }
        Ok(())
    }
}

/// One artifact identity presented to admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4Artifact {
    pub format: Qwen4ArtifactFormat,
    pub quant_geometry: Qwen4QuantGeometry,
}

impl Qwen4Artifact {
    pub const fn new(format: Qwen4ArtifactFormat, quant_geometry: Qwen4QuantGeometry) -> Self {
        Self {
            format,
            quant_geometry,
        }
    }

    pub const fn halo_hfqm() -> Self {
        Self::new(Qwen4ArtifactFormat::Hfqm, Qwen4QuantGeometry::halo_mixed())
    }
}

/// Pure request to the Qwen4 admission boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4AdmissionRequest {
    pub modality: InputModality,
    pub native_mtp: bool,
    pub effective_mesh: EffectiveMesh,
    pub artifact: Qwen4Artifact,
}

impl Qwen4AdmissionRequest {
    pub const fn text_ar(effective_mesh: EffectiveMesh, artifact: Qwen4Artifact) -> Self {
        Self {
            modality: InputModality::Text,
            native_mtp: false,
            effective_mesh,
            artifact,
        }
    }

    pub const fn text_mtp(effective_mesh: EffectiveMesh, artifact: Qwen4Artifact) -> Self {
        Self {
            modality: InputModality::Text,
            native_mtp: true,
            effective_mesh,
            artifact,
        }
    }
}

/// Why a request was refused.  Every variant is detected before allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionError {
    InvalidConfig(String),
    UnsupportedModality(InputModality),
    UnsupportedCapability(Qwen4Capability),
    NonSingleMesh(EffectiveMesh),
    UnsupportedFormat(Qwen4ArtifactFormat),
    WrongGeometry {
        component: &'static str,
        expected: QuantTensorGeometry,
        got: QuantTensorGeometry,
    },
    WrongDtype {
        component: &'static str,
        expected: QuantFormat,
        got: QuantFormat,
    },
}

/// Alias retaining the distinction between format/quant admission errors and
/// config parser errors for callers that expose one error type.
pub type Qwen4AdmissionError = AdmissionError;

impl fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(error) => write!(f, "qwen4 admission: invalid config: {error}"),
            Self::UnsupportedModality(modality) => {
                write!(f, "qwen4 admission: unsupported modality {modality:?}")
            }
            Self::UnsupportedCapability(capability) => {
                write!(f, "qwen4 admission: unsupported capability {capability:?}")
            }
            Self::NonSingleMesh(mesh) => write!(
                f,
                "qwen4 admission: only Single is admitted (pp={}, tp={}, ep={})",
                mesh.pp, mesh.tp, mesh.ep
            ),
            Self::UnsupportedFormat(format) => {
                write!(f, "qwen4 admission: unsupported artifact format {format:?}")
            }
            Self::WrongGeometry {
                component,
                expected,
                got,
            } => write!(
                f,
                "qwen4 admission: {component} geometry {:?} does not match {:?}",
                got, expected
            ),
            Self::WrongDtype {
                component,
                expected,
                got,
            } => write!(
                f,
                "qwen4 admission: {component} dtype {got:?} does not match {expected:?}"
            ),
        }
    }
}

impl std::error::Error for AdmissionError {}

/// Successful, immutable result passed to a later loader/carrier.  It carries
/// no GPU or reader handle and therefore cannot imply production admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4Admission {
    pub architecture_id: u32,
    pub capabilities: Qwen4Capabilities,
    pub effective_mesh: EffectiveMesh,
    pub native_mtp: bool,
}

impl Qwen4Admission {
    /// Validate all pure boundaries in order: config, capability, mesh,
    /// container, then exact quant recipe.
    pub fn admit(
        config: &Qwen4Config,
        request: &Qwen4AdmissionRequest,
    ) -> Result<Self, AdmissionError> {
        config.validate().map_err(AdmissionError::InvalidConfig)?;
        let capabilities = Qwen4Capabilities::text_mtp();
        if !capabilities.supports_modality(request.modality) {
            return Err(AdmissionError::UnsupportedModality(request.modality));
        }
        if request.native_mtp && !capabilities.supports(Qwen4Capability::NativeMtp) {
            return Err(AdmissionError::UnsupportedCapability(
                Qwen4Capability::NativeMtp,
            ));
        }
        if !request.effective_mesh.is_single() {
            return Err(AdmissionError::NonSingleMesh(request.effective_mesh));
        }
        if !request.artifact.format.is_hfqm() {
            return Err(AdmissionError::UnsupportedFormat(
                request.artifact.format.clone(),
            ));
        }
        request
            .artifact
            .quant_geometry
            .validate()
            .map_err(|error| match error {
                QuantAdmissionError::WrongGeometry {
                    component,
                    expected,
                    got,
                } => AdmissionError::WrongGeometry {
                    component,
                    expected,
                    got,
                },
                QuantAdmissionError::WrongDtype {
                    component,
                    expected,
                    got,
                } => AdmissionError::WrongDtype {
                    component,
                    expected,
                    got,
                },
            })?;
        Ok(Self {
            architecture_id: ARCH_ID,
            capabilities,
            effective_mesh: request.effective_mesh,
            native_mtp: request.native_mtp,
        })
    }

    /// Free-function style entry point for callers that do not retain an
    /// admission object as a type.
    pub fn admit_text_mtp(
        config: &Qwen4Config,
        request: &Qwen4AdmissionRequest,
    ) -> Result<Self, AdmissionError> {
        Self::admit(config, request)
    }
}

/// Errors local to quant geometry validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantAdmissionError {
    WrongGeometry {
        component: &'static str,
        expected: QuantTensorGeometry,
        got: QuantTensorGeometry,
    },
    WrongDtype {
        component: &'static str,
        expected: QuantFormat,
        got: QuantFormat,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Qwen4Config;
    use serde_json::json;

    fn config() -> Qwen4Config {
        let layers: Vec<_> = (0..48)
            .map(|idx| {
                if idx % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        Qwen4Config::from_value(&json!({
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text", "dtype": "bfloat16",
                "hidden_size": 2560, "vocab_size": 248320, "num_hidden_layers": 48,
                "max_position_embeddings": 262144, "num_attention_heads": 24,
                "num_key_value_heads": 2, "head_dim": 256, "partial_rotary_factor": 0.25,
                "rope_parameters": {"rope_theta": 10000000.0}, "attention_bias": false,
                "layer_types": layers, "full_attention_interval": 4,
                "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4, "mamba_ssm_dtype": "float32",
                "indexer_n_heads": 4, "indexer_kv_heads": 1, "indexer_head_dim": 128,
                "indexer_budget": 2048, "indexer_compress_ratio": 4,
                "num_experts": 512, "num_experts_per_tok": 10,
                "moe_intermediate_size": 640, "shared_expert_intermediate_size": 640,
                "hc_count": 4, "hc_lowrank": 320, "ple_layer_ids": [2],
                "ple_conv_kernel_size": 4, "ple_embed_dim": 2560,
                "split_ngram_parts": 128, "heads_per_ngram": 8, "ngram_size": 3,
                "ngram_vocab_size_base": 20000000, "make_ngram_vocab_size_divisible_by": 128,
                "eos_token_id": 248044, "tie_word_embeddings": false,
                "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": false,
                "output_gate_type": "sigmoid",
                "mtp": {"hybrid": true, "layer_types": ["full_attention"],
                        "num_hidden_layers": 1, "rope_theta": 10000000.0}
            }
        }))
        .unwrap()
    }

    #[test]
    fn text_mtp_single_hfqm_is_admitted() {
        let request =
            Qwen4AdmissionRequest::text_mtp(EffectiveMesh::single(), Qwen4Artifact::halo_hfqm());
        let admission = Qwen4Admission::admit(&config(), &request).unwrap();
        assert_eq!(admission.architecture_id, ARCH_ID);
        assert!(admission.capabilities.text);
        assert!(admission.capabilities.native_mtp);
    }

    #[test]
    fn images_videos_meshes_and_formats_fail_closed() {
        let mut request =
            Qwen4AdmissionRequest::text_ar(EffectiveMesh::single(), Qwen4Artifact::halo_hfqm());
        request.modality = InputModality::Image;
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::UnsupportedModality(InputModality::Image))
        ));
        request.modality = InputModality::Video;
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::UnsupportedModality(InputModality::Video))
        ));
        request.modality = InputModality::Text;
        request.effective_mesh = EffectiveMesh::new(1, 2, 1);
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::NonSingleMesh(_))
        ));
        request.effective_mesh = EffectiveMesh::single();
        request.artifact.format = Qwen4ArtifactFormat::Safetensors;
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::UnsupportedFormat(
                Qwen4ArtifactFormat::Safetensors
            ))
        ));
    }

    #[test]
    fn wrong_quant_geometry_is_refused() {
        let mut artifact = Qwen4Artifact::halo_hfqm();
        artifact.quant_geometry.routed_down.k = 2560;
        let request = Qwen4AdmissionRequest::text_ar(EffectiveMesh::single(), artifact);
        assert!(matches!(
            Qwen4Admission::admit(&config(), &request),
            Err(AdmissionError::WrongGeometry {
                component: "routed down",
                ..
            })
        ));
    }
}
