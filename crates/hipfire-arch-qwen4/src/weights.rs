// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! Typed Qwen4 manifest declarations and GPU-owned assembly.
//!
//! The names in this module are the names emitted by the pinned
//! `Qwen/Qwen3.8-Flash-Next` checkpoint.  In particular, the decoder uses the
//! `model.language_model.*` namespace, QSA keeps separate q/k/v projections,
//! and the PLE table is a set of numerically suffixed external rows.  The
//! runtime's [`WeightLoadTransaction`] remains the only allocation owner until
//! this module finalizes handles into [`Qwen4Weights`].

use crate::config::{LayerType, Qwen4Config};
#[cfg(test)]
use crate::config::{Qwen4MtpConfig, RecurrentStateDType, SourceDType};
use hipfire_runtime::model_source::SourceRangeDescriptor;
#[cfg(test)]
use hipfire_runtime::weight_manifest::WeightResidency;
use hipfire_runtime::weight_manifest::{
    DTypeConstraint, PinTarget, PlacementHint, ShardPolicy, StateEntry, StateKind, WeightEntry,
};
use hipfire_runtime::weight_store::{TakenWeight, WeightHandle, WeightLoadTransaction};
use rdna_compute::{DType, Gpu};
use std::collections::BTreeSet;
use std::fmt;

/// Mixed Halo artifact recipe: all routed gate/up records use MQ4G256V2.
pub const ROUTED_GATE_UP_DTYPE: DType = DType::MQ4G256V2;
/// Mixed Halo artifact recipe: all routed down records use the qt=3 Q8/F16
/// runtime representation, carried by the existing Q8_0 wire tag.
pub const ROUTED_DOWN_DTYPE: DType = DType::Q8_0;
pub const PLE_SHARD_ROWS: usize = 2_500_012;
pub const PLE_ROW_WIDTH: usize = 160;
pub const PLE_SHARD_COUNT: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum TensorRole {
    TokenEmbedding,
    LanguageHead,
    HyperConnection,
    HyperConnectionNorm,
    HyperConnectionBlockInject,
    HyperConnectionDown,
    HyperConnectionUp,
    GdnQkv,
    GdnZ,
    GdnA,
    GdnB,
    GdnDtBias,
    GdnALog,
    GdnNorm,
    GdnConv,
    GdnOutput,
    QsaQ,
    QsaK,
    QsaV,
    QsaOutput,
    QsaQNorm,
    QsaKNorm,
    QsaIndexerQk,
    QsaIndexerQNorm,
    QsaIndexerKNorm,
    Router,
    SharedExpertGateScalar,
    SharedExpertGate,
    SharedExpertUp,
    SharedExpertDown,
    RoutedGateUp,
    RoutedDown,
    PleShard,
    PleKey,
    PleValue,
    PleNorm,
    PleConv,
    PleMetadata,
    MtpEmbeddingProjection,
    MtpHiddenProjection,
    MtpNorm,
}

/// Typed logical reference.  The runtime manifest remains authoritative for
/// placement, aliases, source dtype, and residency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRef {
    pub name: String,
    pub role: TensorRole,
    pub layer: Option<usize>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorRef {
    pub fn new(
        name: impl Into<String>,
        role: TensorRole,
        layer: Option<usize>,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Result<Self, WeightError> {
        let name = name.into();
        if name.is_empty() || shape.is_empty() || shape.iter().any(|&dim| dim == 0) {
            return Err(WeightError::InvalidShape { name, shape });
        }
        Ok(Self {
            name,
            role,
            layer,
            shape,
            dtype,
        })
    }
}

/// One external PLE descriptor, borrowed from the canonical fulfilled
/// transaction.  It has no resident handle by design.
#[derive(Clone, Debug)]
pub struct ExternalRowsRef {
    pub name: String,
    pub layer: usize,
    pub row_bytes: usize,
    pub physical_rows: usize,
    pub valid_rows: usize,
}

impl ExternalRowsRef {
    pub fn validate_descriptor(
        &self,
        descriptor: &SourceRangeDescriptor,
    ) -> Result<(), WeightError> {
        let expected_len = self
            .row_bytes
            .checked_mul(self.physical_rows)
            .ok_or_else(|| WeightError::ShapeOverflow(self.name.clone()))?;
        if descriptor.length != expected_len as u64 {
            return Err(WeightError::RangeLength {
                name: self.name.clone(),
                expected: expected_len as u64,
                actual: descriptor.length,
            });
        }
        if descriptor.logical_shape != [self.physical_rows, PLE_ROW_WIDTH]
            || !descriptor.dtype.eq_ignore_ascii_case("BF16")
        {
            return Err(WeightError::DescriptorMismatch(self.name.clone()));
        }
        if self.valid_rows == 0 || self.valid_rows > self.physical_rows {
            return Err(WeightError::InvalidValidRows {
                name: self.name.clone(),
                valid_rows: self.valid_rows,
                physical_rows: self.physical_rows,
            });
        }
        Ok(())
    }
}

/// I64 metadata is retained as an exact source declaration rather than being
/// coerced into a floating-point `WeightEntry`.  The HFQM writer serializes
/// these arrays under its versioned `qwen4_ple` object; they never become GPU
/// resident weight handles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetadataTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub source_dtype: String,
}

impl MetadataTensor {
    fn i64(name: impl Into<String>, shape: Vec<usize>) -> Self {
        Self {
            name: name.into(),
            shape,
            source_dtype: "I64".to_string(),
        }
    }
}

/// Pure logical Qwen4 manifest.  No GPU, source, or allocator is touched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen4Manifest {
    pub weights: Vec<WeightEntry>,
    pub state: Vec<StateEntry>,
    pub metadata: Vec<MetadataTensor>,
}

impl Qwen4Manifest {
    pub fn build(config: &Qwen4Config) -> Result<Self, WeightError> {
        config.validate().map_err(WeightError::Config)?;
        let bf16 = DTypeConstraint::source_exact(DType::BF16);
        // The source checkpoint is BF16, while a converted HFQM artifact
        // presents routed records in their physical quantized dtype.  Admit
        // exactly those two representations; never label a quant record BF16.
        let quant_gate_up =
            DTypeConstraint::source_from_sources(vec![DType::BF16, ROUTED_GATE_UP_DTYPE]);
        let quant_down = DTypeConstraint::source_from_sources(vec![DType::BF16, ROUTED_DOWN_DTYPE]);
        let model = |name: &str,
                     shape: Vec<usize>,
                     dtype: DType,
                     policy: ShardPolicy,
                     source: &DTypeConstraint| {
            WeightEntry::model_with_dtype_constraint(name, shape, dtype, source.clone(), policy)
        };
        let layer = |name: &str,
                     layer_idx: usize,
                     shape: Vec<usize>,
                     dtype: DType,
                     policy: ShardPolicy,
                     source: &DTypeConstraint| {
            WeightEntry::layer_with_dtype_constraint(
                name,
                layer_idx,
                shape,
                dtype,
                source.clone(),
                policy,
            )
        };
        let hidden = config.hidden_size;
        let hc_wide = config
            .hc_count
            .checked_mul(hidden)
            .ok_or_else(|| WeightError::ShapeOverflow("hyper-connection width".to_string()))?;
        let hc_rank = config.hc_lowrank;
        let q_dim = config
            .num_attention_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| WeightError::ShapeOverflow("QSA q width".to_string()))?;
        let kv_dim = config
            .num_key_value_heads
            .checked_mul(config.head_dim)
            .ok_or_else(|| WeightError::ShapeOverflow("QSA kv width".to_string()))?;
        let indexer_qk_dim = (config.indexer_n_heads + config.indexer_kv_heads)
            .checked_mul(config.indexer_head_dim)
            .ok_or_else(|| WeightError::ShapeOverflow("QSA indexer width".to_string()))?;
        let gdn_qk = config
            .linear_num_key_heads
            .checked_mul(config.linear_key_head_dim)
            .ok_or_else(|| WeightError::ShapeOverflow("GDN key width".to_string()))?;
        let gdn_v = config
            .linear_num_value_heads
            .checked_mul(config.linear_value_head_dim)
            .ok_or_else(|| WeightError::ShapeOverflow("GDN value width".to_string()))?;
        let gdn_qkv = 2usize
            .checked_mul(gdn_qk)
            .and_then(|v| v.checked_add(gdn_v))
            .ok_or_else(|| WeightError::ShapeOverflow("GDN qkv width".to_string()))?;
        let ple_channels = config
            .ple_embed_dim
            .checked_mul(config.hc_count)
            .ok_or_else(|| WeightError::ShapeOverflow("PLE channel width".to_string()))?;

        let mut weights = Vec::with_capacity(1_330);
        weights.push(
            model(
                "model.language_model.embed_tokens.weight",
                vec![config.vocab_size, hidden],
                DType::BF16,
                ShardPolicy::Pin(PinTarget::Embed),
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Embed)),
        );
        weights.push(
            model(
                "lm_head.weight",
                vec![config.vocab_size, hidden],
                DType::BF16,
                ShardPolicy::Pin(PinTarget::Output),
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        // MTP aliases exactly the input embedding.  The target has no MTP
        // lm_head tensor; the distinct trunk lm_head remains the output head.
        weights.push(
            model(
                "mtp.embed_tokens.weight",
                vec![config.vocab_size, hidden],
                DType::BF16,
                ShardPolicy::Tied {
                    source: "model.language_model.embed_tokens.weight".to_string(),
                },
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Embed)),
        );
        // Final HC mixer is the only trunk-level post-layer transform.  Its
        // hc_norm is the final normalization; no invented RMSNorm is added.
        weights.push(
            model(
                "model.language_model.hyper_connection_mixer.hc_norm.weight",
                vec![hc_wide],
                DType::BF16,
                ShardPolicy::Pin(PinTarget::Output),
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        weights.push(
            model(
                "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
                vec![hc_rank, hc_wide],
                DType::BF16,
                ShardPolicy::ColumnShard { axis: 0 },
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );
        weights.push(
            model(
                "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
                vec![hc_wide, hc_rank],
                DType::BF16,
                ShardPolicy::RowShard { axis: 1 },
                &bf16,
            )
            .with_placement(PlacementHint::Pin(PinTarget::Output)),
        );

        for (layer_idx, kind) in config.layer_types.iter().copied().enumerate() {
            let prefix = format!("model.language_model.layers.{layer_idx}");
            push_hyper_connection_entries(
                &mut weights,
                &layer,
                &prefix,
                layer_idx,
                "attn_hyper_connection",
                hidden,
                hc_wide,
                hc_rank,
                &bf16,
            );
            push_hyper_connection_entries(
                &mut weights,
                &layer,
                &prefix,
                layer_idx,
                "mlp_hyper_connection",
                hidden,
                hc_wide,
                hc_rank,
                &bf16,
            );
            match kind {
                LayerType::LinearAttention => {
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.A_log"),
                        layer_idx,
                        vec![config.linear_num_value_heads],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.conv1d.weight"),
                        layer_idx,
                        vec![gdn_qkv, 1, config.linear_conv_kernel_dim],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.dt_bias"),
                        layer_idx,
                        vec![config.linear_num_value_heads],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_a.weight"),
                        layer_idx,
                        vec![config.linear_num_value_heads, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_b.weight"),
                        layer_idx,
                        vec![config.linear_num_value_heads, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                        layer_idx,
                        vec![gdn_qkv, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.in_proj_z.weight"),
                        layer_idx,
                        vec![gdn_v, hidden],
                        DType::BF16,
                        ShardPolicy::ColumnShard { axis: 0 },
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.norm.weight"),
                        layer_idx,
                        vec![config.linear_value_head_dim],
                        DType::BF16,
                        ShardPolicy::Replicate,
                        &bf16,
                    ));
                    weights.push(layer(
                        &format!("{prefix}.linear_attn.out_proj.weight"),
                        layer_idx,
                        vec![hidden, gdn_v],
                        DType::BF16,
                        ShardPolicy::RowShard { axis: 1 },
                        &bf16,
                    ));
                }
                LayerType::FullAttention => {
                    push_qsa_entries(
                        &mut weights,
                        &layer,
                        &prefix,
                        layer_idx,
                        hidden,
                        q_dim,
                        kv_dim,
                        indexer_qk_dim,
                        config,
                        &bf16,
                    );
                }
            }
            push_moe_entries(
                &mut weights,
                &layer,
                &prefix,
                layer_idx,
                hidden,
                config,
                &bf16,
                &quant_gate_up,
                &quant_down,
            );
            if layer_idx == 1 {
                push_ple_entries(
                    &mut weights,
                    &layer,
                    &prefix,
                    layer_idx,
                    hidden,
                    ple_channels,
                    config,
                    &bf16,
                );
            }
        }

        // Native MTP has one full-attention/MoE/HC layer and no PLE.  Keep the
        // source's exact names and shapes; fc_hidden is [H,H] in this
        // checkpoint even though its consumer widens the activation.
        weights.push(mtp_model(
            "mtp.fc_embedding.weight",
            vec![hidden, hidden],
            DType::BF16,
            ShardPolicy::ColumnShard { axis: 0 },
            &bf16,
        ));
        weights.push(mtp_model(
            "mtp.fc_hidden.weight",
            vec![hidden, hidden],
            DType::BF16,
            ShardPolicy::ColumnShard { axis: 0 },
            &bf16,
        ));
        weights.push(mtp_model(
            "mtp.pre_fc_norm_embedding.weight",
            vec![hidden],
            DType::BF16,
            ShardPolicy::Replicate,
            &bf16,
        ));
        weights.push(mtp_model(
            "mtp.pre_fc_norm_hidden.weight",
            vec![hc_wide],
            DType::BF16,
            ShardPolicy::Replicate,
            &bf16,
        ));
        push_mtp_hyper_connection_entries(&mut weights, hc_wide, hc_rank, &bf16);
        push_hyper_connection_entries(
            &mut weights,
            &mtp_layer,
            "mtp.layers.0",
            0,
            "attn_hyper_connection",
            hidden,
            hc_wide,
            hc_rank,
            &bf16,
        );
        push_hyper_connection_entries(
            &mut weights,
            &mtp_layer,
            "mtp.layers.0",
            0,
            "mlp_hyper_connection",
            hidden,
            hc_wide,
            hc_rank,
            &bf16,
        );
        push_qsa_entries(
            &mut weights,
            &mtp_layer,
            "mtp.layers.0",
            0,
            hidden,
            q_dim,
            kv_dim,
            indexer_qk_dim,
            config,
            &bf16,
        );
        push_moe_entries(
            &mut weights,
            &mtp_layer,
            "mtp.layers.0",
            0,
            hidden,
            config,
            &bf16,
            &quant_gate_up,
            &quant_down,
        );

        // These declarations preserve the exact I64 payload identities for
        // the qwen4_ple container.  They are deliberately not WeightEntry
        // values because no GPU tensor or float dtype may represent them.
        let metadata = vec![
            MetadataTensor::i64(
                "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
                vec![3],
            ),
            MetadataTensor::i64(
                "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
                vec![PLE_HEAD_COUNT],
            ),
            MetadataTensor::i64(
                "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
                vec![PLE_HEAD_COUNT],
            ),
        ];

        let mut state = Vec::new();
        for (layer_idx, kind) in config.layer_types.iter().copied().enumerate() {
            state.push(match kind {
                LayerType::LinearAttention => StateEntry::new(StateKind::Recurrent, layer_idx),
                LayerType::FullAttention => StateEntry::new(
                    StateKind::Kv {
                        quant: "f32-qsa-full-raw-pooled".into(),
                    },
                    layer_idx,
                ),
            });
            if kind == LayerType::LinearAttention {
                state.push(StateEntry::new(StateKind::Conv, layer_idx));
            }
        }
        // PLE is injected at zero-based layer 1 and retains nine rows of
        // depthwise-convolution history.  StateKind has no family-local name;
        // the architecture state owner keeps this entry distinct by type.
        state.push(StateEntry::new(StateKind::Conv, 1));
        Ok(Self {
            weights,
            state,
            metadata,
        })
    }

    pub fn entry(&self, name: &str, layer: Option<usize>) -> Option<&WeightEntry> {
        self.weights
            .iter()
            .find(|entry| entry.name == name && entry.layer == layer)
    }

    pub fn external_entries(&self) -> impl Iterator<Item = &WeightEntry> {
        self.weights
            .iter()
            .filter(|entry| entry.residency.is_external())
    }

    pub fn resident_keys(&self) -> impl Iterator<Item = (&str, Option<usize>)> {
        self.weights
            .iter()
            .filter(|entry| !entry.residency.is_external())
            .map(|entry| (entry.name.as_str(), entry.layer))
    }
    /// Number of source tensor records represented by this text manifest.
    /// The only non-source record is the tied MTP embedding alias; the three
    /// exact I64 PLE metadata arrays are counted from `metadata`.
    pub fn source_tensor_count(&self) -> usize {
        let aliases = self
            .weights
            .iter()
            .filter(|entry| matches!(entry.policy, ShardPolicy::Tied { .. }))
            .count();
        self.weights
            .len()
            .saturating_sub(aliases)
            .saturating_add(self.metadata.len())
    }
}

fn mtp_model(
    name: &str,
    shape: Vec<usize>,
    dtype: DType,
    policy: ShardPolicy,
    source: &DTypeConstraint,
) -> WeightEntry {
    WeightEntry::model_with_dtype_constraint(name, shape, dtype, source.clone(), policy)
}

fn mtp_layer(
    name: &str,
    _layer: usize,
    shape: Vec<usize>,
    dtype: DType,
    policy: ShardPolicy,
    source: &DTypeConstraint,
) -> WeightEntry {
    // MTP names live in a separate model namespace, not the trunk's layer
    // placement scope.  Keep their manifest identity layerless.
    mtp_model(name, shape, dtype, policy, source)
}

fn push_hyper_connection_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    name: &str,
    hidden: usize,
    hc_wide: usize,
    hc_rank: usize,
    bf16: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let prefix = format!("{prefix}.{name}");
    weights.push(layer(
        &format!("{prefix}.block_inject_weight.weight"),
        layer_idx,
        vec![4, hc_wide],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{prefix}.hc_norm.weight"),
        layer_idx,
        vec![hc_wide],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{prefix}.input_mix_weight_down.weight"),
        layer_idx,
        vec![hc_rank, hc_wide],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(layer(
        &format!("{prefix}.input_mix_weight_up.weight"),
        layer_idx,
        vec![hc_wide, hc_rank],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
    let _ = hidden;
}

fn push_mtp_hyper_connection_entries(
    weights: &mut Vec<WeightEntry>,
    hc_wide: usize,
    hc_rank: usize,
    bf16: &DTypeConstraint,
) {
    weights.push(mtp_model(
        "mtp.hyper_connection_mixer.hc_norm.weight",
        vec![hc_wide],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(mtp_model(
        "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
        vec![hc_rank, hc_wide],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(mtp_model(
        "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
        vec![hc_wide, hc_rank],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
}

fn push_qsa_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    indexer_qk_dim: usize,
    config: &Qwen4Config,
    bf16: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let push =
        |weights: &mut Vec<WeightEntry>, suffix: &str, shape: Vec<usize>, policy: ShardPolicy| {
            weights.push(layer(
                &format!("{prefix}.self_attn.{suffix}"),
                layer_idx,
                shape,
                DType::BF16,
                policy,
                bf16,
            ));
        };
    push(
        weights,
        "indexer.index_qk_proj.weight",
        vec![indexer_qk_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
    push(
        weights,
        "indexer.k_layernorm.weight",
        vec![config.indexer_head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "indexer.q_layernorm.weight",
        vec![config.indexer_head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "k_norm.weight",
        vec![config.head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "k_proj.weight",
        vec![kv_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
    push(
        weights,
        "o_proj.weight",
        vec![hidden, q_dim],
        ShardPolicy::RowShard { axis: 1 },
    );
    push(
        weights,
        "q_norm.weight",
        vec![config.head_dim],
        ShardPolicy::Replicate,
    );
    push(
        weights,
        "q_proj.weight",
        vec![2 * q_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
    push(
        weights,
        "v_proj.weight",
        vec![kv_dim, hidden],
        ShardPolicy::ColumnShard { axis: 0 },
    );
}

fn push_moe_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    hidden: usize,
    config: &Qwen4Config,
    bf16: &DTypeConstraint,
    quant_gate_up: &DTypeConstraint,
    quant_down: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let moe = format!("{prefix}.mlp");
    weights.push(layer(
        &format!("{moe}.experts.down_proj"),
        layer_idx,
        vec![config.num_experts, hidden, config.moe_intermediate_size],
        ROUTED_DOWN_DTYPE,
        ShardPolicy::ExpertTensorSharded {
            n_experts: config.num_experts,
            inner: Box::new(ShardPolicy::RowShard { axis: 2 }),
        },
        quant_down,
    ));
    weights.push(layer(
        &format!("{moe}.experts.gate_up_proj"),
        layer_idx,
        vec![config.num_experts, 2 * config.moe_intermediate_size, hidden],
        ROUTED_GATE_UP_DTYPE,
        ShardPolicy::ExpertTensorSharded {
            n_experts: config.num_experts,
            inner: Box::new(ShardPolicy::ColumnShard { axis: 1 }),
        },
        quant_gate_up,
    ));
    weights.push(layer(
        &format!("{moe}.gate.weight"),
        layer_idx,
        vec![config.num_experts, hidden],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{moe}.shared_expert.down_proj.weight"),
        layer_idx,
        vec![hidden, config.shared_expert_intermediate_size],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
    weights.push(layer(
        &format!("{moe}.shared_expert.gate_proj.weight"),
        layer_idx,
        vec![config.shared_expert_intermediate_size, hidden],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(layer(
        &format!("{moe}.shared_expert.up_proj.weight"),
        layer_idx,
        vec![config.shared_expert_intermediate_size, hidden],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(layer(
        &format!("{moe}.shared_expert_gate.weight"),
        layer_idx,
        vec![1, hidden],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
}

fn push_ple_entries<F>(
    weights: &mut Vec<WeightEntry>,
    layer: &F,
    prefix: &str,
    layer_idx: usize,
    hidden: usize,
    ple_channels: usize,
    config: &Qwen4Config,
    bf16: &DTypeConstraint,
) where
    F: Fn(&str, usize, Vec<usize>, DType, ShardPolicy, &DTypeConstraint) -> WeightEntry,
{
    let ple = format!("{prefix}.ple");
    weights.push(layer(
        &format!("{ple}.conv1d.weight"),
        layer_idx,
        vec![ple_channels, 1, config.ple_conv_kernel_size],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.key_proj.weight"),
        layer_idx,
        vec![ple_channels, hidden],
        DType::BF16,
        ShardPolicy::ColumnShard { axis: 0 },
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.norm_conv.weight"),
        layer_idx,
        vec![ple_channels],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.norm_key.weight"),
        layer_idx,
        vec![ple_channels],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.norm_query.weight"),
        layer_idx,
        vec![ple_channels],
        DType::BF16,
        ShardPolicy::Replicate,
        bf16,
    ));
    weights.push(layer(
        &format!("{ple}.value_proj.weight"),
        layer_idx,
        vec![hidden, hidden],
        DType::BF16,
        ShardPolicy::RowShard { axis: 1 },
        bf16,
    ));
    for shard in 0..PLE_SHARD_COUNT {
        let name = format!("{ple}.ple_embedding.ngram_embedding.shard_{shard}.weight");
        weights.push(
            layer(
                &name,
                layer_idx,
                vec![PLE_SHARD_ROWS, PLE_ROW_WIDTH],
                DType::BF16,
                ShardPolicy::Replicate,
                bf16,
            )
            .external_rows(PLE_ROW_WIDTH * 2, ple_valid_rows_for_shard(shard)),
        );
    }
}

/// GPU placement requested by the typed Qwen4 assembler.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Qwen4Placement {
    pub name: String,
    pub layer: Option<usize>,
    pub device: usize,
}

#[derive(Clone, Debug)]
pub struct HyperConnectionWeights {
    pub block_inject: TensorRef,
    pub hc_norm: TensorRef,
    pub input_mix_down: TensorRef,
    pub input_mix_up: TensorRef,
}

#[derive(Clone, Debug)]
pub struct GdnWeights {
    pub a_log: TensorRef,
    pub dt_bias: TensorRef,
    pub in_proj_a: TensorRef,
    pub in_proj_b: TensorRef,
    pub qkv: TensorRef,
    pub z: TensorRef,
    pub norm: TensorRef,
    pub conv: TensorRef,
    pub output: TensorRef,
}

#[derive(Clone, Debug)]
pub struct QsaWeights {
    pub indexer_qk: TensorRef,
    pub indexer_k_norm: TensorRef,
    pub indexer_q_norm: TensorRef,
    pub k_norm: TensorRef,
    pub k: TensorRef,
    pub output: TensorRef,
    pub q_norm: TensorRef,
    pub q: TensorRef,
    pub v: TensorRef,
}

#[derive(Clone, Debug)]
pub struct PleWeights {
    pub conv: TensorRef,
    pub key: TensorRef,
    pub norm_conv: TensorRef,
    pub norm_key: TensorRef,
    pub norm_query: TensorRef,
    pub value: TensorRef,
}

#[derive(Clone, Debug)]
pub struct MoeWeights {
    pub experts_down: TensorRef,
    pub experts_gate_up: TensorRef,
    pub gate: TensorRef,
    pub shared_down: TensorRef,
    pub shared_gate: TensorRef,
    pub shared_up: TensorRef,
    pub shared_gate_scalar: TensorRef,
}

#[derive(Clone, Debug)]
pub struct Qwen4LayerWeights {
    pub layer: usize,
    pub kind: LayerType,
    pub attn_hyper: HyperConnectionWeights,
    pub mlp_hyper: HyperConnectionWeights,
    pub attention: Option<QsaWeights>,
    pub gdn: Option<GdnWeights>,
    pub ple: Option<PleWeights>,
    pub moe: MoeWeights,
}

#[derive(Clone, Debug)]
pub struct Qwen4MtpWeights {
    pub fc_embedding: TensorRef,
    pub fc_hidden: TensorRef,
    pub pre_fc_norm_embedding: TensorRef,
    pub pre_fc_norm_hidden: TensorRef,
    pub attn_hyper: HyperConnectionWeights,
    pub mlp_hyper: HyperConnectionWeights,
    pub attention: QsaWeights,
    pub moe: MoeWeights,
    pub final_hyper: HyperConnectionWeights,
}

/// Published resident ownership.  External PLE descriptors remain in the
/// attached transaction census; no fake handle is stored here.
pub struct Qwen4Weights {
    pub manifest: Qwen4Manifest,
    pub taken: Vec<TakenWeight>,
    pub layer_refs: Vec<Qwen4LayerWeights>,
    pub mtp: Qwen4MtpWeights,
}

impl Qwen4Weights {
    pub fn assemble(
        tx: &mut WeightLoadTransaction,
        config: &Qwen4Config,
        placements: &[Qwen4Placement],
    ) -> Result<Self, WeightError> {
        let manifest = Qwen4Manifest::build(config)?;
        let expected = manifest
            .resident_keys()
            .map(|(name, layer)| (name.to_string(), layer))
            .collect::<BTreeSet<_>>();
        let provided = placements
            .iter()
            .map(|placement| (placement.name.clone(), placement.layer))
            .collect::<BTreeSet<_>>();
        if expected != provided {
            return Err(WeightError::PlacementSetMismatch {
                expected: expected.len(),
                actual: provided.len(),
            });
        }
        let mut assembly = tx.begin_assembly();
        for placement in placements {
            if assembly
                .take(&placement.name, placement.layer, placement.device)
                .is_none()
            {
                return Err(WeightError::MissingPlacement {
                    name: placement.name.clone(),
                    layer: placement.layer,
                    device: placement.device,
                });
            }
        }
        let taken = assembly.commit().finalize();
        let layer_refs = build_layer_refs(config)?;
        let mtp = build_mtp_refs(config)?;
        Ok(Self {
            manifest,
            taken,
            layer_refs,
            mtp,
        })
    }

    pub fn external_descriptor<'a>(
        tx: &'a WeightLoadTransaction,
        placement: &Qwen4Placement,
    ) -> Option<&'a SourceRangeDescriptor> {
        tx.external_descriptor(&placement.name, placement.layer, placement.device)
    }

    /// Explicit GPU teardown for handles finalized from the canonical
    /// transaction.  The attached transaction census is drained separately,
    /// after readers and these resident handles are gone.
    pub fn free_gpu(self, gpu: &mut Gpu) -> hip_bridge::HipResult<()> {
        let mut first = None;
        for taken in self.taken {
            if let WeightHandle::Resident(tensor) = taken.handle {
                if let Err(error) = gpu.free_tensor(tensor) {
                    if first.is_none() {
                        first = Some(error);
                    }
                }
            }
        }
        first.map_or(Ok(()), Err)
    }
}

fn build_hyper_refs(
    prefix: &str,
    layer: Option<usize>,
    hc_wide: usize,
    hc_rank: usize,
) -> Result<HyperConnectionWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(HyperConnectionWeights {
        block_inject: tr(
            "block_inject_weight.weight",
            TensorRole::HyperConnectionBlockInject,
            vec![4, hc_wide],
        )?,
        hc_norm: tr(
            "hc_norm.weight",
            TensorRole::HyperConnectionNorm,
            vec![hc_wide],
        )?,
        input_mix_down: tr(
            "input_mix_weight_down.weight",
            TensorRole::HyperConnectionDown,
            vec![hc_rank, hc_wide],
        )?,
        input_mix_up: tr(
            "input_mix_weight_up.weight",
            TensorRole::HyperConnectionUp,
            vec![hc_wide, hc_rank],
        )?,
    })
}

fn build_layer_refs(config: &Qwen4Config) -> Result<Vec<Qwen4LayerWeights>, WeightError> {
    let hidden = config.hidden_size;
    let hc_wide = config.hc_count * hidden;
    let hc_rank = config.hc_lowrank;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let indexer_qk_dim =
        (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
    let gdn_qk = config.linear_num_key_heads * config.linear_key_head_dim;
    let gdn_v = config.linear_num_value_heads * config.linear_value_head_dim;
    let gdn_qkv = 2 * gdn_qk + gdn_v;
    let ple_channels = config.ple_embed_dim * config.hc_count;
    let mut refs = Vec::with_capacity(config.num_hidden_layers);
    for (layer, kind) in config.layer_types.iter().copied().enumerate() {
        let prefix = format!("model.language_model.layers.{layer}");
        let attn_hyper = build_hyper_refs(
            &format!("{prefix}.attn_hyper_connection"),
            Some(layer),
            hc_wide,
            hc_rank,
        )?;
        let mlp_hyper = build_hyper_refs(
            &format!("{prefix}.mlp_hyper_connection"),
            Some(layer),
            hc_wide,
            hc_rank,
        )?;
        let attention = if kind == LayerType::FullAttention {
            Some(build_qsa_refs(
                &format!("{prefix}.self_attn"),
                Some(layer),
                hidden,
                q_dim,
                kv_dim,
                indexer_qk_dim,
                config,
            )?)
        } else {
            None
        };
        let gdn = if kind == LayerType::LinearAttention {
            Some(build_gdn_refs(
                &format!("{prefix}.linear_attn"),
                Some(layer),
                hidden,
                gdn_qkv,
                gdn_v,
                config,
            )?)
        } else {
            None
        };
        let ple = if layer == 1 {
            Some(build_ple_refs(
                &format!("{prefix}.ple"),
                Some(layer),
                hidden,
                ple_channels,
                config,
            )?)
        } else {
            None
        };
        let moe = build_moe_refs(&format!("{prefix}.mlp"), Some(layer), hidden, config)?;
        refs.push(Qwen4LayerWeights {
            layer,
            kind,
            attn_hyper,
            mlp_hyper,
            attention,
            gdn,
            ple,
            moe,
        });
    }
    Ok(refs)
}

fn build_mtp_refs(config: &Qwen4Config) -> Result<Qwen4MtpWeights, WeightError> {
    let hidden = config.hidden_size;
    let hc_wide = config.hc_count * hidden;
    let hc_rank = config.hc_lowrank;
    let q_dim = config.num_attention_heads * config.head_dim;
    let kv_dim = config.num_key_value_heads * config.head_dim;
    let indexer_qk_dim =
        (config.indexer_n_heads + config.indexer_kv_heads) * config.indexer_head_dim;
    Ok(Qwen4MtpWeights {
        fc_embedding: TensorRef::new(
            "mtp.fc_embedding.weight",
            TensorRole::MtpEmbeddingProjection,
            None,
            vec![hidden, hidden],
            DType::BF16,
        )?,
        fc_hidden: TensorRef::new(
            "mtp.fc_hidden.weight",
            TensorRole::MtpHiddenProjection,
            None,
            vec![hidden, hidden],
            DType::BF16,
        )?,
        pre_fc_norm_embedding: TensorRef::new(
            "mtp.pre_fc_norm_embedding.weight",
            TensorRole::MtpNorm,
            None,
            vec![hidden],
            DType::BF16,
        )?,
        pre_fc_norm_hidden: TensorRef::new(
            "mtp.pre_fc_norm_hidden.weight",
            TensorRole::MtpNorm,
            None,
            vec![hc_wide],
            DType::BF16,
        )?,
        attn_hyper: build_hyper_refs("mtp.layers.0.attn_hyper_connection", None, hc_wide, hc_rank)?,
        mlp_hyper: build_hyper_refs("mtp.layers.0.mlp_hyper_connection", None, hc_wide, hc_rank)?,
        attention: build_qsa_refs(
            "mtp.layers.0.self_attn",
            None,
            hidden,
            q_dim,
            kv_dim,
            indexer_qk_dim,
            config,
        )?,
        moe: build_moe_refs("mtp.layers.0.mlp", None, hidden, config)?,
        final_hyper: build_hyper_refs("mtp.hyper_connection_mixer", None, hc_wide, hc_rank)?,
    })
}

fn build_gdn_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    qkv: usize,
    value: usize,
    config: &Qwen4Config,
) -> Result<GdnWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(GdnWeights {
        a_log: tr(
            "A_log",
            TensorRole::GdnALog,
            vec![config.linear_num_value_heads],
        )?,
        dt_bias: tr(
            "dt_bias",
            TensorRole::GdnDtBias,
            vec![config.linear_num_value_heads],
        )?,
        in_proj_a: tr(
            "in_proj_a.weight",
            TensorRole::GdnA,
            vec![config.linear_num_value_heads, hidden],
        )?,
        in_proj_b: tr(
            "in_proj_b.weight",
            TensorRole::GdnB,
            vec![config.linear_num_value_heads, hidden],
        )?,
        qkv: tr("in_proj_qkv.weight", TensorRole::GdnQkv, vec![qkv, hidden])?,
        z: tr("in_proj_z.weight", TensorRole::GdnZ, vec![value, hidden])?,
        norm: tr(
            "norm.weight",
            TensorRole::GdnNorm,
            vec![config.linear_value_head_dim],
        )?,
        conv: tr(
            "conv1d.weight",
            TensorRole::GdnConv,
            vec![qkv, 1, config.linear_conv_kernel_dim],
        )?,
        output: tr(
            "out_proj.weight",
            TensorRole::GdnOutput,
            vec![hidden, value],
        )?,
    })
}

fn build_qsa_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    indexer_qk_dim: usize,
    config: &Qwen4Config,
) -> Result<QsaWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(QsaWeights {
        indexer_qk: tr(
            "indexer.index_qk_proj.weight",
            TensorRole::QsaIndexerQk,
            vec![indexer_qk_dim, hidden],
        )?,
        indexer_k_norm: tr(
            "indexer.k_layernorm.weight",
            TensorRole::QsaIndexerKNorm,
            vec![config.indexer_head_dim],
        )?,
        indexer_q_norm: tr(
            "indexer.q_layernorm.weight",
            TensorRole::QsaIndexerQNorm,
            vec![config.indexer_head_dim],
        )?,
        k_norm: tr("k_norm.weight", TensorRole::QsaKNorm, vec![config.head_dim])?,
        k: tr("k_proj.weight", TensorRole::QsaK, vec![kv_dim, hidden])?,
        output: tr("o_proj.weight", TensorRole::QsaOutput, vec![hidden, q_dim])?,
        q_norm: tr("q_norm.weight", TensorRole::QsaQNorm, vec![config.head_dim])?,
        q: tr("q_proj.weight", TensorRole::QsaQ, vec![2 * q_dim, hidden])?,
        v: tr("v_proj.weight", TensorRole::QsaV, vec![kv_dim, hidden])?,
    })
}

fn build_ple_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    channels: usize,
    config: &Qwen4Config,
) -> Result<PleWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>| {
        TensorRef::new(
            format!("{prefix}.{suffix}"),
            role,
            layer,
            shape,
            DType::BF16,
        )
    };
    Ok(PleWeights {
        conv: tr(
            "conv1d.weight",
            TensorRole::PleConv,
            vec![channels, 1, config.ple_conv_kernel_size],
        )?,
        key: tr(
            "key_proj.weight",
            TensorRole::PleKey,
            vec![channels, hidden],
        )?,
        norm_conv: tr("norm_conv.weight", TensorRole::PleNorm, vec![channels])?,
        norm_key: tr("norm_key.weight", TensorRole::PleNorm, vec![channels])?,
        norm_query: tr("norm_query.weight", TensorRole::PleNorm, vec![channels])?,
        value: tr(
            "value_proj.weight",
            TensorRole::PleValue,
            vec![hidden, hidden],
        )?,
    })
}

fn build_moe_refs(
    prefix: &str,
    layer: Option<usize>,
    hidden: usize,
    config: &Qwen4Config,
) -> Result<MoeWeights, WeightError> {
    let tr = |suffix: &str, role: TensorRole, shape: Vec<usize>, dtype: DType| {
        TensorRef::new(format!("{prefix}.{suffix}"), role, layer, shape, dtype)
    };
    Ok(MoeWeights {
        experts_down: tr(
            "experts.down_proj",
            TensorRole::RoutedDown,
            vec![config.num_experts, hidden, config.moe_intermediate_size],
            ROUTED_DOWN_DTYPE,
        )?,
        experts_gate_up: tr(
            "experts.gate_up_proj",
            TensorRole::RoutedGateUp,
            vec![config.num_experts, 2 * config.moe_intermediate_size, hidden],
            ROUTED_GATE_UP_DTYPE,
        )?,
        gate: tr(
            "gate.weight",
            TensorRole::Router,
            vec![config.num_experts, hidden],
            DType::BF16,
        )?,
        shared_down: tr(
            "shared_expert.down_proj.weight",
            TensorRole::SharedExpertDown,
            vec![hidden, config.shared_expert_intermediate_size],
            DType::BF16,
        )?,
        shared_gate: tr(
            "shared_expert.gate_proj.weight",
            TensorRole::SharedExpertGate,
            vec![config.shared_expert_intermediate_size, hidden],
            DType::BF16,
        )?,
        shared_up: tr(
            "shared_expert.up_proj.weight",
            TensorRole::SharedExpertUp,
            vec![config.shared_expert_intermediate_size, hidden],
            DType::BF16,
        )?,
        shared_gate_scalar: tr(
            "shared_expert_gate.weight",
            TensorRole::SharedExpertGateScalar,
            vec![1, hidden],
            DType::BF16,
        )?,
    })
}

pub fn ple_valid_rows_for_shard(shard: usize) -> usize {
    crate::ple::PLE_VALID_ROWS
        .saturating_sub((shard as u64).saturating_mul(PLE_SHARD_ROWS as u64))
        .min(PLE_SHARD_ROWS as u64) as usize
}

pub const PLE_HEAD_COUNT: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightError {
    Config(String),
    InvalidShape {
        name: String,
        shape: Vec<usize>,
    },
    ShapeOverflow(String),
    DescriptorMismatch(String),
    RangeLength {
        name: String,
        expected: u64,
        actual: u64,
    },
    InvalidValidRows {
        name: String,
        valid_rows: usize,
        physical_rows: usize,
    },
    PlacementSetMismatch {
        expected: usize,
        actual: usize,
    },
    MissingPlacement {
        name: String,
        layer: Option<usize>,
        device: usize,
    },
    MissingExternalDescriptor {
        name: String,
        layer: Option<usize>,
        device: usize,
    },
    PleShardCount {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "Qwen4 manifest config: {message}"),
            Self::InvalidShape { name, shape } => write!(f, "{name}: invalid shape {shape:?}"),
            Self::ShapeOverflow(name) => write!(f, "{name}: shape overflow"),
            Self::DescriptorMismatch(name) => {
                write!(f, "{name}: external descriptor mismatch")
            }
            Self::RangeLength {
                name,
                expected,
                actual,
            } => write!(f, "{name}: range length expected {expected}, got {actual}"),
            Self::InvalidValidRows {
                name,
                valid_rows,
                physical_rows,
            } => write!(f, "{name}: valid_rows={valid_rows} outside {physical_rows}"),
            Self::PlacementSetMismatch { expected, actual } => {
                write!(f, "Qwen4 placement set expected {expected}, got {actual}")
            }
            Self::MissingPlacement {
                name,
                layer,
                device,
            } => write!(
                f,
                "missing resident placement {name}[layer {layer:?}] on device {device}"
            ),
            Self::MissingExternalDescriptor {
                name,
                layer,
                device,
            } => write!(
                f,
                "missing external descriptor {name}[layer {layer:?}] on device {device}"
            ),
            Self::PleShardCount { expected, actual } => {
                write!(f, "Qwen4 PLE shard count expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for WeightError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ple_shards_exclude_padding() {
        let total: usize = (0..PLE_SHARD_COUNT).map(ple_valid_rows_for_shard).sum();
        assert_eq!(total, 320_001_446);
        assert_eq!(ple_valid_rows_for_shard(127), 2_499_922);
    }
    fn pinned_config() -> Qwen4Config {
        Qwen4Config {
            architecture_id: 16,
            model_type: "qwen4_exp".into(),
            text_model_type: "qwen4_exp_text".into(),
            dtype: SourceDType::BF16,
            hidden_size: 2560,
            vocab_size: 248320,
            num_hidden_layers: 48,
            max_position_embeddings: 262144,
            num_attention_heads: 24,
            num_key_value_heads: 2,
            head_dim: 256,
            partial_rotary_factor: 0.25,
            rope_theta: 10_000_000.0,
            attention_bias: false,
            layer_types: (0..48)
                .map(|layer| {
                    if layer % 4 == 3 {
                        LayerType::FullAttention
                    } else {
                        LayerType::LinearAttention
                    }
                })
                .collect(),
            full_attention_interval: 4,
            linear_num_key_heads: 16,
            linear_num_value_heads: 48,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            recurrent_state_dtype: RecurrentStateDType::F32,
            indexer_n_heads: 4,
            indexer_kv_heads: 1,
            indexer_head_dim: 128,
            indexer_budget: 2048,
            indexer_compress_ratio: 4,
            num_experts: 512,
            num_experts_per_tok: 10,
            moe_intermediate_size: 640,
            shared_expert_intermediate_size: 640,
            norm_topk_prob: true,
            output_gate_type: "sigmoid".into(),
            hc_count: 4,
            hc_lowrank: 320,
            ple_layer_ids: vec![2],
            ple_conv_kernel_size: 4,
            ple_embed_dim: 2560,
            split_ngram_parts: 128,
            heads_per_ngram: 8,
            ngram_size: 3,
            ngram_vocab_size_base: 20_000_000,
            make_ngram_vocab_size_divisible_by: 128,
            eos_token_id: 248044,
            tie_word_embeddings: false,
            mtp_num_hidden_layers: 1,
            mtp_use_dedicated_embeddings: false,
            mtp: Qwen4MtpConfig {
                num_hidden_layers: 1,
                layer_types: vec![LayerType::FullAttention],
                hybrid: true,
                rope_theta: 10_000_000.0,
            },
        }
    }

    #[test]
    fn manifest_matches_pinned_text_inventory_and_rejects_legacy_names() {
        let manifest = Qwen4Manifest::build(&pinned_config()).expect("pinned config manifest");
        // The source has 1,658 records, of which 333 are vision records.  The
        // text manifest accounts for all 1,325 remaining records; its one MTP
        // embedding alias is excluded by source_tensor_count and its three
        // exact I64 PLE arrays are included through `metadata`.
        assert_eq!(manifest.source_tensor_count(), 1325);
        assert_eq!(manifest.source_tensor_count() + 333, 1658);
        let expect = |name: &str, layer: Option<usize>, shape: &[usize]| {
            let entry = manifest
                .entry(name, layer)
                .unwrap_or_else(|| panic!("missing exact inventory tensor {name} layer {layer:?}"));
            assert_eq!(entry.logical_shape, shape, "{name} shape");
        };
        expect(
            "model.language_model.layers.0.linear_attn.A_log",
            Some(0),
            &[48],
        );
        expect(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            Some(0),
            &[10240, 2560],
        );
        expect(
            "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
            Some(3),
            &[640, 2560],
        );
        expect(
            "model.language_model.layers.3.self_attn.q_proj.weight",
            Some(3),
            &[12288, 2560],
        );
        expect(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
            Some(1),
            &[PLE_SHARD_ROWS, PLE_ROW_WIDTH],
        );
        expect("mtp.fc_hidden.weight", None, &[2560, 2560]);
        expect("mtp.layers.0.self_attn.q_proj.weight", None, &[12288, 2560]);
        expect(
            "model.language_model.hyper_connection_mixer.hc_norm.weight",
            None,
            &[10240],
        );
        assert!(manifest
            .entry("model.layers.0.linear_attn.in_proj_qkv.weight", Some(0))
            .is_none());
        assert!(manifest
            .entry(
                "model.language_model.layers.0.self_attn.qkv_proj.weight",
                Some(0)
            )
            .is_none());
        assert!(manifest
            .entry("model.language_model.layers.0.mlp.router.weight", Some(0))
            .is_none());
        assert!(manifest.metadata.iter().any(|tensor| {
            tensor.name == "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"
                && tensor.source_dtype == "I64"
                && tensor.shape == [3]
        }));
        assert!(matches!(
            manifest
                .entry(
                    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight",
                    Some(1)
                )
                .expect("PLE shard"),
            WeightEntry {
                residency: WeightResidency::ExternalRows { .. },
                ..
            }
        ));
    }
}
