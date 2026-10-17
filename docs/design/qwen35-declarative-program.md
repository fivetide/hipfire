# Qwen3.5 declarative layer program (#666 G8)

Status: slice 1 (single-GPU decode layers) is branch-implemented
(HIPF-117). It recuts #795 from current `beta`. #795 is the design
reference only, and none of its bodies are cherry-picked.

## Goal

Qwen3.5 (and 3.6/3.8, same architecture) declares each decoder layer as
typed `Step`s, as Gemma 4 and Qwen4 do. `hipfire-dispatch` owns the executor
bodies and every arch-, shape- and dtype-gated route choice. The architecture
crate keeps:

- weight, recurrent-state and KV ownership;
- the per-layer program shape (`src/qwen35/program.rs`): which op, bound to
  which tensors;
- the lifecycle: hipGraph, Redline, DFlash capture and the chunk loop.

## Slice 1: decode layers

Single-GPU decode (`forward_scratch_layers_lowered`) runs each layer as

    [DeltaNetMixer | GatedAttention, SwigluFfn | Moe]

The new ops live in `hipfire-dispatch/src/pipeline/hybrid.rs`:

| Step | Stages | Route choices owned by `hybrid` |
|---|---|---|
| `DeltaNetMixer` | `project` → `prepare` → `recur` → `gated_norm` → `output` | QKVZA scalar prep, conv + QK norm, conv scalar prep, compact QK (2:1, 3:1), FP32/Q8/Q8-compact/Q4 state, fused gated norm + MQ rotate |
| `GatedAttention` | `project` → `attend` → `output` | fused FA prep, Q8 KV-write fold, TriAttention tap, compaction offset, fused gate + MQ-rotate epilogue |
| `SwigluFfn` | `project` → `down` | MQ SiLU·rotate + SwiGLU-residual GEMV, HFQ residual GEMV, plain GEMV + add for the rotated formats (MFP4, MQ8, MQ4G128, ParoQ4G128, …), `memory.offload_exec=cpu` split |

MoE layers keep `moe_ffn_dispatch` (the sealed MoE program with its CPU
expert splice).

The executor bodies were moved from current beta's `Qwen35Bindings` super-op
handlers. Every gate exists once:

- `hybrid::{gdn_compact_qk_div, gated_norm_mq_rotate, fused_attention_prep,
  fused_attention_epilogue, qkvza_scalar_prep, conv_scalar_prep,
  conv_qknorm, gfx1151_radiowave, gfx1201_state_fusions}` read
  `HybridDims` (built by `program::hybrid_dims`).
- The qwen35 copies are deleted. `kv_cache_attention_dispatch` (hand arms,
  dense TP, the batched-prefill FA fallback) calls `hybrid::kv_attention`.
  `forward_slots/vmm/exact.rs` calls the hybrid predicates. The dense 27B
  shape predicate is `HybridDims::is_dense_27b_shape` (the widened prefill
  admission uses it too).
- `qkv/qkvza/gate_up_via_execute_steps` forward to `hybrid::normed_projection`.

`lower_variant` and `Qwen35Bindings` stay for the EP decode executor and the
exact VMM slot route. Their handlers delegate to the same hybrid stages, so
each body exists once.

Left for later slices: the hand arms (`HIPFIRE_FORWARD_LOWERED=0`, DFlash
hidden-state capture, VL mrope, `memory.offload_exec=cpu` spills that own
host CPU weights), batched prefill/verify executors, and the MTP head.
`hipfire_runtime::llama::weight_gemv_swiglu_residual` still serves those
hand arms and the other architectures. `hybrid::swiglu_down_residual` is
its dispatch twin: its rotated-format arm runs `GemvFamily::run_auto`
where the runtime runs `weight_gemv`.

## Gates

Each slice byte-compares `mq4_prefill_state_oracle` dumps (prefill logits,
Q8 KV, GDN state + EF, conv state, every decode step's logits) against its
beta base, on `qwen3.5-9b.mq4`, `qwen3.8-27b.mq4-xt` and
`qwen3.6-35b-a3b.mq4r`, with at least two chunk boundaries plus an odd tail.
It also needs greedy serve A/B rows for the routes it touches, and Redline
`--kv-mode q8 --pm4` shadow parity with an unchanged decode tape hash.
Evidence: HIPF-117.
