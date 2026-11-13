## Default-ON env feature flags in crates/hipfire-config (sweep C3)

Scope: `crates/hipfire-config/` at upstream beta 39f5fd421. Read-only. Line numbers are in `src/lib.rs` unless noted. Schema rows (`process_bool_field!`, `field!`, `diagnostic_bool_field!`, `process_auto_bool_field!`) list the schema macro line as the read site; the actual read happens in the consumer named in the notes column. "OFF-path size" is not measured per flag; it is marked `branch-site` (one or a few call sites) or `unmeasured`.

The `introduced` column comes from `git log -S'<ENV>' --oneline | tail -1`. Cells marked `?` had not finished when this file was written (the lookup was still running in the background), so they are unknown, not absent.

| env var | kind | file:line of read | default-ON evidence | what OFF does | OFF-path size | obsolete signal | introduced |
|---|---|---|---|---|---|---|---|
| HIPFIRE_SERVE_BATCH_SPEC | bool | lib.rs:1392 | `process_bool_field!(..., Bool(true), ...)` | Disables cross-request speculative verify rows on the VMM batch route (singleton spec unchanged) | branch-site | NONE | 90bbee027 |
| HIPFIRE_SERVE_BATCH_SAMPLED | bool | lib.rs:1410 | `Bool(true)`; help "Default on" | Sampled requests are not admitted to the VMM batch route | branch-site | NONE | 251a0c7ff |
| HIPFIRE_NORMALIZE_PROMPT | bool | lib.rs:1739 | `DefaultValue::Bool(true)` | No newline-run collapse before tokenization | branch-site | NONE | cef7caf05 |
| HIPFIRE_MTP_SAMPLED | bool | lib.rs:1946 (schema), lib.rs:5152 `mtp_sampled_enabled()` | `DefaultValue::Bool(true)`; `developer_bool(..., true)` | Sampled (temp>0) requests go to AR; no MTP rejection sampling. Consumers: loader carriers.rs:1335, qwen4 mtp_spec.rs:5481, 12726 | branch-site | NONE | 2988baa90 |
| HIPFIRE_DEFAULT_CHATML | bool | lib.rs:2054 | `DefaultValue::Bool(true)` | No fallback ChatML frame when no template resolves | branch-site | NONE | f0482fd3a |
| HIPFIRE_JINJA_CHAT | bool | lib.rs:2066 | `Bool(true)`; help says `=0` selects hand-rolled ChatScaffold | Chat prompts use the hand-rolled ChatScaffold instead of the model Jinja template | branch-site | "hand-rolled" path is a fallback | d937c7e38 |
| HIPFIRE_TP_USE_RCCL | bool | lib.rs:2345 | `Bool(true)` | Tensor-parallel all-reduce without RCCL | branch-site | NONE | cf6ad952a |
| HIPFIRE_PREFILL_BATCHED | bool | lib.rs:2354 | `Bool(true)` | Non-batched prefill kernels | branch-site | NONE | eb629ce62 |
| HIPFIRE_DRAFT_F16 | bool | lib.rs:2373 | `Bool(true)` | DFlash draft activations in F32 | branch-site | NONE | ec786d050 |
| HIPFIRE_MQ4V2_GATEUP_K5120 | bool | lib.rs:2531 | `Bool(true)`; help "default on" | Generic gate/up kernel instead of K5120 fixed-group kernel (gfx1100) | branch-site | help calls the generic kernel the restore target | 288e5f767 |
| HIPFIRE_GFX12_MQ4V2_FP8_GATEUP | bool | lib.rs:2549 | `Bool(true)`; FeatureFlags default `arch=="gfx1201"` (rdna-compute) | gfx1201 FP8-WMMA gate/up prefill route off | branch-site | NONE | decb2f62a |
| HIPFIRE_GFX12_MQ4V2_FP8_RESID | bool | lib.rs:2558 | `Bool(true)`; FeatureFlags `arch=="gfx1201"` | FP8-WMMA residual prefill off | branch-site | NONE | decb2f62a |
| HIPFIRE_GFX12_MQ4V2_FP8_QKVZA | bool | lib.rs:2567 | `Bool(true)`; FeatureFlags `arch=="gfx1201"` | FP8-WMMA 4-way QKVZA prefill off | branch-site | NONE | decb2f62a |
| HIPFIRE_GFX12_MQ4V2_FP8_QKV | bool | lib.rs:2576 | `Bool(true)`; FeatureFlags `arch=="gfx1201"` | FP8-WMMA 3-way QKV prefill off | branch-site | NONE | decb2f62a |
| HIPFIRE_GFX12_MQ4V2_FP8_V2 | bool | lib.rs:2585 | `Bool(true)` | Staged-tile v2 FP8 prefill off (falls back to the non-v2 FP8 symbols) | branch-site | NONE | 567836f80 |
| HIPFIRE_GFX12_GDN_PRE_FUSED | bool | lib.rs:2594 | `Bool(true)` | Separate sigmoid+conv+qknorm launches instead of fused gdn_pre_batched_gfx1201 | branch-site | NONE | 7589008be |
| HIPFIRE_GFX12_GDN_CHUNK_SCAN | bool | lib.rs:2603 | `Bool(true)` | Ordinary (non-chunked) GDN prefill scan | branch-site | NONE | 0b4d3e48c |
| HIPFIRE_GFX12_SILU_QUANT_FUSED | bool | lib.rs:2612 | `Bool(true)` | Standalone quantize_int4_mmq_ds128 launch after SwiGLU | branch-site | NONE | 1401a2c53 |
| HIPFIRE_GFX12_PRODUCER_QUANT_FUSED | bool | lib.rs:2621 | `Bool(true)` | Standalone quantize launches at producer sites (IU4 and A8 twins) | branch-site | NONE | 2ebc00a00 |
| HIPFIRE_G12_NORM | bool | lib.rs:2630 | `Bool(true)`; help "restore the incumbent _gfx12 symbols" | Incumbent _gfx12 RMSNorm/gated-norm producers | branch-site | help says "incumbent" | ? |
| HIPFIRE_G12_DEC_NORM | bool | lib.rs:2639 | `Bool(true)` | Single-workgroup decode norm launches (gfx1201, gfx1151) | branch-site | NONE | cb6624c35 |
| HIPFIRE_GFX1100_DEC_NORM | bool | lib.rs:2648 | `Bool(true)` | Single-workgroup decode norm launches (gfx1100) | branch-site | NONE | 3d0098dd5 |
| HIPFIRE_G12_A4C2 | bool | lib.rs:2657 | `Bool(true)` | Round-to-nearest d=amax/7 instead of {5,7} two-scale search | branch-site | NONE | 44ca584ad |
| HIPFIRE_GFX11_PRODUCER_QUANT_FUSED | bool | lib.rs:2666 | `Bool(true)`; help "default on gfx1100/gfx1151" | Standalone int4 quantize launches at gfx11 producer sites | branch-site | NONE | ? |
| HIPFIRE_GFX11_LEAN_PBS | bool | lib.rs:2675 | `Bool(true)` | Full-size gfx11 MQ4V2 prefill scratch | branch-site | NONE | ? |
| HIPFIRE_GFX12_FP8_STREAM | bool | lib.rs:2694 | `Bool(true)` | Separate pack_f32_to_fp8_mq4v2_gfx12 launch | branch-site | NONE | ? |
| HIPFIRE_GFX12_FA2_PREFILL | bool | lib.rs:2703 | `Bool(true)` | Non-GQA-fused FA2 prefill route on gfx1201 | branch-site | NONE | ? |
| HIPFIRE_GFX12_FA_PACKET | bool | lib.rs:2712 | `Bool(true)` | Route-N attention body instead of packet-minimal Q128 | branch-site | help says "route-N" | ? |
| HIPFIRE_ATTN_QRESIDENT | bool | lib.rs:2721 | `Bool(true)` | Non-Q-resident wide-workgroup FA2 prefill | branch-site | NONE | ? |
| HIPFIRE_ATTN_QRESIDENT_V2 | bool | lib.rs:2730 | `Bool(true)` | v1 Q-resident kernel | branch-site | help says "restore the v1" | ? |
| HIPFIRE_VERIFY_ATTN | bool | lib.rs:2739 | `Bool(true)` | Attention_flash/q8 batched tile paths for speculative verify | branch-site | NONE | ? |
| HIPFIRE_GFX1100_FA2_SPLIT_VERIFY | bool | lib.rs:2748 | `Bool(true)`; help "default on exact gfx1100" | Established VerifyAttn/R4-R8/batched route | branch-site | help says "established" | ? |
| HIPFIRE_GFX12_FA_PREP_FUSED | bool | lib.rs:2757 | `Bool(true)` | Separate Q/K norm and RoPE launches | branch-site | NONE | ? |
| HIPFIRE_GFX12_FA_PREP_FP8Q | bool | lib.rs:2766 | `Bool(true)` | F32 Q for Q-resident v2 attention | branch-site | NONE | ? |
| HIPFIRE_GFX11_FA2_PREFILL | bool | lib.rs:2783 | `Bool(true)` | Non-GQA-fused FA2 prefill on gfx1100/gfx1151 | branch-site | NONE | ? |
| HIPFIRE_IU4_PREFILL | bool | lib.rs:2792 | `Bool(true)` | Incumbent prefill MMQ route (non-W4A4 iu4-direct) | branch-site | NONE | ? |
| HIPFIRE_GFX11_IU4_GRIDSPEC | bool | lib.rs:2809 | `Bool(true)` | Guarded loads/stores on all IU4 N-tiles | branch-site | NONE | ? |
| HIPFIRE_GFX11_IU4_SHAPE | bool | lib.rs:2818 | `Bool(true)` | Prior gfx1100 IU4 shape | branch-site | help says "restore the prior" | ? |
| HIPFIRE_IU4_SYMFOLD | bool | lib.rs:2827 | `Bool(true)`; "default on for symmetric artifacts only" | Generic (non-signed-weight) IU4 WMMA | branch-site | NONE | ? |
| HIPFIRE_FP16 | bool | lib.rs:2962 | `Bool(true)` | FP16 kernel routes disallowed | branch-site | NONE | ? |
| HIPFIRE_LM_HEAD_WMMA | bool | lib.rs:2980 | `Bool(true)` | No WMMA LM-head dispatch | branch-site | NONE | ? |
| HIPFIRE_HFQ4G128_MMQ | bool | lib.rs:3005 | `Bool(true)` | No HFQ4G128 MMQ dispatch | branch-site | NONE | ? |
| HIPFIRE_DRAFT_SHARED_QKV_ROTATION | bool | lib.rs:3041 | `Bool(true)`; help "default on exact gfx1100" | Rotate Q/K/V noise input per projection | branch-site | help says "rotate once per projection" | ? |
| HIPFIRE_DRAFT_SHARED_FFN_ROTATION | bool | lib.rs:3050 | `Bool(true)`; "default on exact gfx1100" | Rotate FFN input per projection | branch-site | same | ? |
| HIPFIRE_DRAFT_SHARED_CTX_KV_ROTATION | bool | lib.rs:3059 | `Bool(true)`; "default on exact gfx1100" | Rotate context input per projection | branch-site | same | ? |
| HIPFIRE_MOE_GROUPED_GEMM | bool | lib.rs:3231 | `Bool(true)` | No grouped-GEMM MoE prefill | branch-site | NONE | ? |
| HIPFIRE_AR_GRAPH | bool | lib.rs:3262 | `Bool(true)`; help "default on, except Windows gfx1100" | No AR forward graph capture | branch-site | help cites bug #655 (wrong output) | ? |
| HIPFIRE_GRAPH_MOE | bool | lib.rs:3271 | `Bool(true)` | No graph capture on MoE forward | branch-site | NONE | ? |
| HIPFIRE_DEEPSEEK4_Q8_WMMA | bool | lib.rs:3340 | `Bool(true)` | No DeepSeek4 Q8 WMMA prefill | branch-site | NONE | ? |
| HIPFIRE_DEEPSEEK4_Q8_4W | bool | lib.rs:3349 | `Bool(true)` | No four-wave Q8 WMMA tile | branch-site | NONE | ? |
| HIPFIRE_DDTREE_TREE_LA | bool | lib.rs:3412 | `Bool(true)` | No linearized-ancestor tape replay | branch-site | NONE | ? |
| HIPFIRE_DFLASH_FAST_SAMPLE | bool | lib.rs:3421 | `Bool(true)` | No GPU sampling on sampled DFlash verify | branch-site | NONE | ? |
| HIPFIRE_DFLASH_Q8_LMHEAD_WMMA | bool | lib.rs:3430 | `Bool(true)` | No Q8 WMMA LM-head in DFlash verify | branch-site | NONE | ? |
| HIPFIRE_FUSE_QKV_BIAS | bool | lib.rs:3439 | `Bool(true)` | No bias folding into fused QKV decode | branch-site | NONE | ? |
| HIPFIRE_REPLAY_PM4_GCR_TRIM | bool (diagnostic) | lib.rs:3617 | `diagnostic_bool_field!(..., Bool(true), ...)` | No GCR trimming on retained PM4 routes | branch-site | NONE | ? |
| HIPFIRE_QWEN4_VMM_GFX1100 | bool | devices.rs:125 `qwen4_vmm_gfx1100_enabled()` | `crate::developer_bool(..., true)` | Qwen4 VMM not admitted on gfx1100 (consumers: hipfire-arch-qwen4 kv_backend.rs:70,290; rdna-compute dispatch.rs:5360) | branch-site | NONE | ? |
| HIPFIRE_QWEN4_LANE_RESERVE_TRIM | bool | lib.rs:4618 `qwen4_lane_reserve_trim()` | `developer_bool(..., true)` | Per-layer capture-ring allocation and old lane reservation (consumer: hipfire-arch-qwen4 state.rs:2165) | branch-site | doc says "0 restores ... byte for byte" | ? |
| HIPFIRE_QWEN4_LANE_APPEND_SHARE | bool | lib.rs:4633 `qwen4_lane_append_share()` | `developer_bool(..., true)` | Per-drafter MTP append scratch (consumers: loader carriers.rs:3232, 2-site; qwen4 mtp_spec.rs:5772) | branch-site | doc says "restores the per-drafter allocation" | ? |
| HIPFIRE_QWEN4_LANE_PAGE_BUDGET | bool | lib.rs:4647 `qwen4_lane_page_budget()` | `developer_bool(..., true)` | Admission by slots and length only (consumer: qwen4 lane/pages.rs:78) | branch-site | doc says "restores the previous admission" | ? |
| HIPFIRE_DFLASH_CKPT_RESUME | bool | lib.rs:5035 (`mtp_cache_policy().checkpoint_resume`) | `developer_bool(..., true)` | No checkpoint-resume ring (loader lib.rs:2308; qwen35 mtp_speculator.rs:470-492) | branch-site | NONE | ? |
| HIPFIRE_SPEC_WINDOW_ROLLBACK | bool | lib.rs:5046 (`mtp_cache_policy().window_rollback`) | `developer_bool(..., true)` | Window rollback skipped (qwen35 mtp_speculator.rs:742) | branch-site | NONE | ? |
| HIPFIRE_QWEN4_D2_MIXED_CPWAIT | bool (conditional default) | lib.rs:12001 `qwen4_d2_mixed_cpwait(arch)` | `developer_bool(..., qwen4_gfx1100_tier_package(arch))`: ON only on exact gfx1100 with tier package | Mixed hot/cold D2 layers use copy transport + host join (consumer qwen4 gpu_forward.rs:6648) | branch-site | NONE | ? |
| HIPFIRE_QWEN4_EXPERT_HOTSET_GFX1100 | kill switch (bool, gfx1100 package) | lib.rs:11815 `qwen4_gfx1100_tier_package()` | `gfx1100_tier_package_for`: `arch=="gfx1100" && switch.unwrap_or(true)` | Whole gfx1100 tier package off (consumers: qwen4 expert_hotset.rs:83,126; slowlink_admission.rs:146) | multi-member, unmeasured | doc says every member returns to previous default | ? |
| HIPFIRE_QWEN4_DIRECT_LOAD | kill switch (bool) | lib.rs:11374 `qwen4_direct_load()` | `!= Some("0")` | mmap + single-thread copy loader (consumers: loader carriers.rs:6003, 7102) | branch-site | NONE | ? |
| HIPFIRE_EP_GATHER_DIRECT | kill switch (bool) | lib.rs:11454 `ep_gather_direct()` | `!= Some("0")` | No direct gather for EP payloads (carriers.rs:9314) | branch-site | NONE | ? |
| HIPFIRE_EP_SHARE_HOST | kill switch (bool) | lib.rs:11466 `ep_share_host()` | `!= Some("0")` | Each rank maps its own pinned host copy (weight_store.rs:2750) | branch-site | NONE | ? |
| HIPFIRE_EP_MTP_EXPERTS_RESIDENT | kill switch (bool) | lib.rs:11481 `ep_mtp_experts_resident()` | `!= Some("0")` | MTP experts not kept resident (carriers.rs:957, 2559) | branch-site | NONE | ? |
| HIPFIRE_EP_HC8_PLAN_CHARGE | kill switch (bool) | lib.rs:11495 `ep_hc8_plan_charge()` | `!= Some("0")` | EP plan does not charge HC8 (carriers.rs:2537) | branch-site | NONE | ? |
| HIPFIRE_EP_RESERVE_CENSUS | kill switch (bool) | lib.rs:11512 `ep_reserve_census()` | `!= Some("0")` | No reserve census (carriers.rs:4227; qwen4 expert_residency.rs:4770; bundle.rs:1325) | branch-site | NONE | ? |
| HIPFIRE_QWEN4_MTP_HEAD_D2 | kill switch (typed bool) | lib.rs:11237 `qwen4_mtp_head_d2()` | `Option<bool>`; unset arms head D2 when trunk D2 is armed (qwen4 bundle.rs:2042, mtp_gpu.rs:3008) | MTP head keeps GPU expert route (HeadD2Off::KnobOff) | branch-site | NONE | ? |
| HIPFIRE_SPEC_LOOKUP_TAIL | kill switch (bool) | lib.rs:5246 `dflash_lookup_tail_enabled()` | `value != Some("0")` | DFlash lookup Tail windows off (consumer qwen35 dflash_spec.rs:3386) | branch-site | NONE | ? |
| HIPFIRE_NO_REGISTRY_FETCH | kill switch (bool) | read in hipfire-registry/src/lib.rs:1781 (config lib.rs:3750 only lists it in BOOTSTRAP_ENV) | `!= Some("1")` so fetch is on unless set to 1 | No registry network fetch (offline) | branch-site | NONE | ? |
| HIPFIRE_OOM_GUARD | mode (auto) | lib.rs:886 schema; lib.rs:4504 `oom_guard_effective()` | Default auto: on for unified-memory arches (gfx1035/1036/1103/115x), off for discrete; consumers carriers.rs (several), rdna-compute kv_slots.rs:345 | Memory preflight OOM guard off | branch-site | NONE | ? |
| HIPFIRE_DFLASH_LOOKUP | mode (auto) | lib.rs:1688 schema; lib.rs:5236 `dflash_lookup_enabled()` | Default auto; `!matches!(value, Some("0"\|"off"))` | No verbatim-copy lookup in DFlash2 verify (consumer qwen35 dflash_spec.rs:3385) | branch-site | NONE | ? |
| HIPFIRE_MTP_MODE | mode (auto) | lib.rs:1907 schema | Default auto arms native MTP for producers declaring a measured AR floor (native Qwen4) | Plain AR for all producers | unmeasured | NONE | ? |
| HIPFIRE_MTP_NGRAM | mode (auto) | lib.rs:1931 schema; lib.rs:5071 | Default auto arms on arch 16 gfx1151/gfx1201 | Off (`0`/`off`) | branch-site | NONE | ? |
| HIPFIRE_MTP_NGRAM_COPY | mode (auto) | lib.rs:1943 schema; lib.rs:5205 | Default auto arms on arch 16 gfx1151/gfx1201 | Off (`0`/`off` = kill switch) | branch-site | help says kill switch | ? |
| HIPFIRE_GRAPH | mode (auto) | lib.rs:3254 schema | Default auto follows architecture policy (Windows gfx1100 excluded) | Forward graph capture off | unmeasured | NONE | ? |
| HIPFIRE_GFX11_Q8_FA2_WIDE | mode (auto) | lib.rs:2775 schema | auto enables on exact gfx1100 and gfx1151; other arches off | Whole-chunk Q8 FA2 prefill off | unmeasured | NONE | ? |
| HIPFIRE_PM_DECODE | mode (auto) | lib.rs:2801 schema | auto enables on exact gfx1201 only | PeaceMaker twins not loaded; hipcc compile instead | unmeasured | NONE | ? |
| HIPFIRE_GEMMA4_BATCHED_EMBEDDING_PREFILL | mode (auto) | lib.rs:3307 schema | auto enables on validated gfx1100/gfx1201 | Per-token embedding lookups | unmeasured | NONE | ? |
| HIPFIRE_GEMMA4_PLE_BRANCH_BATCHED_PREFILL | mode (auto) | lib.rs:3324 schema | auto-enabled on validated gfx1100/gfx1201 | No PLE branch batching | unmeasured | NONE | ? |
| HIPFIRE_GEMMA4_PLE_ACTIVATION_FUSED_PREFILL | mode (auto) | lib.rs:3332 schema | auto enables on validated gfx1100/gfx1201 | Unfused PLE activation and strided multiply | unmeasured | NONE | ? |
| HIPFIRE_REPLAY_PM4_GFX11_VMEM_ACQUIRE | mode (auto) | lib.rs:3624 schema | auto selects gfx1151 | No Radiowave VMEM acquires on gfx11 | unmeasured | NONE | ? |
| HIPFIRE_QWEN4_NPU_TRUNK | mode (auto) | lib.rs:3690 schema | auto admits calibrated gfx1151 NPU prefill | GPU-only trunk projection (help: "off is the GPU-only kill switch") | unmeasured | help says kill switch | ? |
| HIPFIRE_GFX1201_PM4_PACING | mode (auto) | lib.rs:3632 schema | auto = 64-dword NOP after every dispatch | NOP pacing off (`off`/`0`) | unmeasured | NONE | ? |

## Notes

1. Row count: 86 rows. 55 are schema `Bool(true)` defaults (including `DefaultValue::Bool(true)` entries and one diagnostic bool). The other 31 are accessor-only or `auto`-mode defaults: 17 kill-switch or accessor rows and 14 mode rows.
2. `introduced` is `?` for every row except the 22 that finished in the background lookup (intro.tsv). The background lookup (xargs over `git log -S`, roughly 30 s per flag) was still running when this file was written, so the `?` cells are unknown and may be filled later. Re-run the pickaxe for those names if the table is used.
3. Arch-gated defaults: many schema `Bool(true)` rows say "default on exact gfx1201/gfx1100" in help text. Their effective default comes from `rdna-compute/src/feature_flags.rs` (`unwrap_or(arch == ...)`), which is outside this scope. Those rows are ON only on the named arch, not everywhere.
4. Not assessed: about 20 `process_auto_bool_field!` override keys with `auto` (for example `kernel.gemv_dp4a`, `kernel.hfq3_mmq`, `kernel.rdna3_*`, `kernel.gfx942_*`, `kernel.moe_grouped_i8`, `kernel.mmq`). Their help says "Override the architecture-selected X", so their ON/OFF state depends on arch selection in rdna-compute. Also not assessed: `speculation.mode` (HIPFIRE_SPECULATION), `memory.mmq.screen` (HIPFIRE_MMQ_SCREEN), `attention.flash` (HIPFIRE_ATTN_FLASH), `speculation.dflash_ngram_block` (HIPFIRE_DFLASH_NGRAM_BLOCK), and `kernel.lm_head_f16`. Their defaults are `auto` with no clear on/off meaning.
5. Excluded: `HIPFIRE_QWEN4_EP` (`ep_killed`). It is a kill switch for an opt-in feature (`topology.ep` defaults to 1), so it is not default ON.
6. Excluded: `HIPFIRE_QWEN4_D2_READY`, `_JOIN`, `_RETURN` and `_PCIE_SHARE`. Their defaults depend on the gfx1100 tier package and on the load, so they are conditional and not listed as separate rows.
7. Out of scope but relevant: `crates/rdna-compute/src/feature_flags.rs` has 47 `unwrap_or(true)` lines. They are the real consumer-side defaults for many of the schema rows above and should be swept in the rdna-compute scope.
8. Read-site caveat: schema rows (`process_bool_field!`) define the default, but the runtime read goes through the typed ProcessConfig and FeatureFlags, so the cited lib.rs line is the definition site, not the consumer.
9. Tests excluded: `#[cfg(test)]` blocks in lib.rs (starting at line 6858), intent.rs, effective.rs, tuning.rs and devices.rs were scanned only for the purpose of finding non-test reads.
