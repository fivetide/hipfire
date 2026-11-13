# Default-ON `HIPFIRE_*` feature-flag sweep (HIPF-396)

Snapshot: `beta` @ `39f5fd421`, swept 2026-11-13.

## What this is

During kernel and batching work, many `HIPFIRE_*` environment flags were added to gate a new code path. Most of those paths later became the default, but the flag and its OFF path were kept. This sweep lists every flag whose gated path runs when the variable is **unset**. We start by assuming each one is obsolete; this list is input for deciding which to remove. It is not a removal verdict.

## Method

- Nine small (Haiku) agents each swept one crate cluster (`rdna-compute/feature_flags.rs`, the rest of `rdna-compute`, `hipfire-config`, `qwen4`, the `qwen35` family, `hipfire-runtime`, `deepseek4` plus the MoE archs, `dispatch`/`generate`/`loader`, and everything else). They looked for `developer_bool(.., true)`, `!= "0"` / `!= Ok("0")`, `unwrap_or(true)`, schema `Bool(true)`, kill-switch names (`*_NO_*`, `*_DISABLE*`, `*_OFF`), and modes whose default picks the new path. Default-OFF flags, numeric tunables and test-only reads were skipped.
- The cluster outputs were merged and deduplicated by name. There are **561 unique flags** in 630 rows; 65 flags are read in two places (usually a schema `Bool(true)` plus a `FeatureFlags` read).
- "First seen" is the first commit that added the name to any `.rs` file, taken from one pass over `git log -p`.

## Tiers

| tier | meaning | count |
|---|---|---|
| **A-old** | default-ON on every arch, first seen before 2026-09-01. These are the strongest removal candidates. | **112** |
| B-recent | default-ON on every arch, first seen 2026-09-01 or later. The rollout may still be in progress. | 327 |
| C-arch-gated | the default depends on arch or caps (for example ON for gfx1151 only). The OFF path is live on other archs, so only the flag itself might go. | 122 |

`signal` is a keyword hit (legacy / rollback / kill / A/B / fallback / restore / opt-out) in the flag's doc comment or its OFF description. Of the 112 A-old flags, 38 have one.

Most common name families across all tiers: QWEN4 156, DEEPSEEK4 41, EP 33, DFLASH 16, CB 16, RAON 16, GFX12 15, GDN 14, MTP 13, MOE 12, GFX1100 12, GLIMMER 9.

## Caveats (read before deleting anything)

- **Removing a flag is not the same as removing its OFF path.** Many "universal" WMMA/MMQ paths still have a capability check downstream, and the fallback serves archs that lack the capability. Check that before deleting fallback code.
- Some OFF paths are deliberately kept as diagnostic or bisect tools (for example the DeepSeek4 `COMP_ROPE_POS` `mid`/`end` modes).
- The agents' "OFF-path size" and "no consumer" cells are unreliable. One spot-checked "no consumer" claim was wrong, so those cells were dropped from the merged table. The read sites and default idioms were spot-checked and are correct.
- A few flags are read in two places (`hipfire-runtime::config` and `rdna-compute::FeatureFlags`) with slightly different idioms. That duplication is a cleanup item in its own right.

## Files

- `default_on_flags.tsv`: the full merged table (561 rows) with tier, env var, kind, first seen, introducing commit, signal, read sites, default idiom, OFF behaviour, and source cluster.
- `clusters/`: raw per-cluster agent tables with notes.

## Tier A: default-ON everywhere, older than 2026-09-01 (112)

| env var | kind | first seen | introduced by | what OFF selects | signal |
|---|---|---|---|---|---|
| `HIPFIRE_PREFILL_BATCHED` | bool | 2026-04-11 | eb629ce62 perf(prefill): batched LA kernel path — 1.78-2.04× | Non-batched prefill kernels | fallback |
| `HIPFIRE_FP16` | killswitch | 2026-04-12 | 438894a18 perf(wmma): 4.1× prefill via WMMA GEMM + turbo→giv | FP16 path disabled (fp16_disabled) |  |
| `HIPFIRE_LM_HEAD_WMMA` | killswitch | 2026-04-16 | d8b06b668 feat(rdna-compute): gemm_hfq4g256_batched_lmhead f | lm_head WMMA path disabled |  |
| `HIPFIRE_GRAPH_MOE` | killswitch | 2026-04-18 | f5d40c513 diag(hipgraph-moe): reproducer + kv_write blob mig | MoE forward not hipGraph-captured |  |
| `HIPFIRE_ROCBLAS_OFF` | killswitch | 2026-04-18 | 2b76548a7 feat(rocblas): HIPFIRE_ROCBLAS_OFF=1 kill-switch f | rocBLAS route disabled | fallback |
| `HIPFIRE_DRAFT_F16` | bool | 2026-04-21 | ec786d050 perf(dflash): F16 draft via mw16 WMMA (+1.8× on 27 | DFlash draft activations in F32 | legacy |
| `HIPFIRE_DDTREE_TREE_LA` | killswitch | 2026-04-22 | dad946f2b feat(dflash): task #101 Phase 3d — populate parent | non-linearized-ancestor ddtree tape (slow path) | fallback |
| `HIPFIRE_NORMALIZE_PROMPT` | bool | 2026-04-24 | 6601a79cb dflash(prompt-shape): Phase 1 — collapse \n{3,} →  | No newline-run collapse before tokenization |  |
| `HIPFIRE_JINJA_CHAT` | bool | 2026-05-09 | a491a709e feat(runtime): Jinja chat-template rendering — Sta | Chat prompts use the hand-rolled ChatScaffold instead of the model Jinja template | fallback |
| `HIPFIRE_MOE_GROUPED_GEMM` | bool | 2026-05-18 | 805d04902 feat(prefill): Path 1 + Path 2 MoE prefill grouped | non-grouped MoE prefill GEMM path (0 / off) |  |
| `HIPFIRE_HFQ4G128_MMQ` | killswitch | 2026-05-22 | c2a76da07 feat(paroquant-prefill): non-grouped i8 MMQ for HF | HFQ4G128 GEMM on the non-MMQ path | kill |
| `HIPFIRE_DEEPSEEK4_ATTN` | mode (default  | 2026-05-23 | b263fb3cc feat(arch-deepseek4): add DeepSeek V4 Flash suppor | pos0 diagnostic attention path |  |
| `HIPFIRE_DEEPSEEK4_COMP_F16_WMMA` | bool | 2026-05-23 | b263fb3cc feat(arch-deepseek4): add DeepSeek V4 Flash suppor | f32 register-tiled compressor GEMM (microbench says ~26x slower) |  |
| `HIPFIRE_DEEPSEEK4_COMP_ROPE_POS` | mode (default  | 2026-05-23 | b263fb3cc feat(arch-deepseek4): add DeepSeek V4 Flash suppor | "mid" / "end" diagnostic RoPE placement | A/B |
| `HIPFIRE_DEEPSEEK4_HFQ4_WMMA` | bool | 2026-05-23 | b263fb3cc feat(arch-deepseek4): add DeepSeek V4 Flash suppor | non-WMMA HFQ4 projection; body in hipfire-dispatch |  |
| `HIPFIRE_DEEPSEEK4_LOAD_MTP` | bool | 2026-05-23 | b263fb3cc feat(arch-deepseek4): add DeepSeek V4 Flash suppor | MTP head not loaded; speculation unavailable |  |
| `HIPFIRE_DEEPSEEK4_MOE` | bool | 2026-05-23 | b263fb3cc feat(arch-deepseek4): add DeepSeek V4 Flash suppor | skips routed-expert dispatch (shared-only FFN) |  |
| `HIPFIRE_DEEPSEEK4_MOE_DETERMINISTIC` | killswitch | 2026-05-23 | b263fb3cc feat(arch-deepseek4): add DeepSeek V4 Flash suppor | expanded/non-deterministic write path instead of the deterministic one |  |
| `HIPFIRE_DEEPSEEK4_MOE_GROUPED` | killswitch | 2026-05-23 | b263fb3cc feat(arch-deepseek4): add DeepSeek V4 Flash suppor | grouped DS4 MoE path off |  |
| `HIPFIRE_DEEPSEEK4_MTP_HEAD_HC` | bool | 2026-05-23 | b263fb3cc feat(arch-deepseek4): add DeepSeek V4 Flash suppor | MTP step 8 reads stream 0 only (pre-fix head) |  |
| `HIPFIRE_DEEPSEEK4_Q8_WMMA` | killswitch | 2026-05-23 | b263fb3cc feat(arch-deepseek4): add DeepSeek V4 Flash suppor | Q8_0 dense WMMA prefill disabled; scalar chunked fallback | fallback |
| `HIPFIRE_DEEPSEEK4_UPLOAD_EXPERTS` | bool | 2026-05-23 | b263fb3cc feat(arch-deepseek4): add DeepSeek V4 Flash suppor | experts not uploaded to GPU |  |
| `HIPFIRE_DEEPSEEK4_Q8_4W` | killswitch | 2026-05-27 | a87a2f43d perf(deepseek4): shape-gated 4-warp Q8 WMMA in pre | 4-warp Q8_0 WMMA kernel not used; single-warp 16x16 kernel instead | fallback |
| `HIPFIRE_MTP_MODE` | mode (auto) | 2026-05-28 | e648ed4d6 feat: wire mtp_mode/mtp_k into global and per-mode | Plain AR for all producers |  |
| `HIPFIRE_DEEPSEEK4_BATCH_HEAD` | bool | 2026-05-29 | 1c480ff55 perf(deepseek4): batch verify lm-head (+2.07%) + g | legacy per-position scalar LM-head loop | legacy |
| `HIPFIRE_DFLASH_MOE_DRAFT_FFN_GRAPH` | bool (default  | 2026-05-30 | 1cafa1923 [codex] Land A3B MTP and DFlash perfmaxx work (#35 | draft FFN runs per layer (not graphed) |  |
| `HIPFIRE_DFLASH_MOE_VERIFY_GRAPH_LMHEAD` | bool (default  | 2026-05-30 | 1cafa1923 [codex] Land A3B MTP and DFlash perfmaxx work (#35 | MoE verify graph without the extended lm_head graph (old forward-only verify graph) |  |
| `HIPFIRE_DFLASH_Q8_LMHEAD_WMMA` | bool | 2026-05-30 | 1cafa1923 [codex] Land A3B MTP and DFlash perfmaxx work (#35 | Q8 DFlash lm_head on the non-WMMA path (opt-out values 0/false/off/no) | opt-out |
| `HIPFIRE_MTP_DEVICE_TOKEN_CHAIN` | bool (Err => t | 2026-05-30 | 1cafa1923 [codex] Land A3B MTP and DFlash perfmaxx work (#35 | host-side proposal loop (host data dependency per step) |  |
| `HIPFIRE_MTP_GPU_ACCEPT` | bool (Err => t | 2026-05-30 | 1cafa1923 [codex] Land A3B MTP and DFlash perfmaxx work (#35 | host prefix-accept and bonus selection |  |
| `HIPFIRE_MTP_Q8_VERIFY_WMMA` | bool (None =>  | 2026-05-30 | 1cafa1923 [codex] Land A3B MTP and DFlash perfmaxx work (#35 | chunked dispatch falls to scalar (no gfx12 WMMA route) | opt out |
| `HIPFIRE_CACHE_CKPT_RESUME` | killswitch | 2026-05-31 | 0884535cc Qwen3.5/3.6 grammar-guided tool calling and prompt | checkpoint resume ring off |  |
| `HIPFIRE_DFLASH_CKPT_RESUME` | bool | 2026-05-31 | 0884535cc Qwen3.5/3.6 grammar-guided tool calling and prompt | No checkpoint-resume ring (loader lib.rs:2308; qwen35 mtp_speculator.rs:470-492) |  |
| `HIPFIRE_QWEN_PROMPT_CACHE` | killswitch | 2026-05-31 | 0884535cc Qwen3.5/3.6 grammar-guided tool calling and prompt | prompt/turn cache disabled (no reuse) |  |
| `HIPFIRE_DUMP_HIDDEN` | killswitch (si | 2026-06-03 | 25720cff3 fix(gemm): correct F32 batched GEMM output layout  | the norm fold is disabled whenever the debug dump is on |  |
| `HIPFIRE_DEEPSEEK4_DSA_WMMA` | bool | 2026-06-06 | ee91233e0 ds4: head-batched WMMA for DSA direct attention (+ | non-WMMA direct DSA attention |  |
| `HIPFIRE_EP_PEER_ALLREDUCE` | mode (default  | 2026-06-07 | e3ab1fc55 ship6(ep): peer-direct 2-rank all-reduce (HIPFIRE_ | explicit 0/1 select the non-canonical all-reduce variant (which one is not verified here) |  |
| `HIPFIRE_FORWARD_LOWERED` | killswitch (!= | 2026-06-07 | 9545e4cc7 feat(dispatch): #397 Ship 6 — C-design super-op su | legacy (non-lowered) qwen2 forward | legacy |
| `HIPFIRE_TP_USE_RCCL` | bool | 2026-06-07 | cf6ad952a ship6(tp): Stage 1+2 — ShardConfig + RCCL all-redu | Tensor-parallel all-reduce without RCCL |  |
| `HIPFIRE_DN_STATE_EF` | tri-state (map | 2026-06-08 | fe0a9c5ba feat(qwen35): cherry-pick q8_ef DeltaNet state req | no per-element f16 error-feedback residual for Q8 DeltaNet state | opt-out |
| `HIPFIRE_EP_PEER_ALLREDUCE_DECODE` | mode | 2026-06-08 | 616829ae9 ship6(ep): keep decode on RCCL (peer-direct opt-in | =1 legacy unrooted peer; =0 RCCL | legacy |
| `HIPFIRE_GATE_UP_BT` | killswitch | 2026-06-08 | 6223db8f3 feat(gfx12): batch-tiled prefill GEMMs (gate_up/re | non-batch-tiled gate_up/residual/qkvza prefill GEMMs |  |
| `HIPFIRE_EP_DUMP_POS` | inverse bool ( | 2026-06-09 | e5e223d43 debug(ep): HIPFIRE_EP_DUMP_POS per-(pos,layer,rank | setting it disables TP graph admission and enables per-position dump |  |
| `HIPFIRE_COHERE2MOE_Q8_SCALAR` | inverse bool ( | 2026-06-10 | 93e6ce684 cohere2moe: WMMA Q8 prefill projections (~9x) + fi | scalar sub-batched chunked Q8 prefill (no WMMA) | fallback |
| `HIPFIRE_MOE_AWQ` | killswitch (!= | 2026-06-11 | 3e5f2e9c1 feat(moe-awq): wire per-expert down AWQ through th | AWQ pointer table not armed (plain MoE expert loading) |  |
| `HIPFIRE_C2M_EMPTY_TURN_GUARD` | bool | 2026-06-12 | 996958732 cohere2moe: O(window) windowed prefill + empty-tur | no EOS suppression / forced think-close |  |
| `HIPFIRE_Q8_PREFILL_WMMA` | mode | 2026-06-12 | dfed8cc65 perf(gfx1151): enable A3B MQ6 MoE prefill fast pat | Q8 prefill WMMA off |  |
| `HIPFIRE_SAMPLE_PARALLEL` | bool | 2026-06-12 | f145aeddd perf(sample): multi-workgroup parallel sample_top_ | single-workgroup sample_top_p |  |
| `HIPFIRE_MTP_HEAD_LMHEAD_WMMA` | bool-style (!= | 2026-06-16 | 1495be040 perf(mtp): route MTP-head lm_head through chunked  | scalar lm_head GEMV fallback | fallback |
| `HIPFIRE_MTP_TAPE_REPLAY` | bool-style (!= | 2026-06-16 | 3085f0df3 feat(mtp): route A3B rollback through GDN-tape rep | original full-replay path (restore + full re-run) instead of GDN-tape replay | original |
| `HIPFIRE_DEFAULT_CHATML` | bool | 2026-06-17 | f0482fd3a fix(serve): never-bare chat-template fallback (def | No fallback ChatML frame when no template resolves | fallback |
| `HIPFIRE_DFLASH_FAST_SAMPLE` | killswitch | 2026-06-23 | 26052d104 feat(serve): sampled DFlash spec-decode with GPU-s | non-fast (host-side) DFlash sampling path for temp>0 | fallback |
| `HIPFIRE_MTP_SAMPLED` | bool | 2026-06-24 | 2988baa90 wip: sampled-MTP route p_min-clear (save before ma | Sampled (temp>0) requests go to AR; no MTP rejection sampling. Consumers: loader carriers.rs:1335, qwen4 mtp_s |  |
| `HIPFIRE_FUSE_QKV_BIAS` | bool | 2026-06-25 | e3ab13cae feat(dispatch): fold qwen2 QKV bias into fused_qkv | three separate QKV BiasAdd launches per layer |  |
| `HIPFIRE_DFLASH_TEMP_SPEC` | killswitch | 2026-06-26 | 5884a8948 feat(serve): enable distribution-correct temp>0 DF | temp>0 DFlash spec decode disabled |  |
| `HIPFIRE_DEEPSEEK4_DSPARK` | bool (EXAMPLE  | 2026-06-28 | a389f7c5f feat(dspark): wire DSpark draft module into spec-d | MTP drafter instead of DSpark in the bench |  |
| `HIPFIRE_DSPARK_HFQ4_WMMA` | bool | 2026-07-01 | 076318599 refactor(dspark): final-review followups (env rena | non-WMMA HFQ4 path |  |
| `HIPFIRE_DSPARK_Q8_4W` | bool (opt-out, | 2026-07-01 | 076318599 refactor(dspark): final-review followups (env rena | non-4-wave Q8 WMMA kernel |  |
| `HIPFIRE_DSPARK_Q8_WMMA` | bool | 2026-07-01 | 076318599 refactor(dspark): final-review followups (env rena | non-WMMA Q8 GEMM path |  |
| `HIPFIRE_DSPARK_ADAPTIVE_BLOCK` | bool | 2026-07-02 | 0d81d047c feat(dspark): drive draft block via BlockControlle | fixed block size (block_controller = None) | fallback |
| `HIPFIRE_SAMPLE_FAST` | bool | 2026-07-11 | fe044f551 perf(redline): accelerate sampled FWHT decode | non-fast sampled FWHT decode |  |
| `HIPFIRE_GFX1201_ROUTER_W64` | killswitch | 2026-07-12 | f55ec99e1 perf(gfx1201): fuse wave64 router at 195.53 tok/s | wave64 router off on gfx1201 |  |
| `HIPFIRE_REPLAY_PM4_GCR_TRIM` | bool (diagnost | 2026-07-12 | cc8546a61 perf(redline): trim gfx12 retained-PM4 GCR actions | No GCR trimming on retained PM4 routes |  |
| `HIPFIRE_GATED_NORM_MQ_ROTATE` | killswitch | 2026-07-15 | 6d1844235 radiowave: fuse gated norm with MQ rotation | gated norm not fused with MQ rotation |  |
| `HIPFIRE_MOE_COMBINE_NEXT_RMS` | arch-gated kil | 2026-07-15 | 994a6fbaa radiowave: fuse routed combine with next RMS | two-dispatch routed-combine + next-RMS schedule |  |
| `HIPFIRE_MOE_COMBINE_NEXT_RMS_RENORM` | inverted kills | 2026-07-15 | 47cae363f radiowave: add combine exactness split probe | =1 turns off reuse of fused rotation |  |
| `HIPFIRE_MOE_GATE_UP_TIGHT_GRID` | bool | 2026-07-15 | cf78a5e52 gemv: add gfx1100 tight gate-up grid experiment | wide (non-tight) MoE gate/up grid |  |
| `HIPFIRE_QWEN35_FA_EPILOGUE_FUSE` | killswitch | 2026-07-15 | d2bb69f00 radiowave: fuse full-attention decode tail | full-attention decode tail unfused |  |
| `HIPFIRE_QWEN35_FA_PREP_FUSE` | killswitch | 2026-07-15 | e740f6875 radiowave: fuse full-attention preparation | full-attention prep unfused |  |
| `HIPFIRE_NO_REGISTRY_FETCH` | kill switch (b | 2026-07-20 | 2a960e190 feat: replace TypeScript control plane with Rust | No registry network fetch (offline) | fallback |
| `HIPFIRE_DEEPSEEK4_E8_PREFILL_B2` | bool | 2026-07-24 | e86184fb2 feat(deepseek4): freeze MQ2R gfx1151 route v2 | B1 admission baseline (one 16-token tile) |  |
| `HIPFIRE_DEEPSEEK4_E8_PREFILL_B4` | bool | 2026-07-24 | e86184fb2 feat(deepseek4): freeze MQ2R gfx1151 route v2 | retains B2 path |  |
| `HIPFIRE_DEEPSEEK4_E8_U4` | bool | 2026-07-24 | e86184fb2 feat(deepseek4): freeze MQ2R gfx1151 route v2 | original one-row E8 decode schedule | original |
| `HIPFIRE_DEEPSEEK4_E8_WO_GROUPED` | bool | 2026-07-24 | e86184fb2 feat(deepseek4): freeze MQ2R gfx1151 route v2 | 8 separate E8 O-LoRA GEMVs (gemv_auto loop) |  |
| `HIPFIRE_DEEPSEEK4_FFN_OVERLAP` | bool | 2026-07-24 | e86184fb2 feat(deepseek4): freeze MQ2R gfx1151 route v2 | serial shared + routed FFN (no overlap) | fallback |
| `HIPFIRE_DEEPSEEK4_HC_CONTROL_FINALIZE_FUSED` | bool | 2026-07-24 | e86184fb2 feat(deepseek4): freeze MQ2R gfx1151 route v2 | unfused HC control |  |
| `HIPFIRE_DEEPSEEK4_HC_PINGPONG` | bool | 2026-07-24 | e86184fb2 feat(deepseek4): freeze MQ2R gfx1151 route v2 | D2D copies (86/token) instead of buffer swap |  |
| `HIPFIRE_DEEPSEEK4_E8_BATCHED_GEMV` | mode (numeric  | 2026-07-26 | b8efa47a7 feat(deepseek4): batched E8 decode GEMV for specul | value 0 keeps all batches on WMMA tile |  |
| `HIPFIRE_NO_DEVICE_COMPILER` | killswitch | 2026-07-26 | 21fe45705 fix(rocm): ROCM_PATH for device compiler, pin over | device compiler treated as absent (set = OFF path) | kill |
| `HIPFIRE_MOE_NINEPATH` | mode | 2026-07-28 | 14a7cc7d7 moe: nine-path fused MoE decode (D3+D4) for unifor | 0/off disable D4/mq4g128v2 nine-path; d3 selects D3 only |  |
| `HIPFIRE_FLASH_PREFILL_KERNEL` | mode (default  | 2026-07-29 | cf7e060d7 feat(dispatch): route flash prefill to WMMA by def | scalar flash prefill kernel |  |
| `HIPFIRE_FLASH_PREFILL_FIXED_HD` | killswitch | 2026-07-30 | 4acea3540 perf(gfx11/gfx12): specialize Q8 prefill and fix r | non-fixed head_dim flash prefill (dynamic head dim, VGPR spill risk) |  |
| `HIPFIRE_FLASH_PREFILL_PREFETCH_V` | killswitch | 2026-07-30 | 4acea3540 perf(gfx11/gfx12): specialize Q8 prefill and fix r | no V prefetch (_pv0 kernel suffix) |  |
| `HIPFIRE_DEEPSEEK4_GFX942_E8_WO_GROUPED` | bool | 2026-08-01 | c27139da4 archive(gfx942): preserve failed DS4 CDNA experime | serial 8-way gemv_auto loop | fallback |
| `HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_PARALLEL` | bool | 2026-08-01 | c27139da4 archive(gfx942): preserve failed DS4 CDNA experime | serial indexer top-K (emergency fallback) | fallback |
| `HIPFIRE_DEEPSEEK4_GFX1151_INDEXER_TOPK_TWOSTAGE` | bool | 2026-08-02 | a098bd274 perf(ds4): two-stage indexer top-K — 2.0x at 128K  | legacy O(N^2) rank-count on one workgroup | legacy |
| `HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_BOUNDED` | bool | 2026-08-02 | 5e4d7505f perf(ds4): gfx942 indexer top-K was O(N^2) on one  | F1 rank-count kernel (O(N^2), single workgroup) | fallback |
| `HIPFIRE_DEEPSEEK4_GFX942_INDEXER_TOPK_TWOSTAGE` | bool | 2026-08-02 | a098bd274 perf(ds4): two-stage indexer top-K — 2.0x at 128K  | falls back to F2 bounded kernel |  |
| `HIPFIRE_DEEPSEEK4_E8_ATTN_PACK_B3` | bool | 2026-08-04 | 7d0535aaa perf(ds4): pack B3 attention projections | per-projection E8 attention input path |  |
| `HIPFIRE_DEEPSEEK4_E8_BATCHED_PAIR_B3` | bool | 2026-08-04 | dd784d73f perf(ds4): pair B3 E8 verify projections | unpaired B3 E8 verify projections; body in hipfire-dispatch |  |
| `HIPFIRE_DEEPSEEK4_MOE_DOWN_BATCHED_K8ALL` | killswitch | 2026-08-04 | 18579dec3 perf(ds4): use K8-all down kernel for batched veri | non-K8-all batched down kernel |  |
| `HIPFIRE_DEEPSEEK4_GFX1100_INDEXER_TOPK_TWOSTAGE` | bool | 2026-08-06 | 4913b5a09 perf(deepseek4): admit gfx1100 two-stage indexer | legacy O(N^2) rank-count kernel | legacy |
| `HIPFIRE_DEEPSEEK4_GFX1201_INDEXER_TOPK_TWOSTAGE` | bool | 2026-08-06 | 8b690b612 perf(deepseek4): eliminate gfx1201 indexer cliff | legacy rank-count kernel (after capacity growth; bounded net for 2048 bucket) | legacy |
| `HIPFIRE_DEEPSEEK4_GFX1201_E8_WIDE` | bool | 2026-08-07 | 984f199b1 perf(deepseek4): route wide gfx1201 E8 prefill til | narrow gfx1201 E8 prefill tiles; OFF body lives in hipfire-dispatch (out of scope) |  |
| `HIPFIRE_DEEPSEEK4_GFX1201_HC_FUSED24` | bool | 2026-08-07 | b48249d8f prefill: deploy bit-exact gfx1201 HC control fusio | 24-workgroup HC route |  |
| `HIPFIRE_DEEPSEEK4_GFX1201_TOPK_GATHER_TILED` | bool | 2026-08-07 | 0cfb8a4b8 perf(gfx1201): default DS4 prefill to tiled top-k  | scattered-store top-K gather |  |
| `HIPFIRE_DS4_GATHER_TILED` | bool | 2026-08-07 | 3bc2b47ee perf(ds4): default the tiled LDS top-K gather on g | scattered-store top-K gather |  |
| `HIPFIRE_FAST_SAMPLE` | killswitch | 2026-08-09 | 1b78d1706 feat: add independent continuous batching for Qwen | slow sampling path |  |
| `HIPFIRE_GEMMA4_GRAPH` | mode (default  | 2026-08-10 | 19986cf65 gemma4: port the additive core of #439 onto beta | eager decode_step instead of captured decode graph | fallback |
| `HIPFIRE_GLIMMER_BATCHED_LM_HEAD` | bool | 2026-08-11 | df345f53f glimmer: share one lm_head routine between draft a | batched lm_head off; logits_out paths error out when off |  |
| `HIPFIRE_GLIMMER_KV_VMM` | bool | 2026-08-11 | 930b6c5c3 glimmer: KV through the VMM arena — 131k context n | KvBackend::Legacy allocator instead of VMM arena | Legacy |
| `HIPFIRE_GLIMMER_NO_CENTERED_NORM` | killswitch | 2026-08-11 | bb77ee8d6 glimmer: centered RMSNorm — the checkpoint stores  | norm weights loaded uncentered (no +1.0 offset) |  |
| `HIPFIRE_GLIMMER_SHARED_ROT` | bool | 2026-08-11 | e4e036d9b glimmer: share one FWHT rotation across co-input p | per-projection rotation instead of one shared FWHT across co-input projections | old path |
| `HIPFIRE_GLIMMER_WMMA_FULL` | bool | 2026-08-11 | ea0c63da4 glimmer: WMMA attention on the full layers — +38%  | batched attention on F32 scalar kernels instead of F16 WMMA query-tiled flash | legacy |
| `HIPFIRE_GLIMMER_FUSED_SILU_ROTATE` | bool | 2026-08-12 | f566d1c1c perf(glimmer): fuse silu multiply and MQ rotation | separate silu-multiply and MQ rotation, not fused | Opt-out |
| `HIPFIRE_GLIMMER_GPU_SAMPLE` | killswitch | 2026-08-12 | edd5b4668 glimmer: sample on the GPU and stop dropping top_k | glimmer host sampling |  |
| `HIPFIRE_GLIMMER_PROMPT_CACHE` | killswitch | 2026-08-12 | f1714f423 glimmer: fix chat template parse + add LCP prefix  | glimmer LCP prefix cache disabled |  |
| `HIPFIRE_GLIMMER_SPEC_PROFIT_GUARD` | killswitch | 2026-08-13 | 914774d79 feat(glimmer): retire unprofitable draft requests | draft-request profit guard off |  |
| `HIPFIRE_CALIB_HESSIAN_STORAGE` | mode | 2026-08-18 | 3243261ea quant: add MQ4 v2 and MQ4C quality stack | f32 / dense-f32 / full-f32 / legacy → DenseF32 | legacy |
| `HIPFIRE_GDN_COMPACT3` | killswitch | 2026-08-18 | e8b04f826 perf(gfx1100): accelerate Qwen3.6 MQ4 decode pipel | GDN compact-3 path off |  |
| `HIPFIRE_GFX1100_DENSE_GATE_UP_STAGE_X32` | bool | 2026-08-18 | e8b04f826 perf(gfx1100): accelerate Qwen3.6 MQ4 decode pipel | non-staged-X32 dense gate/up |  |
| `HIPFIRE_MQV2_GFX11_WMMA` | killswitch (va | 2026-08-20 | dc105ea64 perf(mqv2): add gfx11 WMMA prefill family | batched WMMA prefill not admitted for MQ-V2 on gfx11 (falls to non-WMMA path) | kill |
