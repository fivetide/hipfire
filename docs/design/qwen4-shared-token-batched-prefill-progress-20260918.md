# Qwen4 shared token-batched prefill — implementation record

Date: 2026-09-18

This is the mutable implementation/progress record for
[`qwen4-shared-token-batched-prefill.md`](qwen4-shared-token-batched-prefill.md).
The plan remains immutable. This record is updated with concrete ownership,
changed files, evidence, and blockers as work lands.

## Contract accepted before implementation

CodeGraph was attempted first (`which codegraph` / `command -v codegraph`) and
is unavailable on this checkout. The repository rules, kernel-tuning workflow,
`docs/VALIDATION.md`, `docs/ARCHITECTURE.md`, and
`docs/design/sealed-granular-moe.md` were read before source work.

The production contract is:

1. **Family binding only in `hipfire-arch-qwen4`.** Qwen4 owns typed layer and
   weight declarations, resource binding, fixed-capacity batch scratch, state
   handles/marks, PLE epoch/lease ownership, and requested output-row policy.
   It does not own a second prefill interpreter or a token×layer sequencing
   loop after migration.
2. **Common lowering/execution in `hipfire-dispatch`.** The existing
   `pipeline::Step` / `execute_steps` chokepoint remains the only ordered
   executor. It already owns GEMV/GEMM/attention lowering and sealed MoE
   granular stages. Qwen4's new layer program will use semantic, row-batched
   steps (not one public operation per kernel launch); the dispatch side owns
   launch selection, shape checks, alias/lifetime checks, and stateful launch
   order. No callback hides an architecture-owned launch interpreter.
3. **MoE reuses the existing sealed call.** Qwen4 binds
   `MoePrefillParams` and enters `seal_prefill`/`seal_prefill_ep`, then the
   existing granular `Step::Moe*` lowering. The only new admission is the
   exact Qwen4 geometry: `n_exp=512`, `k_top=10`, `hidden=2560`,
   `mi=640`, routed gate/up `MQ4G256V2` (QT44), routed down `MQ4G128V2`
   (QT53), with QT53 allowed only in the already-audited Qwen4 placements.
   Route stamps, source/device/lease identity, complete-call preflight,
   fences, dirty-scratch initialization, canonical slot order, and weighted
   combine remain mandatory. Other families keep their existing semantics.
4. **Execution shape is genuinely layer-major.** A bounded chunk is embedded
   into reusable `[B,*]` buffers. For every layer, all independent projections
   use shared batched GEMM/GEMV machinery. GDN recurrence/convolution, QSA
   append/selection/attention, and PLE depthwise/history updates execute in
   causal row order inside that layer; no future row is visible. This is an
   ordered stateful kernel region surrounded by real matrix work, not a scalar
   loop wrapped in `Step`s.
5. **Outputs and MTP stay explicit.** Ordinary AR prefill computes only the
   requested final logits row unless a caller requests all rows or an
   intermediate/wide capture. Native MTP remains a separate token-shaped
   state owner, but reuses common projection/activation/attention operations
   where applicable and does not silently claim batched MTP.
6. **Bounded resources.** Scratch is allocated once at attach time for the
   configured chunk cap; QSA/GDN/PLE state arenas are not widened to context
   or duplicated per token. PLE uses its existing bounded reader, lease, and
   epoch cleanup protocol.

## Concrete file ownership

### Existing shared owners to extend

- `crates/hipfire-dispatch/src/pipeline/steps.rs`: add only the semantic
  row-batched steps required by Qwen4 plus their complete-call preflight and
  launch arms; keep `Step::Moe*` as the sealed MoE authority.
- `crates/hipfire-dispatch/src/pipeline/mod.rs`: shared Qwen4 lowering helpers
  and exact QT44/QT53 grouped down selection; generic MoE rejection remains
  fail-closed outside the exact Qwen4 predicate.
- `crates/hipfire-dispatch/src/families/gemm.rs` and existing rotation/
  attention families: reuse, do not duplicate, batch projection dispatch.
- `crates/hipfire-dispatch/src/families/moe.rs`,
  `pipeline/moe_program.rs`, and `pipeline/sealed_moe.rs`: preserve and, where
  necessary, complete the exact Qwen4 top-10/QT44/QT53 shared route.
- `crates/rdna-compute/src/qwen4.rs` and `kernels/src/qwen4_ops.hip`: add
  row-batched state/pointwise wrappers only where existing scalar Qwen4
  operations cannot express the required bounded layer program. Quantized
  projection kernels remain in their existing shared GEMM/MoE owners.

### Qwen4 family binders and state owners

- `crates/hipfire-arch-qwen4/src/gpu_forward.rs`: reduce to typed resource
  binding plus invocation of the shared layer program; retain public
  `forward_token`/`forward_chunk` and scalar compatibility through the same
  program, not a second interpreter.
- `crates/hipfire-arch-qwen4/src/bundle.rs`: preserve the attach/unload owner
  and expose the bounded chunk/output-row contract without per-call device
  allocation.
- `crates/hipfire-arch-qwen4/src/state.rs`: retain GPU state allocation,
  snapshot/restore, and marks; expose only typed state views/counters needed by
  the shared executor.
- `crates/hipfire-arch-qwen4/src/ple_rows.rs`: retain bounded source reader,
  leases, epochs, and cleanup; the shared program consumes a borrowed lease at
  the configured layer boundary.
- `crates/hipfire-arch-qwen4/src/projection.rs`: retain Qwen4 QT44/QT53
  logical/encoded view geometry and route matching; use shared batch dispatch
  for projections.
- `crates/hipfire-arch-qwen4/src/mtp_gpu.rs` and `mtp_spec.rs`: preserve
  native-MTP ownership and refusal/lifecycle semantics; migrate only reusable
  numerical operations, not MTP sequencing.

### Tests/evidence record owners

- Existing Qwen4 CPU equation/shape tests and dispatch tests remain the
  narrow no-GPU contract tests. New tests defend chunk tiling, requested-row
  output, causal state transitions, exact Qwen4 admission/refusal, and
  preflight-before-effect behavior—not enum/source wiring.
- GPU/model validation belongs to the delegated baseline/validation worker and
  the main session. Do not overwrite baseline binaries or caches before the
  baseline-saved signal.

## Pre-existing dirty tree (preserved, not authored by this implementation)

At record creation, `git status --short` reported:

- `crates/hipfire-arch-qwen4/examples/qwen4_parity.rs`
- `crates/hipfire-arch-qwen4/reference_oracle/equations.py`
- `crates/hipfire-arch-qwen4/reference_oracle/upstream.py`
- `crates/hipfire-arch-qwen4/src/gpu_forward.rs`
- `crates/hipfire-arch-qwen4/src/mtp_gpu.rs`
- `crates/hipfire-dispatch/src/pipeline/sealed_moe.rs`
- `crates/rdna-compute/src/moe.rs`
- `crates/rdna-compute/src/qwen4.rs`
- `kernels/src/moe_router_softmax_top10_f32.hip`
- `kernels/src/qwen4_ops.hip`
- `.slim/`
- this design plan

The implementation must layer on these changes and never reset, checkout, or
otherwise destroy them. The two Qwen4 source/kernel files above already carry
uncommitted correctness fixes; edits will be narrow and preserve their intent.

## Progress

- Contract reported to `Main`; implementation authorization received.
- Required design/rules/validation reads complete.
- No source edits or builds performed yet in this session.
- Baseline binary overwrite guard remains active: wait for the baseline-saved
  signal before any build that can replace retained outputs.

## Evidence and blockers

- CodeGraph: unavailable on host; repository exploration used narrow `read`,
  `grep`, and `glob` calls instead.
- Existing oracle failures reported by the user (GDN and routed index 16) are
  named blockers/ground truth. They must not be rerun merely for confirmation
  or hidden by tolerance changes.
- Physical EP2/EP4 and any unavailable Qwen4/G5 admission hardware remain
  validation blockers, not implementation fallbacks.
- No numerical or performance claim is made by this record until the matched
  baseline/candidate routes are run and recorded by the main validation flow.

## Changed files (implementation-owned)
## Source checkpoint — 2026-09-18

The exact Qwen4 prefill route now has a cohesive shared lowering module
(`crates/hipfire-dispatch/src/pipeline/qwen4_prefill.rs`) and the sealed
selection/route producer reaches it only for the exact replicated
512-expert/top-10/QT44/QT53 geometry.  Qwen4 EP remains rejected before GPU
work.  QT53 shared-down uses a dense row-batched projection rather than a
per-row scalar loop; shared G128 rotation is performed once before that
projection.  Grouped QT53 down rows are BF16-rounded before weighted combine,
matching the scalar route's per-expert output boundary.  The Qwen4 shared
residual fold uses a row-batched source-exact BF16 product/add operation.

Central dispatch ownership is now represented by
`crates/hipfire-dispatch/src/pipeline/qwen4_program.rs`, registered from
`pipeline/mod.rs`.  Its typed descriptors (`Qwen4LayerDescription`,
`Qwen4ProgramDims`, state views, scratch views, and `Qwen4MoeBinding`) and
`execute_layer` batch BF16 HC/GDN/QSA projections while keeping recurrent/QSA
state transitions ordered and routing batched MoE through `seal_prefill` and
`Step::Moe`.  Family binding/refactor into this executor remains the next
integration boundary; the module is not yet an end-to-end product path.

Scoped evidence:

- `cargo check -p hipfire-dispatch --features deltanet` passed after the
  centralized module and QT53 batched additions.
- Earlier `cargo check -p hipfire-arch-qwen4` passed before the central module
  was introduced; no Qwen4 family wiring claim is made from that result.

Changed files authored by this implementation (append-only):

- `crates/hipfire-dispatch/src/pipeline/qwen4_prefill.rs`
- `crates/hipfire-dispatch/src/pipeline/qwen4_program.rs`
- `crates/hipfire-dispatch/src/pipeline/mod.rs`
- `crates/hipfire-dispatch/src/pipeline/moe_program.rs`
- `crates/hipfire-dispatch/src/pipeline/sealed_moe.rs`
- `crates/hipfire-dispatch/src/families/gemv.rs`
- `crates/rdna-compute/src/gemm.rs`
- `crates/rdna-compute/src/kernels.rs`
- `crates/rdna-compute/src/qwen4.rs`
- `kernels/src/gemm_mq4g128v2_batched.hip`
- `kernels/src/qwen4_ops.hip`

None yet. This list is append-only for files changed by this implementation;
pre-existing dirty files above remain separately identified.

## Source checkpoint — post-proof cleanup

The immutable plan remains unchanged. The shared execution implementation now
includes the Qwen4 typed layer program, bounded chunk/final-row output policy,
complete-call MoE preflight, QT44/QT53 grouped routing, and exact BF16
multi-row dense projection. Temporary stage dumps, input replay, and the
unused `Qwen4LayerScratch::moe_input` view were removed after validation.

Validation evidence retained outside the source tree:

- QT53 dense probe: `.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/qt53-dense-probe.json` (report SHA256 prefix `f0a84d3422`; source SHA256
  `6fc79077de05b580c99abfba8586d19d8c8449e98a14ed1faf8b4d9013da9d5f`; exact
  real layer-0 expert and synthetic cancellation comparisons for N=2/3/5).
- Immutable fixed BF16 oracle binary SHA256
  `5cb6b4a3269a9006788c88e046e3a6a6188504b7c84b6e62ea34c6665fe8be3a`;
  cap-4/cap-8 final-row and state checks are exact.
- Fresh AR/MTP smoke reports and logs are under
  `.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/`;
  preserved fixed binaries are separate from the later source cleanup.
- `cargo build --release --workspace --all-targets --locked` passed on the
  post-cleanup source; log:
  `.codeinsight+research/qwen4/shared-token-prefill/final-verification/build-release-workspace-all-targets-locked-final.log`.

The current acceptance blocker is a fresh warm AR context-291 transcript
difference between the baseline and candidate despite identical request bytes
and settings. Validation is isolating exact tokenizer-produced token IDs at
the first divergent token across scalar, cap-128, and cap-4 executions. The
small retained oracle and QT53 channel probes do not substitute for this
long-state boundary evidence; no final product-equivalence or performance
claim is made until that isolation completes.

## Verification checkpoint — host gates

The post-cleanup source passed `cargo build --release --workspace
--all-targets --locked` and the explicit touched-Rust `rustfmt --check`;
the logs are retained under `final-verification/`.  The full workspace
library-test run recorded in
`test-lib-workspace-locked.log` ended with passing test results before the
later mechanical launcher `Vec`-to-fixed-array edits.  Its post-array rerun
was intentionally canceled to release the GPU lock for the context-291
isolation oracle; the post-array release all-target build is the compile
proof for those edits.

The no-GPU command was run as:

`LD_LIBRARY_PATH=/nix/store/1xw5xccqqh1xw3mvd70hyil6x418wxcm-gcc-14.3.0-lib/lib scripts/no-gpu-ci.sh`

The NixOS `libstdc++` path allowed the Rust check and no-GPU unit-test
sections to complete.  Python collected 387 tests: 381 passed and 6
failed.  The evidence is exact: one failure is the host's absent
`/bin/bash` in the KernelAtlas shell-chain test; five are
`tests/test_mq4c_repack.py` failures because its imported `mq4c_repack`
module lacks `HfqmError`, `main`, and `parse_hfqm_index`.  This cutover did
not edit that test, its provider module, or their API.  Because the script
stops at that Python section, the remaining phases were run individually
with the same `LD_LIBRARY_PATH`: Redline's 179-test log records the
MQ4R registry-card hash mismatch and mocked cargo-argv failures in
`tools/redline/tests/test_golden.py`; install revision has one host
failure from a temporary fake-cargo `#!/bin/bash` shebang; uninstall
passes, and the environment/docs check passes.  This cutover did not edit
the registry card, `tools/redline/tests/test_golden.py`, or the installer
tests/scripts.  Exact logs are `no-gpu-ci-libstdcxx.log`,
`no-gpu-redline-unittest.log`, `no-gpu-install-revision.log`,
`no-gpu-uninstall.log`, and `no-gpu-env-docs.log`.

The reachable layering/registry checks were also run independently:
quant registry and arch-dispatch checks pass.  The frozen layering check
reports `hipfire-arch-qwen4` absent from `scripts/layering.txt`; the
dispatch-bypass check reports the same crate's three calls absent from
`docs/governance/debt-dispatch-bypass.txt`.  The user-provided branch
scope identifies that Qwen4 governance entry as a prerequisite from before
this task; it is separate from the API cutover, and neither governance
file was edited.  A narrow `git diff --name-only` over those files and the
failing test/provider/install paths is empty; the captured evidence is
`untouched-gate-scope.log`.  The complete `leanup-ratchets.sh` result is
retained in `leanup-ratchets.log`: its seven threshold failures are exactly
`daemon_arch_id`, `daemon_arch_refs`, `daemon_lines`, `ungated_examples`,
`layer_unlisted_crates`, `bypass_unlisted`, and `bypass_total`.  No ratchet
budget inflation or governance edit is made here.  The Qwen4 exported-step
seams themselves pass the focused `qwen4_program` (2 tests) and
`sealed_moe` (34 tests) suites.

Qwen4 product PM4 admission remains out of scope, and physical EP2/EP4
validation is unavailable.  The canonical qwen3.8-27b.mq4-xt dense-trunk
retained regression harness is locally runnable on gfx1151; because the
common Step interpreter changed, that regression proof remains owed to the
validation worker after context-291 isolation.  The warm context-291 scalar
versus batched diagnostic is still active; this record is provisional and
makes no completion, product-equivalence, performance, or marketing claim.

## Verification checkpoint — shared retained host regression (2026-09-18)

The post-cleanup host verification is now recorded without changing the
immutable plan or any prior checkpoint.  The canonical dense-trunk fixture was
the already-verified
`~/.hipfire/models/qwen3.8-27b.mq4-xt` (SHA256
`9f91556f7e0431a077d03756a7102d0154108757289e6e5fe9a2d204c0c9eeb7`), loaded
as `qwen3_5` on `gfx1151`.  The harness used a private `HOME` and kernel cache,
`HIP_VISIBLE_DEVICES=0`, manual capture, and `HIPFIRE_REPLAY_BACKEND=shadow`.
No Qwen4 retained admission or product claim is made.

Reachable scoped retained evidence:

- The exact AQL route was run with
  `scripts/redline_daemon_harness.py --skip-prefill --decode-context 32
  --shadow-iterations 15`.  Capture was stable at 1,139 launches, 17 unique
  kernels, sequence hash `7a1072b2b002bce1`; the AQL contract probe reported
  17 kernels.  Fifteen-position HIP/AQL/blob shadow parity passed:
  `bit_exact`, `blob_bit_exact`, logits, KV, recurrent state, GDN state, and
  `gdn_frame_exact` were all true; dispatches `1139`, packets `1140`, and
  `queue_id=2`.  The report is
  `.codeinsight+research/qwen4/shared-token-prefill/final-retained/dense-trunk-aql-context32-final.json`
  (SHA256
  `40f0a1269ad6dce1e18e3643c44b0e7311f039aa409e1de2b4b7553aa6065715`);
  raw daemon and stdout logs are retained beside it.  This is scoped AQL
  discovery/correctness evidence, not a full default-route acceptance or
  timed-arm Redline certification.
- The same route with `--pm4` reached stable capture and the same 17-kernel
  AQL contract probe, then failed closed during PM4 preparation:
  `gemv_mq4g256v2_residual: GFX10/GFX11 PM4 dispatch does not yet support
  scratch (private=20, dynamic_callstack=false)`.  The raw PM4 daemon/stdout
  logs are
  `.codeinsight+research/qwen4/shared-token-prefill/final-retained/dense-trunk-pm4-context32-final.log`
  and `.stdout.log`; no PM4 pass or promotion claim is made.
- The canonical default `--decode-context 128` attempt failed before capture
  during Qwen prefill prime with
  `valid_lane_mask: max_batch must be 1..64`.  Context 32 was used only to
  reach the bounded diagnostic path; it is not full default-route coverage.
  Source provenance is retained in
  `.codeinsight+research/qwen4/shared-token-prefill/final-retained/lane-mask-provenance.txt`:
  `valid_lane_mask` is unchanged from commit `8cf5eb0389` (2026-08-15), the
  sequential `valid_lane_mask(n)` call is from `27abd53e46` (2026-09-16), and
  the current working diff only changes `execute_steps` arguments from
  `&[Step]` to `&mut [Step]`.  The failure therefore predates this Step
  mutability cutover; no unrelated source edit was made.

Final host test proof:

- `flock -w3600 /tmp/hipfire-gpu.lock` plus
  `flock -w3600 /tmp/hipfire-build.lock`, with `RUST_TEST_THREADS=1`,
  `cargo test --lib --workspace --locked` completed with exit code 0.  The
  raw log is
  `.codeinsight+research/qwen4/shared-token-prefill/final-verification/test-lib-workspace-locked-post-harness.log`.
  Warnings were existing dead-code/unused-variable warnings; no test failed.
- The current daemon binary used by the retained harness has SHA256
  `e0112c3b95f5a44528be1dc164a60874251dc3f09485bc4616da18830c8341ec`.

The warm context-291 scalar-versus-batched diagnostic remains active, the
scalar mismatch remains an acceptance blocker, and this checkpoint makes no
completion, product-equivalence, performance, PM4-admission, or marketing
claim.

## Verification checkpoint — source-matched scalar parity (2026-09-18)

The preserved old-source checkout was reconstructed from commit
`96553691be5eac650039adc7b3b376ff58606070` plus the tracked patch
`f54e646e9b39b738e1c80f3b6078b89b9df41e49606056b7ff33727b96d290a`.
Its rebuilt daemon is byte-identical to the saved baseline daemon
(`sha256=a2134f43cc7b7f34ac7fef19f60b096b84f70cece71c52d398a6a4e08babb025`,
`md5=a3fc2bd319cb0871c1f477460473e456`).  The model identity is
`sha256=7dcbceb4f501a66abef81cc624b86a4da250850ba79acf8dcc6a5204a5c8303f`,
`md5=fda74d3760dc803e778e9b30a2fe0ebd`.

The exact 291-token manifest is shared by old and candidate
(`payload sha256=e0dd3b389ed16741371b44229b355896105d83d657703fd6242aa6b75a13dbee`);
the source-matched serve request remains
`prompt_md5=973900074bfd15d4adeeecdff3359082`,
`request_md5=c634a65ab0a325960028c431c3cd7534`.  A fresh old-source scalar
cap-1 oracle and the current candidate cap-1 oracle are bit-identical for
all 291 full-vocabulary logits (`sha256=87394feac971f9a20b88d6d59511663c4f9faef05233a2401063923f81bc8f1e`)
and for captured states at positions 128, 256, and 291
(`13f452e641f322142aa8d52142caf1e3fade720be517720d8654ee18a1e56a0d`,
`b8cd2c9c486a98d636093c37876fc321f3555ad7c4b708a2cb6e76970ac5bbbd`,
`362aef3fcd5fb5abf499c668ccfc489bc6a39f3cd22ea66be5efaf37989ee382`).
There is no differing row or compared boundary tensor.  The compact
provenance/equality record is
`.codeinsight+research/qwen4/shared-token-prefill/baseline-source-matched/scalar-parity-summary.json`.

This establishes scalar/cap-1 parity, not product continuation equivalence:
the warm product continuation/transcript mismatch is not yet resolved.
`qwen4-completion-check` owns the remaining product caller/current-binary
investigation; no product-equivalence or performance claim is made.

## Verification checkpoint — post-HC host validation (2026-09-18)

The HC-specific source fix restored the missing `hc_read_batch(gpu, dims,
&layer.attn_hyper.read, &streams, scratch, rows)?` immediately before the
attention lowering in `crates/hipfire-dispatch/src/pipeline/qwen4_program.rs`.
The semantic source identity used for the numerical/product build was
`a553fe861610e79776c9b9856fcd92329e5200a887cb798c99d76a8a178847b8`.  After
the requested formatter-only collapse of that call to one rustfmt line, the
final `qwen4_program.rs` SHA256 is
`f1f37bfd5f19aea3301f5de7ff455e710097b66197e6436b67f3cf4cf13377b3`.
No semantic source change was made by the formatter correction; the product
binaries below were built from the semantic source and retain their identity.

Host gates, with raw logs retained under
`.codeinsight+research/qwen4/shared-token-prefill/final-verification/`:

- `flock -w 3600 /tmp/hipfire-build.lock cargo build --release --workspace
  --all-targets --locked` passed without taking the GPU lock.  The exact
  product paths are `/home/bjoern/hipfire/target/release/daemon` (MD5
  `cda02f6fc9688e215ec8ad95e4f3c9dc`, SHA256
  `bb346923ed1d974282d2b0068a6df16bcac6230499800f89ea9ed931671de53f`) and
  `/home/bjoern/hipfire/target/release/hipfire` (MD5
  `12553c586915590ccc2f355c4d513446`, SHA256
  `4e9211dc6f28fb6a671e045034a645c2619f493d2521223ed76cc946ebb633de`).
  Log: `post-hc-build-release-workspace-all-targets-locked.log`.
- With both `/tmp/hipfire-build.lock` and `/tmp/hipfire-gpu.lock`, `RUST_TEST_THREADS=1
  cargo test --lib --workspace --locked` passed with exit code 0; every
  workspace library suite in the log passed.  Log:
  `post-hc-test-lib-workspace-locked.log`.
- With `/tmp/hipfire-build.lock` and no GPU work, `cargo clippy --workspace
  --all-targets --locked` passed with exit code 0.  The log records the
  repository's existing warnings; no warning suppression or `-D warnings`
  change was used.  Log: `post-hc-clippy-workspace-all-targets-locked.log`.
- Scoped `rustfmt --edition 2021 --check --config skip_children=true` now
  passes all 33 touched Rust paths.  The 13 post-baseline introduced files
  were formatted only in their owned hunks, and the one additional
  `qwen4_bf16_scaled_add_batched` call hunk inside initially dirty
  `crates/rdna-compute/src/qwen4.rs` was formatted narrowly.  Initial dirty
  baseline hunks were otherwise preserved.  Classification and final proof:
  `post-hc-fmt-classification.log`, `post-hc-fmt-owned-rust.log`,
  `post-hc-fmt-classification-final.log`, `post-hc-fmt-final.log`.
- The existing gfx1151 cache identity is recorded in
  `post-hc-kernel-cache-identity.log`: 342 `.hash` files, sorted content
  manifest SHA256
  `a03ddd17a2b356569a8a9caac58c1ed18dd1e1eeed2b3a45ed7fceefab58e863`.

Post-fix HC numerical evidence on gfx1151 is now exact for the scoped oracle
routes:

- The fresh cap-1 two-token report
  `fixed-cut-current/runs/post-fix-cap1-two.json` (SHA256
  `a1ca46140142ef73d0dc80e3325ae7d4c7407f209d7fbad9556ad7c60e3b3438`)
  matches the old scalar rows exactly: row hashes
  `f4acc642299657321271b9db58aea0fb04f7f2d23f00fd88e3352a5de55769dc` and
  `23a42e3aec9723bfd740eb62ae9b07968c8f5a155a27a58b1882cd77e2584e90`.
- The cap-1 versus natural cap-128 first-129 comparison passed at boundaries
  `[128, 129]`; the shared raw comparison SHA256 is
  `94703015d94d910050bd6d166b4a05b503ca8700dfb1d209423e08837d2d28c0`.
  Reports are `fixed-cut-current/runs/post-fix-cap1-first129.json` (SHA256
  `02caf97a238b670a5750d4378e47a0be0a0919914c350d732db0bc8c71f3d62e`) and
  `fixed-cut-current/runs/post-fix-cap128-first129-natural.json` (SHA256
  `0f43a3c5f5195fcae7e99154bb28417b0219a17a6308317a3727af88b99d44f1`).
- The natural final-only slices `[0,128]`, `[128,256]`, `[256,291]` passed
  in `fixed-cut-current/runs/post-fix-cap128-final-api.json` (SHA256
  `30e7bbd3ca037bbc7139ec00777ff6ca36c2c678312b1611b4f2c66c416ba530`):
  all 291 rows were finite with `max_abs=0.0`, and the final state comparison
  was exact.
- Continuation/reset passed for 17 rows with full/partition/token logits and
  final state `max_abs=0.0`; report
  `fixed-cut-current/runs/post-fix-continuation-reset.json` (SHA256
  `10a9ba002840f899e673b6255cd896394f59b8e0cc84945f20f5693ff2212bc8`).
- The repeated dirty-combine probe passed for three tokens × hidden 2560:
  repeated expert IDs and nonzero residuals were exercised, grouped/indexed
  one-pass and two-pass outputs were CPU bit-exact, and every compared
  `max_abs` was `0.0`.  Report:
  `fixed-cut-current/runs/post-fix-dirty-combine.json` (SHA256
  `75fabbb03397a1366d29bb2e066030d6521780896cc197613a4b338030ca881f`).

This checkpoint is HC-specific oracle and host-gate evidence.  It does not
rerun dense retained replay, does not change the previously documented AQL
context-32 / PM4 scratch and default-128 lane-limit evidence, and makes no
Qwen4 retained admission, PM4 promotion, or performance claim.  The
performance worker now owns the clean product-CLI continuation and profile
routes using the exact binary identities above.

## Verification checkpoint — current implementation and compiler-enabled product smoke (2026-09-18)

The implementation checkpoint is now source-matched to the current product
build.  The Qwen4 family crate remains a typed binding/descriptor layer; the
shared lowering and execution choke point is `Step`/`execute_steps` in
`hipfire-dispatch`.  Qwen4 execution is layer-major over bounded slices:
shared matrix work uses the bounded slice while stateful GDN/QSA/PLE transitions
retain row order.  The sealed MoE route retains the exact expert geometry
(`n_exp=512`, `top_k=10`, `hidden=2560`, `mi=640`, QT44 gate/up, QT53 down)
and its route/lease/fence/dirty-initialization/slot-order contract.  The
final-only API returns the final row and final state; `bundle.reset` is the
reset boundary used by the continuation proof.  Native MTP remains a separate
token-shaped route and is not folded into the ordinary AR final-row policy.

The HC correction is the current-stream attention preparation immediately
before GDN/QSA:
`hc_read_batch(gpu, dims, &layer.attn_hyper.read, &streams, scratch, rows)?`.
It prepares the layer attention input from the current streams before GDN/QSA
consumes `moe_input`; this is not a claim that attention follows the same
layer's later MLP/MoE sequence.  The semantic source identity used for the
product build was `a553fe861610e79776c9b9856fcd92329e5200a887cb798c99d76a8a178847b8`;
the formatter-only final source identity is
`f1f37bfd5f19aea3301f5de7ff455e710097b66197e6436b67f3cf4cf13377b3`.

The historical PLE failure is distinct from the HC omission.  In the old
max-128 path, the layer-1 PLE application was guarded by
`ple.lease.is_none()`, so row 0 acquired/applied the PLE lease while later
rows skipped it; the first divergence was row 2 and the state divergence was
`gdn[1].recurrent`.  The corrected bounded reader/lease/epoch cleanup and the
current HC read are both exercised by the post-fix reports; no tolerance
relaxation, cache fallback, special-case input, or production debug hook was
added.

Compiler-enabled product smoke used fresh private caches and the current
source; `HIPFIRE_NO_DEVICE_COMPILER` was not set.  The compiler identity was
HIP 7.2.53211-9999 / clang 22.0.0 under
`/nix/store/lqklrnx2bc9k765jyxc0d8q6h15wlybb-clr-7.2.3`.  The final AR16
result is
`.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/final-ar16-compiler/result.json`
(SHA256 `cab493e281e2a089f1a64acfc084c8b15df813b45affcb124eb14bbb51e60940`);
it is `ctx=291`, `gen=16`, `finish=length`, with one terminal frame and the
current-source-matched no-speculation text
`The text you provided appears to be a mix of a **pangram**`.  Its explicit
scalar comparison record is
`.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/final-ar16-compiler/scalar-comparison.json`
(SHA256 `077263f8d8de098214b1728a1b871314cd6eaf7d22ece712ceec75fe41d190b6`).
That record now carries a direct scalar-vs-batched greedy continuation
proof; it no longer relies on the teacher-forced prefix/state comparison for
the 16-token product transcript.

The native MTP smoke is
`.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/final-mtp-smoke-compiler/result.json`
(SHA256 `d7c4cca76b155a4d1b9f6d6eb0546abb3f1b9cc3d53ff7ca389a2e8ac565c6c3`):
`ctx=42`, `gen=16`, `finish=length`, `mtp=true`, `cycles=6`, `tau=1.5`, and
the expected Paris continuation.  Both product runs use daemon SHA256
`bb346923ed1d974282d2b0068a6df16bcac6230499800f89ea9ed931671de53f`,
CLI SHA256 `4e9211dc6f28fb6a671e045034a645c2619f493d2521223ed76cc946ebb633de`,
and model SHA256
`7dcbceb4f501a66abef81cc624b86a4da250850ba79acf8dcc6a5204a5c8303f`.

The prior source-matched checkpoint's statement that product continuation
equivalence remained unresolved is historical/provisional for this final
source-matched smoke: it must not be replaced with the old product
multirow-PLE-bug transcript.  The exact continuation fixture below is the
authoritative sampled-id/logit/state comparison.

## Verification checkpoint — direct scalar-vs-batched greedy continuation (2026-09-18)

The direct fixture
`.codeinsight+research/qwen4/shared-token-prefill/fixed-cut-current/runs/greedy16-scalar-batched/result.json`
(SHA256 `b5eefc6ad17bdfccbe6cd69ecefe929acb5711d105e17d35ba31e517d8e1b3bc`)
loads one current bundle, resets it, runs scalar `forward_token` prefill over
the exact 291-token manifest, and takes the final-row greedy argmax.  It then
resets the same loaded bundle, runs the natural `[0,128]`, `[128,256]`,
`[256,291]` final-only path, and applies the identical standard 16-token
greedy schedule (prefill final-row argmax plus 15 `forward_token`
transitions).  This is a direct full-16 continuation proof, not an inference
from the earlier 17-row teacher-forced reset test.

The result is `status=pass`: scalar and natural-128 prefill final logits and
state are bit-exact; all 16 emitted IDs, all 16 logits rows, and all 16
captured per-step states are bit-exact.  Both ID sequences are
`[760,1414,488,3766,7701,310,381,264,6311,314,264,2972,79,512,2319,332]`,
which decode to
`The text you provided appears to be a mix of a **pangram**`, exactly the
final AR16 transcript.  The scratch oracle identities are binary
`52c9eef230b082d32736a191659cae74404b334eedb2a781a16efcd40854c507`,
source `1899c7d8f1990edf5bdef839438a83305883db0d017a46af34608ec674f77846`,
`Cargo.toml` `aaf5600d29f73f3e8696bcab346f420aafac8062f608ce0884d6c2d8e71b99b1`,
and `Cargo.lock`
`974133b46fd6bba9571c55202a7f9d6eea828b9f0589b977a7ef3558217dc50d`.
The compiler-enabled private cache contains 88 files with listing SHA256
`dbf4e11d6d6550f95464c5786ccf37a2b074c2b52d35b6d262aba317935dfd64` and
content SHA256
`90b2d76079620bf715c87b10ab04eef4bd03eb35502dfcbafa701ae504d0ec5c`.

Remaining boundaries are unchanged: physical EP2/EP4 validation is
unavailable, Qwen4 retained admission/PM4 promotion is out of scope, dense
retained replay was not rerun, and no performance/marketing claim is made by
this checkpoint.  Clean product benchmark/profile evidence remains owned by
the performance worker; the final scoped measurement follows below as a separate dated checkpoint.

## Verification checkpoint — final scoped product measurements (2026-09-18)

The final dated measurement record is
[`docs/perf-checkpoints/2026-09-18-qwen4-shared-token-batched-prefill-final-measurement.md`](../perf-checkpoints/2026-09-18-qwen4-shared-token-batched-prefill-final-measurement.md).
It is historical, fixture-bound evidence only.  It records six fresh
compiler-enabled product CLI samples on `gfx1151` with q8 contiguous KV,
speculation off, noslots/stateless workload, `max_tokens=16`, ten warmups, and
one measured run per fresh private home:

- 42-token short prompt, three raw samples: prefill median `18.5 tok/s`,
  decode median `8.4 tok/s`, TTFT median `2273.3 ms`.  The CLI explicitly
  labels this prompt's prefill number launch-overhead-only.
- 291-token prompt, three raw samples: prefill median `5.5 tok/s`, decode
  median `1.8 tok/s`, TTFT median `53160.3 ms`.  These are observations under
  unknown external GPU contention, not an A/B or speedup claim.

The separate serve smoke returned no completed visible generation
(`ctx=0`, `gen=0`, `finish=null`) and is excluded from those medians.  The
rocprof route is blocked at startup/compiler probing: the daemon retry,
corrected direct oracle profile, and tiny rocprof smoke produced no CSV or
runtime markers.  The `HIPFIRE_PROFILE=1` fallback completed a normal short
bench but emitted no usable profile markers.  Consequently this checkpoint
claims no grouped benefit, launch count, kernel count, ISA attribution, or
dynamic-memory result; memory scope is **STATIC only**.  The historical
old-PLE output-equivalence caveat and all raw paths/hashes are in the dated
record.

## Six-lever tuning lineage — fixture-bound (2026-09-19)

The gfx1151 campaign covers six staged tuning records, not only the final HC
rows/grid-Y candidate. Each record freezes a distinct source/cache/product
identity; these are scoped engineering evidence, not six product promotions:

| # | staged lever | immutable evidence and disposition |
|---:|---|---|
| 1 | QSA selection-only parallelism | [`selection-candidate/manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/selection-candidate/manifest.json) — `parallel-qsa-select-only`, frozen before the attention edit; no standalone product claim. |
| 2 | QSA selection-plus-attention parallelism | [`attention-candidate/manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/attention-candidate/manifest.json) — `parallel-qsa-select-and-attention`, immutable pre-sentinel evidence; no promotion claim. |
| 3 | QSA selection sentinel | [`selection-sentinel/manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/selection-sentinel/manifest.json) — accepted pre-GDN route with exact oracle/profile comparisons; not a product baseline. |
| 4 | GDN shared exact-128 BF16 QK norm | [`gdn-shared-norm128/manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/gdn-shared-norm128/manifest.json) — accepted pre-N8 route with exact oracle/profile comparisons; not a product baseline. |
| 5 | gfx1151 N8 measured-prefill multirow allowlist | [`n8-gfx1151/manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/n8-gfx1151/manifest.json) — accepted kernel optimization and frozen A comparison; no product-win claim. |
| 6 | HC rows/grid-Y batching | [`hc-rows-gridy-20260919-v3/manifest.json`](../../.codeinsight+research/qwen4/perf-targets-20260918/baseline/hc-rows-gridy-20260919-v3/manifest.json) — current final candidate; no product-target claim. |

The first two rows are deliberately separate QSA source/cache stages, while
the selection sentinel is the accepted combined route that gates the GDN
stage. The six records therefore preserve the full tuning lineage without
reinterpreting any earlier historical checkpoint.

## Verification checkpoint — HC rows/grid-Y final candidate (2026-09-19)

The immutable dated record is
[`docs/perf-checkpoints/2026-09-19-qwen4-hc-rows-gridy-final-measurement.md`](../perf-checkpoints/2026-09-19-qwen4-hc-rows-gridy-final-measurement.md).
It supersedes no prior historical record and makes no product target or
promotion claim.

The explicit ROCTX natural-marker HC counts are **112,326** for frozen N8
(56,163 norm + 28,227 read + 27,936 write) and **1,158** for the current HC
candidate (579 + 291 + 288). The candidate kernel trace confirms grid-Y=1 for
scalar and batched continuation, and grid-Y=128/35 for the natural 128/128/35
chunks. The withdrawn heuristic `109024` split is not used.

The focused primitive probe is exact for rows 1, 2, 35, and 128; the fresh
greedy-16 scalar-vs-natural oracle and cross-version N8/original comparisons pass
with zero step mismatches. A separate deterministic Paris correctness fixture
passes ordinary AR and active native MTP independently: both return `Paris`,
finish normally, emit content, see `done`, and have no runaway or stream error;
MTP reports `tau=1.0`, one cycle, and `mtp=true`. This fixture is correctness
evidence only, not a performance observation.

The preflight-gated fresh product ABBAAB used rebuilt candidate binaries:
prefill medians are A `26.0` vs B `28.8` tok/s, decode medians A `8.8` vs B
`8.9` tok/s, and the B/A decode ratio is `1.011364` (not a meaningful decode
win). User thresholds `prefill >500` and `decode >=25` are both unmet and
remain open/blocked; further uncontended profiling is required. The record
does not assign the gap to contention or promise that an idle GPU would solve
it. The stale-binary ABBAAB is preserved as invalid.

Fresh release build, workspace library tests, workspace all-target clippy, and
scoped Rust formatting checks all completed with exit 0; raw host-gate logs
and all GPU evidence are retained at the paths in the dated record. Physical
EP2/EP4, retained replay/PM4 admission, and product promotion remain outside
this checkpoint.
