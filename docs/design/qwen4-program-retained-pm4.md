# Retained PM4 replay for the Qwen4 declarative program

## Status and purpose

**Status: G1 and G2 implemented on `feat/qwen38-flash-next` (branch-implemented,
not certified).** Two remaining gaps — G3 (QSA geometry/shared-memory shape) and
G4 (sealed MoE pointer contract) — are **open decisions** and are not started.

This record is the tracking document for enabling Redline retained PM4 replay for
Qwen3.8 Flash-Next (`hipfire-arch-qwen4`) **at the shared engine/dispatch level**,
without adding PM4-specific code to the architecture crate. `hipfire-arch-qwen4`
still contains zero Redline references after G1/G2.

Follow [the design-record lifecycle](README.md): this file is intent plus a
progress ledger, not a claim of implementation, admission, or speed. Runtime
validation authority remains [`../VALIDATION.md`](../VALIDATION.md); the retained
replay contract is [`../REDLINE.md`](../REDLINE.md); performance protocol is
[`../methodology/perf-benchmarking.md`](../methodology/perf-benchmarking.md).
G3/G4 decisions get their own linked record; do not rewrite this intent once
decided.

## Objective

Make one Qwen4 single-token decode program (the `Step` list built in
`crates/hipfire-arch-qwen4/src/gpu_forward.rs` and executed by
`hipfire-dispatch::pipeline::steps`) a **capture-complete, lowerable retained
tape** whose dynamic values are declared by the program and patched by the replay
layer.

Non-objectives, explicitly:

- No new Redline/PM4 code in `crates/hipfire-arch-qwen4/src/`. The family keeps
  binding typed descriptors; the engine owns the replay contract.
- No prefill, speculative, MTP, or multi-token retained body. REDLINE §3 scope is
  ordinary sequential single-token AR continuation.
- No promotion claim. Certification is the REDLINE §7 ladder, and §8 governs any
  number quoted.

## Baseline (measured, this box)

Fixture: `~/.hipfire/models/qwen3.8-flash-next.mq4r` (`.mq4r` + gfx1151 + single
GPU ⇒ `retained_redline_default` is true, `crates/hipfire-runtime/src/config.rs`).

```bash
# Reproduced 2026-09-20. Isolated HOME is required: the daemon takes an flock on
# $HOME/.hipfire/daemon.pid, and config.json must carry max_seq=2048 for qwen4.
mkdir -p /tmp/qwen4probe/.hipfire
jq -c '.max_seq=2048 | .port=11499' ~/.hipfire/config.json > /tmp/qwen4probe/.hipfire/config.json
HOME=/tmp/qwen4probe HIPFIRE_LOCAL=1 \
  target/release/hipfire run ~/.hipfire/models/qwen3.8-flash-next.mq4r "say hi" -n 8
```

Observed (default config): the engine's own retained default arms, then the
declarative program refuses at preflight — **before any decode work**:

```text
[redline] enabling fail-closed retained default on gfx1151 (model_arch=qwen4, drafter=off, transport=pm4)
[qwen4] 2560d 48L 248320 vocab
... qwen4_ar forward_chunk prefill failed: ... preflight Qwen4 typed program:
    Hip("sealed_moe: specialized sealed MoE has no retained-replay pointer contract; refusing before launch")
```

Observed with `HIPFIRE_REPLAY_BACKEND=hip` (same command family): the model loads
and generates on the ordinary HIP path — this is REDLINE gate 1 (healthy
baseline) and the A/B arm for later work. The artifact requires `max_seq=2048`.
The only failure seen with a small token budget is a caller-side framing check
(`open think span at end of generation (validation)`) for a reasoning model that
cannot close `<think>` inside the budget; it is not a GPU/dispatch error.

Two facts follow, and they order the plan:

1. The refusal is the sealed-MoE pointer-contract guard
   (`crates/hipfire-dispatch/src/pipeline/sealed_moe.rs`, `specialized_route &&
   gpu.replay.is_enabled()`), not a missing hook. `is_enabled()` is true for every
   non-`Hip` backend *including* manual shadow, so **no complete Qwen4 tape can be
   captured today**, for any model size.
2. The retained path is already the automatic default for this artifact on this
   arch, so the end-to-end census (G1's acceptance) is only observable once G4's
   guard is replaced. G1 and G2 are therefore verified by their own unit and
   structural evidence now, and by the end-to-end census as soon as G4 lands.

## Why this is an engine-level problem

Evidence established from this branch (symbols are anchors, re-read before
editing):

| Fact | Evidence |
|---|---|
| One recorder entry for the whole engine | `Gpu::launch_maybe_blob_bound` → `ReplayController::record_hip_launch_typed_bound` (`crates/rdna-compute/src/dispatch.rs`, `crates/rdna-compute/src/replay.rs`) |
| PM4 lowering is engine-owned and arch-selected, not arch-authored | `Pm4Architecture::from_name` (`replay.rs`); gfx11 family admitted, gfx1151 among them |
| Step lowering is dispatch-owned | `crates/hipfire-dispatch/src/pipeline/layer_ops.rs` |
| Positions are program data, not architecture logic | `GatedDeltaNetOp.start_position`, `IndexedAttentionState.position` (`layer_ops.rs`) |
| The AR loop is shared engine code | `generate_ar_with_forward` (`crates/hipfire-generate/src/ar.rs`), fed by two closures in `crates/hipfire-generate/src/qwen.rs` |
| The dispatch layer already gates on replay state | `gpu.replay.is_enabled()` guard (`sealed_moe.rs`) |

The one thing the architecture crate cannot supply from outside is *semantics of
its own scalars* (which kernarg word is a function of position, under which
formula). That is exactly what G2 makes declarable, and it is declared at the
*lowering* owner where the value is computed — still not the arch crate.

## Gap inventory

| Gap | What | Owner | Status |
|---|---|---|---|
| G1 | Half the decode program never reaches the recorder: `tensor_ops.rs` was 31 wrappers, 31 raw `Gpu::launch_kernel_blob`, 0 funnel launches | `crates/rdna-compute` | **done** (branch-implemented) |
| G2 | Binding vocabulary could not express a quotient/modulo of position, and declared bindings could not be attached to a launch | `crates/rdna-compute`, `hipfire-dispatch` | **done** (branch-implemented) |
| G3 | Position-switched kernel symbol and dynamic shared memory in QSA (`complete > 0` conditional launch, grid growing with position) | engine (lowering) + a kernel contract decision | open decision |
| G4 | Sealed MoE "no retained-replay pointer contract" refusal: `route_policy` is `Some(Qt44Qt53Grouped)` for qwen4 decode | engine (dispatch) | open decision |

## G1 — recorder funnel coverage (done)

**Problem.** `crates/rdna-compute/src/tensor_ops.rs` owns every Qwen4 trunk state
op (GDN step/conv/params/gate, HC hyper read/write/norm/activation, all QSA ops,
bf16 roundtrip, argmax). Every one of the 31 wrappers launched through the raw
`Gpu::launch_kernel_blob`, which returns before the recorder is consulted. A
capture that "succeeds" therefore contains a truncated tape; REDLINE §7 gate 2
rejects it, and the `capture_blobs.len() == recorded_launches().len()` parity
invariant does not catch it for qwen4 because qwen4 has no HipGraph path.

**Landed.** `Gpu::launch_blob_recorded` is the blob-shaped twin of
`Gpu::launch_maybe_blob`: same recorder entry, same exact-byte capture, same
HipGraph capture-blob accounting, same `last_kernel` attribution, and the same
`Result` shape. `Gpu::launch_kernel_blob` remains the raw entry used by the
funnel itself and by the recorded-HIP oracle, which must not re-record. All 31
`tensor_ops.rs` sites migrated; the duplicated artifact-alias table in the funnel
and in `scratch.rs` collapsed into one `recorded_launch_artifact` resolver, which
the new entry uses too.

**Evidence.**

- `tensor_ops.rs` contains zero `gpu.launch_kernel_blob(` calls.
- GPU-backed test `tensor_ops::tests::recorded_blob_launch_enters_the_tape_with_the_bytes_it_launched`
  (gfx1151): with no recording window the launch leaves no tape entry; inside one
  it records exactly one entry whose kernel, grid, resolved artifact and kernarg
  bytes (`ptr`, `elements`, `scale`) match what was launched, while the kernel
  still executes (readback equals the expected product). The test fails against
  the pre-G1 raw entry.
- `cargo test -p rdna-compute --lib` (260) and `cargo test -p hipfire-dispatch`
  (285 + 1 ignored) pass; the QSA/HC GPU tests in the module exercise migrated
  sites.
- Ordinary-HIP end-to-end probe (below) still runs the model.

**Not yet evidenced.** The end-to-end census (recorded launches == compute
launches for one decode forward) needs G4, because the sealed-MoE guard refuses
before any decode forward completes.

## G2 — declared dynamic bindings (done)

**Problem.** The vocabulary was grid `PositionCeilDiv` (rounds up, narrow-only)
and kernarg `PositionPlusU32` / `GdnFrameU32`. Qwen4 already computes, in dispatch
lowering, scalars the vocabulary cannot name:

- GDN conv `cursor = (start_position + row) % history_rows` (`layer_ops.rs`);
- QSA `block_count = final_position / compress`, `budget_blocks = budget / compress`
  (`layer_ops.rs`).

Today such a difference is discovered only by differencing two recordings
(`synthesize_position_bindings`), which fails closed on anything non-affine.
REDLINE §4 requires dynamic fields to be *named* with an owner; the fix is a
declared binding, not a smarter heuristic.

**Landed.**

1. `ReplayKernargBinding::PositionDivU32 { offset, addend, divisor }` and
   `PositionModU32 { offset, addend, modulus }`, applied through the single shared
   helper `apply_kernarg_bindings_for_dispatch`, so PM4 and the recorded-HIP
   oracle cannot diverge. Zero divisor/modulus, out-of-range offset, and u32
   overflow are explicit errors; all four kinds write through one bounds-checked
   `write_kernarg_u32`.
2. A declared set travels with the launch into `RecordedHipLaunch` and is merged
   at prepare by `merge_declared_kernarg_bindings`, which fails closed on a second
   owner for one `(dispatch, offset)` and rejects a non-position-derived
   declaration (the GDN frame counter is owned by the replay layer, which derives
   it from the recorded launch).
3. `synthesize_position_bindings` skips declared offsets: a named field is not an
   unexplained difference.
4. Declared bindings are part of tape identity (`replay_sequence_hash`), and the
   three duplicated offset `match` arms collapsed into `ReplayKernargBinding::offset()`.
5. First consumers, both at the lowering that computes the value:
   `gated_delta_conv` (ring cursor, `addend = row_index`) and
   `gated_delta_conv_batched` (chunk `start_cursor`, `addend = 0`). The new
   `GatedDeltaConv::row_index` datum is supplied by `layer_ops`, so the
   declaration is exact for a single-token decode and for a multi-row chunk
   alike. Kernel width 1 declares nothing (no ring to index).
6. QSA quotient declarations are deliberately **not** made; they belong to G3.

**Evidence.** Six new unit tests in `crates/rdna-compute/src/replay.rs`
(`position_div_binding_rederives_the_host_quotient`,
`position_mod_binding_rederives_the_ring_cursor`,
`declared_binding_is_not_an_unexplained_kernarg_difference`,
`declared_binding_collision_fails_closed`,
`a_declared_frame_counter_is_not_a_position_binding`,
`declared_bindings_are_part_of_tape_identity`); 87 `replay::` tests pass.

**Not yet evidenced.** Nothing here has been through PM4 preparation, because a
Qwen4 tape cannot yet be captured or prepared (G3/G4).

## G3 — open decision: QSA geometry and shared memory

Not started. The decision is *how* to make these replay-stable, not whether:

- `indexed_attention_pool_rope` is launched only when `complete > 0`, and its grid
  x is `complete = final_position / compress` — a launch that appears, then grows.
  REDLINE records a flat order-preserving tape, so a conditionally-present
  dispatch has no representation; a recorded grid is a hard maximum that may only
  be narrowed.
- `indexed_attention_select_*` and `indexed_attention_attention_*` choose kernel
  symbol *and* dynamic shared memory from position-derived lengths
  (`block_count`, `max_selected`). REDLINE §4/§7-stage-7 requires a
  "replay-stable fixed/tiled design" for a changing block/shared-memory shape.

Candidate directions to evaluate (choose one, record why):

1. Capacity-bounded geometry: launch over the declared capacity, pass live lengths
   as scalars, keep one symbol and one static shared-memory size. Requires kernel
   contracts to mask correctly and a non-replay perf measurement.
2. Always-serial shape: force the `_serial` variant (shared memory 0) whenever
   replay is enabled. Simplest, likely slowest; must be measured against HIP.
3. Position-regime tapes: prepare a small set of tapes keyed by the position
   predicate. Must justify why this is not "capture-time shape replayed at a new
   context" (REDLINE §10 failure atlas).

Required evidence to close: a tau/tok-s measurement on the ordinary HIP path for
the chosen shape (no regression claim without ≥3 fresh-process runs, prompt md5,
binary md5), plus the pool-kernel `block_count == 0` no-op question answered by
kernel contract or by regime gating.

## G4 — open decision: the sealed MoE pointer contract

Not started. `crates/hipfire-arch-qwen4/src/program.rs` binds
`route_policy: Some(MoeRoutePolicy { capability: Qt44Qt53Grouped })`, and
`sealed_moe.rs` refuses every launch while `gpu.replay.is_enabled()` with
"no retained-replay pointer contract". The refusal is correct for what the route
currently guarantees; it must be replaced by a contract, not deleted.

Decision content to settle and record:

1. Which pointers in the specialized route are host-derived (expert pointer
   tables, bound expert ownership, rank/plan-local views) and which are
   allocation-stable for the plan lifetime.
2. Whether the contract is (a) a validated stable pointer set carried by the
   route policy declaration, or (b) a rejection narrowed to the exact sub-case
   that is unstable, with the stable case admitted.
3. The admissibility rule that replaces the blanket refusal: every
   position/state-dependent scalar is either indirect through a persistent buffer
   or covered by a declared binding (the REDLINE §4 rule), enforced in
   `validate_sealed`, with a negative test per rejected shape.
4. Whether qwen4 decode stays the *specialized* route at all for the retained
   body, or whether the retained body is defined over the generic top-10 path.

Until G4 lands, an explicit `HIPFIRE_REPLAY_BACKEND=hip` run is the healthy
baseline (REDLINE gate 1), and it is also the A/B arm for the later census.

## Verification ladder

| Stage | Route | State |
|---|---|---|
| Unit: G1 funnel entry | GPU test `tensor_ops::tests::recorded_blob_launch_enters_the_tape_with_the_bytes_it_launched` (gfx1151; skips elsewhere) | **passing** |
| Unit: G2 binding vocabulary | 6 tests in `crates/rdna-compute/src/replay.rs` | **passing** |
| Structural: no raw launch left in the program's op owner | `tensor_ops.rs` launch-discipline review (0 raw sites) | **passing** |
| Ordinary HIP baseline | `HIPFIRE_REPLAY_BACKEND=hip` daemon probe: loads, generates (`PARIS`), commits | **passing** |
| End-to-end census: recorded launches == compute launches | needs G4 | blocked by design |
| Multi-position HIP vs recorded-blob vs PM4 parity | needs G3, G4 | blocked by design |
| Serve health, stationary matched performance | REDLINE §7 gates 6–7 | blocked by design |

Harness gap to be resolved with G3/G4 (not now): no dispatch-level or engine-level
capture hook exists, and the arch-coupled bench harness
(`crates/hipfire-generate/src/redline.rs`) downcasts to existing model types. The
qwen4 probe today is the daemon auto-default path (`hipfire run` with an isolated
`HOME` holding `max_seq=2048` and a distinct port, because the daemon takes an
flock on `$HOME/.hipfire/daemon.pid`), which is a diagnostic, not route proof.

## Progress ledger

Append-only. One line per landed change with the commit hash once it exists.

- 2026-09-20 — plan written. Baseline refusal reproduced (see above). G1/G2 in
  progress.
- 2026-09-20 — **G1 landed** (branch-implemented, uncommitted at the time of
  writing): `Gpu::launch_blob_recorded` + 31 `tensor_ops.rs` migrations + one
  shared `recorded_launch_artifact` resolver (funnel, scratch, new entry).
  Evidence: GPU tape test, `cargo test -p rdna-compute --lib` 260 pass,
  `cargo test -p hipfire-dispatch` 285 pass, HIP end-to-end `PARIS`.
- 2026-09-20 — **G2 landed** (branch-implemented): `PositionDivU32`/
  `PositionModU32` bindings, declared-binding carriage on `RecordedHipLaunch`,
  prepare-time merge with one-owner-per-slot + position-derived-only rules,
  synthesis skip for declared offsets, binding identity in the sequence hash,
  `offset()` accessor replacing three duplicated match arms, and the first two
  declarations (GDN conv ring cursor, batched chunk start cursor) with
  `GatedDeltaConv::row_index` supplied by `layer_ops`. Evidence: 6 new unit tests
  (87 `replay::` tests pass).
- 2026-09-20 — Default-path probe re-run after G1/G2: still the sealed-MoE
  preflight refusal, i.e. G4 remains the gate for any capture. Unchanged behavior
  is the expected result here, not a regression.

### Next (do not start without a decision)

1. G4 first: until the specialized-route pointer contract exists (or the refusal
   is narrowed to the exactly-unstable sub-case), no complete Qwen4 tape can be
   captured, so no census, parity, or route proof is reachable.
2. Then G3, with the census in hand: the QSA shapes (conditional pool launch,
   position-growing grid, symbol/shared-memory switching) are the next blocker to
   a lowerable tape, and they need a measured decision.
3. Then the REDLINE §7 ladder, notably the state oracle (QSA full/raw/pooled
   keys, selected indices, GDN recurrent/conv state, PLE history).
