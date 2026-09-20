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

## G3 — open decision: QSA geometry, shared memory, and position fields

**Status: analysis complete, decision not taken.** No code changed. This section
is the decision record; the weights below are source-derived for the admitted
geometry and *unmeasured* for cost — the measurement plan is at the end.

### The problem, precisely

The Qwen4 QSA step lowers ~13 launches per full-attention layer. Retained replay
requires a fixed launch sequence (no conditional presence), a fixed symbol, a
fixed block and shared-memory size, a grid that is fixed or only narrows from a
recorded maximum, and every position-derived field either absent or declared.
Four properties of the QSA step violate that, and one of them fails *silently*:

**P1 — the pool launch appears and grows.** `indexed_attention_pool_rope_f32` is
launched only when `block_count > 0` (`layer_ops.rs`), and its `grid.x` is that
position-derived count. The kernel itself is mask-safe (`if (block >= block_count
|| compress <= 0) return;` precedes every read, `kernels/src/tensor_ops.hip`), so
an oversized grid is legal; the wrapper rejects `block_count == 0`
(`crates/rdna-compute/src/tensor_ops.rs`). The count is
`complete = (position + rows) / compress`, monotone non-decreasing, so the
condition flips exactly once, at `complete == 1`.

**P2 — dynamic shared memory varies with position.** Both remaining QSA shapes
size their LDS from position-derived lengths: `indexed_attention_select_*` uses
`block_count * 4` bytes, `indexed_attention_attention_*` uses
`max_selected * 8` where `max_selected = min(end_position, capacity, budget *
compress + compress - 1)`. Each wrapper picks between a batched (dynamic LDS) and
a `_serial` (shared memory 0) symbol when the request exceeds a 64 KiB limit.
REDLINE §4 makes symbol + grid + block + shared memory one identity contract and
requires a "replay-stable fixed/tiled design" for a changing shared-memory shape,
so the *value* must become constant even though the symbol happens not to flip.

**P3 — one position-derived device pointer.** `raw_batch = view(raw_index_keys,
initial_position * index_kv_width, …)` (`layer_ops.rs`) bakes a position-shifted
address into the `copy_rows_strided_f32` destination pointer. Pointers are not
scalars; no binding kind covers an address that moves with position.

**P4 — every position-derived field must be *declared*, because the automatic
route has no calibration pass.** `synthesize_position_bindings` (which
differences two recordings and classifies what changed) is called only from the
manual/speculative path; the automatic MQ4R route goes `Captured → Ready` on a
single recording (`docs/REDLINE.md` §3). There is therefore no mechanism that
notices a stale position-derived kernarg field. A tape whose QSA launches carry
undeclared `position_start` / `block_count` / shifted pointers would replay the
capture-position values: **wrong output, no error, no fallback**. G2's declared
bindings are the mechanism that makes this class explicit, and they are mandatory
here rather than an optimization.

### Admitted geometry (source-derived)

| Quantity | Value | Source |
|---|---|---|
| `max_seq` | exactly 2048 (admission requires it) | `crates/hipfire-loader/src/admission.rs` |
| `compress` / `budget` | 4 / 2048 | `crates/hipfire-arch-qwen4/src/config.rs` |
| indexer heads / kv heads / index_dim | 4 / 1 / 128 | same |
| main heads / kv heads / head_dim | 24 / 2 / 256 | same |
| `qsa_selected_capacity` | `budget + compress - 1 = 2051` | same |
| `pooled_capacity` | `ceil(max_seq / compress) = 512` | `crates/hipfire-arch-qwen4/src/state.rs` |
| pool `grid.x` | `complete`, 1…512 | wrapper + config |
| select LDS | `4 * complete` ≤ 2048 B | wrapper |
| attention LDS | `8 * max_selected` ≤ 16 408 B | wrapper |
| attention grid | `[24, 1, 1]` (24 workgroups) | `n_heads=24`, `head_dim=256`, block 256 |

Consequences that shape the decision: **neither 64 KiB switch flips anywhere in
the admitted range** (attention needs `max_selected > 8192`); for `complete ≥ 1`
(positions ≥ 3) the symbol set is already constant (batched everywhere); and both
dynamic-LDS requests are small enough that a capacity-sized reservation fits the
64 KiB device limit with room to spare. The attention grid is 24 workgroups on a
40-CU device, so LDS reservation cannot become an occupancy limiter there.

### Options

**P1, pool launch presence and grid**

- **A1 — capacity-fixed grid (recommended).** `grid.x = pooled_capacity` (512),
  `block_count` passed as a declared `PositionDivU32 { addend: 1, divisor:
  compress }`. The kernel masks, so active work is unchanged; the cost is one
  compare-and-return per inactive workgroup (≤512 per layer per token, only until
  `complete` saturates). Pros: no new mechanism, no env dependency, no kernel
  change, one tape for every position. Cons: bounded wasted work (~0.5–1 % of a
  token by workgroup-count arithmetic, unmeasured), and the grid no longer
  encodes the active count (readability: the binding and the scalar must stay
  consistent).
- **A2 — dynamic grid narrowing.** Keep `grid.x = complete` at capture and
  declare `ReplayGridBinding::PositionCeilDiv { axis: 0, addend: 1, divisor:
  compress }` (`ceil((p+1)/4) == complete` for `rows == 1`, verified identity),
  prepared at a declared maximum position (`set_prepared_max_position`, with the
  existing `position > prepared_max_position` refusal). Pros: exact grid, zero
  wasted work. Cons: needs `HIPFIRE_REPLAY_PM4_DYNAMIC_GRID` (off by default) to
  even record the binding, forces single-queue PM4, and the grid binding is
  PM4-only, so the recorded-HIP oracle launches a different grid than the PM4
  route (numerically identical because the kernel masks, but it is one more
  difference to explain in the parity ledger).
- **A3 — kernel change: capacity tiling.** Rejected: same result as A1, but pays
  a kernel/ABI re-certification for no functional gain.
- **A4 — keep the conditional by arming later (orthogonal, recommended).** Require
  the retained route to arm only once `complete > 0` (position ≥ 3). The
  condition is monotone, so the tape captured after that point stays valid
  forever, and the first ≤3 decode steps run on HIP. Removes the presence
  problem *and* the zero-count select edge case in one move, with no wrapper or
  kernel change. Alternative D2 below if a uniform-from-position-0 tape is
  preferred.

**P2, dynamic shared memory**

- **B1 — capacity-pinned LDS (recommended).** Size the reservation from the
  declaration instead of the active value: select `= pooled_capacity * 4` (2048 B),
  attention `= qsa_selected_capacity * 8` (16 408 B), with the symbol chosen from
  the pinned shape and the active lengths left as declared scalars. Both kernels
  index their dynamic LDS by the active counts, so a larger reservation is not
  read. Pros: constant symbol/block/shared-memory (the whole P2 contract becomes
  position-free), arithmetic untouched (LDS size never changes which values are
  computed, only where they are staged), free at these grid sizes. Cons: it is an
  *assumption about the kernels* — "reservation ≥ active need is safe" must be
  pinned by a test per variant, and the capacity arithmetic must stay under 64 KiB
  if `max_seq` ever grows.
- **B2 — force the `_serial` variants.** Pros: shared memory 0 everywhere, the
  simplest possible contract. Cons: disqualifying for select — `_serial` launches
  with block `[1,1,1]`, i.e. one thread per row scanning every candidate block;
  for attention `_serial` recomputes the dot per pass (bit-identical, but more
  work). Only viable for attention, and only if B1's assumption fails.
- **B3 — patch `shared_mem` at replay.** Rejected: REDLINE §4 forbids a changing
  shared-memory assumption outright; the PM4 packet field is patchable but the
  semantics are not admissible.
- **B4 — kernel change: explicit `lds_capacity` argument.** Fallback if B1's
  premise is falsified. Pays a kernel re-certification; no benefit over B1 while
  the premise holds.

**P3/P4, position-derived fields**

- **C1 — declare scalars; replace the shifted pointer with the existing offset
  parameter (recommended).** `copy_rows_strided_f32` already takes `dst_col_offset`
  as an `i32` kernarg (offset 32 in its blob). Pass the *base* `raw_index_keys`,
  `dst_row_stride = index_kv_width`, and `dst_col_offset = position *
  index_kv_width` — the row mapping `dst[r * index_kv_width + position *
  index_kv_width + c]` is identical for decode and for a `rows > 1` chunk — and
  declare that slot with a new sibling `PositionMulU32 { offset, factor }`.
  Declare the remaining scalars with the vocabulary G2 landed: `position_start`
  as `PositionPlusU32 { addend: 0 }` on the norm/RoPE, cache-append, select and
  attention launches, `block_count` as `PositionDivU32 { addend: 1, divisor:
  compress }` on pool and select. Pros: no kernel change, no new pointer class,
  uniform for decode and chunked prefill, and it removes the silent-staleness
  hazard for exactly the fields that carry it. Cons: ~9 declared bindings per QSA
  layer (host-side 4-byte patches between replays — cheap, but it is per-layer
  bookkeeping to keep honest), plus `PositionMulU32` is a third arithmetic form in
  the vocabulary, and the params-shaped funnel needs a bindings-aware entry
  (or the copy converts to the blob entry).
- **C2 — keep the shifted view, add a pointer binding.** Rejected: an 8-byte
  address patch needs the capture position and the base-address relationship
  inside the tape contract; strictly more machinery than C1 for the same result.
- **C3 — write raw index keys with the existing cache-append kernel** (pass the
  same tensor as key and value): no new binding kind, but it writes the same 128
  values twice and misuses an append contract for a keys-only cache. Viable,
  less honest than C1.
- **C4 — move position into a device buffer** (kernels read position from memory;
  the pattern the other MQ4R models use for the position-buffer H2D). Pros:
  removes position scalars from the tape entirely. Cons: 4–5 kernel signature
  changes plus re-certification, and it does not address P1 or P2 at all. Keep as
  a fallback if the declared-scalar count becomes unwieldy, not as the first move.
- **C5 — rely on recording differencing (do nothing).** Rejected outright: the
  automatic route never calls `synthesize_position_bindings` (P4), so this is a
  silent-wrongness option, not a cheap one.

### Weighing

| | Mechanism cost | Device cost | Kernel/ABI risk | Failure mode if wrong |
|---|---|---|---|---|
| A1 + B1 + C1 + A4 | lowering only | ≤512 masked workgroups/layer/token; LDS reservations free at 24-worker grids | none | loud: binding/owner mistakes fail at prepare |
| A2 + B1 + C1 + A4 | lowering + env flag + prepared max | none | none | clamp at `prepared_max_position` refuses (fail closed) |
| A1 + B2(attention) + C1 + A4 | lowering only | attention dot recomputed twice | none | loud |
| A1 + B4 + C1 + A4 | lowering + kernel | unknown | kernel re-certification | kernel change invalidates the base tape |

The recommended package is **A1 + B1 + C1 + A4**: it is the only column with no
kernel change, no environment dependency, and no silent failure mode. Its total
device cost is bounded by ~512 masked workgroups per QSA layer per token plus two
constant LDS reservations, and both are unmeasured — which is what the next step
must fix.

### Measurement plan (before any code)

1. **Bit-exactness of pinning.** On the ordinary HIP path, A/B the pinned shapes
   against today's shapes on the same prompt: outputs must be bit-identical
   (pinning changes only launch geometry, never arithmetic). Any difference means
   a premise is wrong (most likely B1's LDS-reservation assumption).
2. **Cost of A1's masked grid and B1's reservations.** Decode tok/s over a fixed
   prompt at several context lengths, ≥3 fresh processes, prompt md5 + binary
   md5 recorded. Expectation is within noise; anything above noise makes A2 the
   pool choice.
3. **Zero-count behaviour, only if D2 is preferred over A4:** prove the batched
   select variant at `block_count == 0` bit-identical to the `_serial` variant
   that the wrapper would pick today, or keep A4 and never launch it.
4. **Binding census.** With a capture window open, assert every position-derived
   kernarg in the QSA step is either declared or provably position-free — the
   concrete form of REDLINE §7 gate 3 for this route.

Nothing above requires G4, but the *end-to-end* form of 1–4 does, because today
no Qwen4 forward can complete with a replay backend enabled.

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
- 2026-09-20 — **G1 landed** (branch-implemented) in `b12185138`:
  `Gpu::launch_blob_recorded` + 31 `tensor_ops.rs` migrations + one shared
  `recorded_launch_artifact` resolver (funnel, scratch, new entry).
  Evidence: GPU tape test, `cargo test -p rdna-compute --lib` 260 pass,
  `cargo test -p hipfire-dispatch` 285 pass, HIP end-to-end `PARIS`.
- 2026-09-20 — **G2 landed** (branch-implemented) in `b12185138`:
  `PositionDivU32`/`PositionModU32` bindings, declared-binding carriage on
  `RecordedHipLaunch`, prepare-time merge with one-owner-per-slot +
  position-derived-only rules, synthesis skip for declared offsets, binding
  identity in the sequence hash, `offset()` accessor replacing three duplicated
  match arms, and the first two declarations (GDN conv ring cursor, batched chunk
  start cursor) with `GatedDeltaConv::row_index` supplied by `layer_ops`.
  Evidence: 6 new unit tests (87 `replay::` tests pass).
- 2026-09-20 — Default-path probe re-run after G1/G2: still the sealed-MoE
  preflight refusal, i.e. G4 remains the gate for any capture. Unchanged behavior
  is the expected result here, not a regression.
- 2026-09-20 — **G3 analyzed, no decision taken** (this section): problem split
  into P1 pool presence/grid, P2 dynamic shared memory, P3 a position-shifted
  destination pointer in the index-key write, and P4 the finding that the
  automatic route has no calibration pass, so an undeclared position-derived field
  is a *silent* wrong-output hazard rather than a loud failure. Admitted geometry
  computed (compress 4, budget 2048, pooled_capacity 512, qsa_selected_capacity
  2051, attention grid 24 workgroups, LDS ≤ 16 408 B); neither 64 KiB switch flips
  in the 2048-token range. Options A1/A2, B1–B4 and C1–C5 recorded with
  pro/contra and a measurement plan; recommendation A1 + B1 + C1 + A4.

### Next

1. G3 measurement plan (bit-exactness of pinning, cost of the masked pool grid and
   the pinned LDS reservations, binding census) — none of it needs G4, but the
   end-to-end form does.
2. G4: until the specialized-route pointer contract exists (or the refusal is
   narrowed to the exactly-unstable sub-case), no complete Qwen4 tape can be
   captured, so no census, parity, or route proof is reachable.
3. Then the REDLINE §7 ladder, notably the state oracle (QSA full/raw/pooled
   keys, selected indices, GDN recurrent/conv state, PLE history).
