# Retained PM4 replay for the Qwen4 declarative program

## Status and purpose

**Status: G1–G3 implemented and G4's scope fix (A2) landed on
`feat/qwen38-flash-next` (branch-implemented, not certified).** The remaining
decision is G4's admission contract (B1, with C2 for the census): until it lands, a
Qwen4 tape cannot exist, so the census, parity, and route proof stay unreachable.

This record is the tracking document for enabling Redline retained PM4 replay for
Qwen3.8 Flash-Next (`hipfire-arch-qwen4`) **at the shared engine/dispatch level**,
without adding PM4-specific code to the architecture crate. `hipfire-arch-qwen4`
still contains zero Redline references after G1–G3.

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
| G3 | Position-switched kernel symbol and dynamic shared memory in QSA (`complete > 0` conditional launch, grid growing with position) | engine (lowering) + a kernel contract decision | **decided + done** (branch-implemented; route arming deferred to the G4 hook) |
| G4 | Sealed MoE "no retained-replay pointer contract" refusal: `route_policy` is `Some(Qt44Qt53Grouped)` for qwen4 decode | engine (dispatch) | **A2 landed** (scoped refusal + engine-side retained-body policy); B1 + C2 pending decision |

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

## G3 — QSA geometry, shared memory, and position fields

**Status: decided and implemented (branch-implemented, uncertified).** Decision:
capacity-fixed pool grid, capacity-pinned LDS with a constant symbol, and every
position-derived field declared at the launch that computes it. Evidence below;
the alternatives and their weights are kept in this record.

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

### Decision and measurement (2026-09-20)

**Decided: A1 + B1 + C1.** Capacity-fixed pool grid, capacity-pinned LDS with a
constant symbol, position-derived fields declared at the launch that computes
them. A2 (dynamic grid narrowing) was dropped once A1 measured free: it would add
an environment gate, a single-queue restriction, and a PM4-only grid binding that
the recorded-HIP oracle does not share, for no measurable gain. B2 was dropped
because the select `_serial` variant launches one thread per row. C4 stays the
fallback if the declared-field count ever becomes unwieldy.

What the measurement showed (fixture: `qwen3.8-flash-next.mq4r`, md5
`fda74d3760dc803e778e9b30a2fe0ebd`; binary md5s recorded in `/tmp/qsa-shape-cost.log`;
prompt `benchmarks/prompts/qwen4_ar_primes.txt`, md5 `0508eec29a44323f62e70fa77d92b834`;
greedy `-n 512 -t 0`, HIP backend, isolated `HOME` with `max_seq=2048`):

| Arm | runs (tok/s) | stream |
|---|---|---|
| position-derived shapes (before) | 10.9, 11.5, 11.4, 8.1, 11.5 → median 11.4 | `2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37`, 116 tokens, `finish=stop` |
| capacity-pinned shapes | 11.4, 11.4, 11.4, 11.4, 11.4 → median 11.4 | byte-identical to the arm above in all 5 pairs |

Interleaved fresh-process pairs. The pinned arm showed no spread at all in five
samples while the derived arm showed one 8.1 outlier; that variance difference is
recorded as an observation, not a claim (5 samples, unknown cause). The
conservative direction is the measurement itself: both knobs are at their worst at
a short context (the attention LDS reservation is pinned at its maximum while the
live length is small, and the pool grid masks the largest share of workgroups), so
this is the arrangement most likely to expose a cost.

Unit evidence, all on gfx1151:

- `tensor_ops::tests::pinned_qsa_shapes_are_bit_identical_to_derived_shapes` —
  an oversized masked pool grid, a larger select LDS reservation, a larger
  attention LDS reservation, and the batched select symbol at an active count of
  zero (against the serial symbol it replaces) are each bit-identical.
- `tensor_ops::tests::qsa_position_fields_are_declared_to_the_recorder` — with a
  recording window open, each QSA launch reports the declared binding the lowering
  intends, and the recorded bytes *at that offset* equal the value the launch
  used, so a drifted offset fails in the test rather than at replay.
- `test_copy_rows_strided_f32_parity` (7 cases, now including
  `row-absolute/dcol=7*len`) — the relaxed `copy_rows_strided_f32` contract is
  bit-exact against the per-row `copy_d2d` reference.
- `cargo test -p rdna-compute --lib` 263 pass; `cargo test -p hipfire-dispatch`
  285 pass.

**Not evidenced.** No tape has been captured, prepared, or replayed: the shapes
are validated on the ordinary HIP path only, and G4 still refuses any Qwen4
forward with a replay backend enabled. The pool launch also remains absent for the
first `compress-1` positions (`layer_ops` keeps `if complete > 0`), so the route
that eventually arms a tape must arm once `complete > 0` — a monotone, one-time
condition — or the pool kernel must accept a zero count. That arming decision
belongs with the G4 hook and is deliberately not pre-empted here.

**Bound-of-record.** `pooled_capacity = ceil(max_seq / compress)` and
`qsa_selected_capacity = budget + compress - 1` are both derived from the admitted
geometry, so the pinned shapes cover every position the admitted configuration can
reach. A different `max_seq` is a different tape identity, which is correct: it is
a reload with different capacities.

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

### Landing

- `IndexedAttentionPoolRope` takes a declared `grid_bound` (the lowering passes
  `pooled_capacity`) and, when the caller declares a position source, the active
  count as `PositionDivU32 { addend: rows, divisor: compress }`; the wrapper
  verifies the caller's count *equals* the declared formula, so the declaration
  cannot drift from the launch.
- `IndexedAttentionSelectBatch` takes a declared `shape_blocks` (LDS + symbol come
  from the bound, not the active count) and declares both its active count and
  `position_start`. `IndexedAttentionAttentionBatch` takes a declared
  `shape_selected` and declares `position_start`.
- `indexed_attention_norm_rope_batch` and `indexed_attention_cache_append_batch`
  declare `position_start`.
- The index-key write no longer bakes `position * index_kv_width` into a device
  pointer: `copy_rows_strided_f32` receives the base tensor plus a
  `dst_col_offset` scalar (its row-absolute contract is now bounded by the
  destination extent rather than by one row pitch) and declares it with the new
  `ReplayKernargBinding::PositionMulU32`. Kernarg offsets are captured where the
  scalar is written (`args.len() - 4`), never hand-counted.
- The `HIPFIRE_QSA_STABLE_SHAPES` measurement gate is gone: the pinned shape is the
  only production shape, so the HIP and replay paths cannot diverge.

**Remaining for the route (with G4):** arm the tape once `complete > 0`, or teach
the pool wrapper to accept a zero count. Also outstanding: the launch census
(REDLINE §7 gate 3) that asserts every position-derived kernarg in a *recorded*
Qwen4 forward is declared — it needs a capturable forward, i.e. G4.

## G4 — the sealed MoE pointer contract and the refusal's blast radius

**Status: analyzed, no decision taken.** No code changed. Evidence below is
source-derived (two read-only surveys plus direct reading); option weights are
judgements and are marked as such.

### The problem, precisely

`crates/hipfire-dispatch/src/pipeline/sealed_moe.rs:1829-1837`:

```rust
let specialized_route = match &self.params {
    SealedParams::Decode(params) => params.route_policy.is_some(),
    SealedParams::Prefill(params) => params.route_policy.is_some(),
};
if specialized_route && gpu.replay.is_enabled() {
    return Err(invalid(
        "specialized sealed MoE has no retained-replay pointer contract; refusing before launch",
    ));
}
```

Qwen4 binds that policy unconditionally (`crates/hipfire-arch-qwen4/src/program.rs`
decode and prefill, plus the legacy `execute_moe`), `is_enabled()` is
`request != Hip && state != Fallback`, and the guard sits in `validate_for_gpu`,
which runs at **preflight** — before any launch, for **every** forward, including
prefill.

Two distinct defects are tangled in that one predicate:

**D1 — the refusal's blast radius exceeds the retained contract.** The tape never
exists for prefill (REDLINE §3 requires prefill to stay outside it), and
`Fallback`/`Hip` forwards would run HIP anyway. Refusing them means a `.mq4r`
Qwen4 artifact on gfx1151 (where `retained_redline_default` arms automatically)
**cannot serve at all**: no prefill, no generation, no fallback. The reproduced
probe in the Baseline section is exactly this. The engine's own fallback semantics
(REDLINE §3: a poisoned route falls back to HIP; an ineligible forward never
records or replays) describe the correct scope, and the current guard does not
implement it. There is no Qwen4 arming/poison hook at all
(`crates/hipfire-arch-qwen4` contains no replay reference), so even a scoped guard
would leave the capture window erroring per token instead of degrading to HIP.

**D2 — the route has no *stated* contract, so the engine refuses instead of
validating.** The survey says the route is already close to conformant:

| Fact | Evidence |
|---|---|
| Expert pointer tables are built **once at load** from load-time device addresses and uploaded once; nothing rewrites them (no Qwen4 weight pager) | `gpu_forward.rs:629-658`, freed only in `free_gpu:694-706` |
| Every kernarg pointer is a model/`Gpu`-lifetime tensor, a pure address-arithmetic slice view, or geometry — no per-forward host-built table, no per-forward H2D inside the sealed call | `mod.rs:393-405`, `dispatch.rs:317-325`, `gpu_forward.rs:643/654` are the only `memcpy_htod_auto` on the path |
| **No position-derived scalar exists anywhere in this route** — every non-pointer kernarg is m/k/batch/top-k/n_exp | `gemv.rs:11884-11892`, `moe.rs:1797-1810` |
| Pointer identity is already re-proved **every seal** against identities frozen at a one-shot bind | `validate_live_binding` (`sealed_moe.rs:3957-4015`), `bind_live` one-shot (`:1148-1172`), `build_live_binding` (`:3476-3545`) |
| A per-expert pointer **mapping fingerprint** already exists | `mapping_fingerprint`, `sealed_moe.rs:3523-3545` |
| Single-rank, single-device only; EP/root-routed/gather paths are never entered | `:1759-1764`, `:1822`, `:1777-1791` |
| The generic path is **not** an alternative: `MQ4G128V2` down requires the policy, and the k=10 generic path is CPU-host-routed and separately refused under capture | `sealed_moe.rs:2511-2516`, `families/moe.rs:493`, `moe_program.rs:1131-1134`, `sealed_moe.rs:1746-1753` |

What is genuinely missing is therefore small and specific:

1. **Table *contents* are not proved on the Single path.** `validate_pointer_table`
   checks dtype and byte capacity only; the *compact* path additionally proves each
   owned entry points at its own local tensor (`sealed_moe.rs:3660-3680`), the
   Single path does not. A stale entry would be dereferenced by the kernel while
   the live-expert check looks satisfied.
2. **The mapping fingerprint is not part of any plan identity.** It is
   dispatch-private, so a retained plan cannot pin "the pointer mapping this tape
   was captured against" and re-prove it at prepare/replay.
3. **Lifetime hazards outside the route's own checks are unnamed**: `gpu.scratch`
   FWHT sign tables are lazily allocated on first use (`gemv.rs:3437-3443`,
   `3560-3562`, `scratch.rs:366-420`) — if first touched *inside* a recorded body
   the tape holds an address nobody promised to keep, and a scratch rebuild while
   a plan lives would dangle it; `HIPFIRE_DUMP_HIDDEN` performs a D2H inside the
   route; `invalidate_for_kv_mode_switch` and model reset/swap must invalidate the
   plan (the switch path already poisons, `dispatch.rs:4253-4255`).
4. **No census exists.** Launch count and geometry stability across positions are
   unverified (REDLINE gate 2), and the census needs a capturable forward — which
   the refusal prevents, while the census is one of the things that would justify
   admission.

### Options

**D1 scope — how wide should the refusal be?**

- **A1 — keep the blanket guard.** Rejected: it makes the model unservable, blocks
  the prefill evidence as well, contradicts REDLINE's fallback semantics, and is
  the one behavior in this area with a product-visible regression. Cost if kept:
  every `.mq4r` Qwen4 forward fails on gfx1151.
- **A2 — scope to the eligible retained body (recommended, independently).**
  Refuse only when the controller would record or route
  (`is_recording() || should_route_pm4() || should_route_aql()`); prefill, `Hip`,
  and `Fallback` forwards launch normally, and a failure inside the capture window
  poisons the controller (sticky fallback) instead of erroring each request. Needs
  the Qwen4 arming/poison/fallback hook — which is also where G3's "arm once
  `complete > 0`" condition lives, so the two remaining items share one hook.
  Pros: restores serving and the prefill evidence; keeps the tape fail-closed
  (capture still refused until D2 is closed); small, testable change (a `.mq4r`
  serve with Redline armed must reach HIP and log a fallback reason). Cons: no
  tape yet; adds a hook whose correctness (poison-on-capture-failure) matters and
  must be tested.
- **A3 — A2 plus admission (D2).** The actual goal.

**D2 contract — how to obtain an admissible route?**

- **B1 — validate-and-admit by extending the existing machinery (recommended).**
  (i) Prove table contents on the Single path (mirror the compact path's entry
  check); (ii) expose the mapping fingerprint as plan identity and require
  capture/prepare/replay to observe the same fingerprint; (iii) name the lifetime
  rules: sign tables resident before the capture window, no scratch rebuild while a
  plan lives, `HIPFIRE_DUMP_HIDDEN` refused while recording, existing invalidation
  paths wired to the plan; (iv) replace the blanket guard with these checks plus
  *named* refusals for the sub-cases that stay out (paged residency, EP>1,
  host-routed fallback); (v) census as gate-2 evidence, with a negative test per
  refused shape. Pros: the route already satisfies the substance per the inventory;
  the checks are cheap and two of them already exist in the compact path; no kernel
  or lowering change; the refusal becomes narrow and self-explaining. Cons: the
  proof burden lands here; the census needs a capturable forward, which needs
  either this admission to be complete or the diagnostic route below.
- **B2 — make stability structural.** Absolute (base-relative) expert addressing or
  a device-side indirection so that residency/placement changes never touch a
  recorded kernarg, plus load-time sign tables. Pros: the contract becomes trivial
  and survives a future weight pager. Cons: kernel/dispatch work for hazards
  nothing currently exercises (no pager, single rank, table written once); the
  cheap half (sign tables resident before capture) belongs in B1 regardless.
  Defer.
- **B3 — run the retained body over the generic route.** Rejected on evidence:
  `MQ4G128V2` down is refused without an architecture-declared policy, and the k=10
  generic path is host-routed (CPU top-K with readback) and separately
  capture-refused. There is no second route to fall back to.
- **B4 — do not admit; keep the route HIP-only.** Honest and cheap. Two spellings:
  (a) add a Qwen4 carve-out to `retained_redline_default` — the Muse Glimmer
  precedent, whose comment says automatic admission is withheld "until that
  lowering lands", so a carve-out would be policy-consistent; and/or (b) rely on
  A2's scoping. Pros: no risk, removes the brick, keeps the product honest about
  an unadmitted route. Cons: G1–G3 stay latent, the census stays unreachable, and a
  carve-out would be a policy decision taken *without* the measurement that would
  justify or refute it. Fallback position, not a first move.

**C — evidence sequencing.**

- **C1 — contract first, then census.** The REDLINE-correct order, but the census
  is the evidence that tells us whether the contract's assumptions hold (launch-set
  and geometry stability across positions), so it is partly circular.
- **C2 — diagnostic capture first, then contract (recommended with B1).** Add an
  explicit, non-default diagnostic that lets a capture proceed with the specialized
  route for *measurement only*: it must not install a plan, must not serve, and
  must never count as route proof — the same posture `HIPFIRE_REPLAY_MANUAL_CAPTURE`
  already has. This yields the launch census and the shape-stability answer (G3's
  last outstanding item) before the admission rule is frozen. Pros: breaks the
  circularity with an existing precedent; cheap. Cons: it is a bypass of a
  fail-closed guard, so its scope, naming, and evidence class must be explicit in
  the code and in this record.

### Weighing

| Package | Cost | Risk of silent wrongness | Evidence produced | Verdict |
|---|---|---|---|---|
| A1 | none | none (refuses) | none | reject — model unservable |
| A2 | low (scope + poison + Qwen4 hook) | low (fallback path must be tested) | serving restored; prefill evidence | **do first**, independently of D2 |
| A2 + B1 + C2 | medium (contents proof, fingerprint identity, census, hook, negative tests) | low — every new check fails closed | census, mapping identity, named refusals, then the REDLINE ladder | **recommended path to admission** |
| A2 + B2 | high (kernel/dispatch) | low | trivial contract | defer; fold the sign-table half into B1 |
| A2 + B3 | — | — | — | reject (no such route) |
| A2/B4 | low | none | none | fallback if the census refutes B1's assumptions |

Recommendation: **A2 now** (it is a defect in its own right: a contract question
must not make a model unservable), then **B1 with C2** for admission, with B4 held
as the honest fallback if the census shows the launch set or geometry moving with
position.

### Landing (A2)

**Problem D1 fixed.** The refusal is now scoped to the retained body instead of
"any replay backend enabled": `ReplayController::retained_body_active()` is
`is_recording() || should_route_aql() || should_route_pm4()`
(`crates/rdna-compute/src/replay.rs`), and the guard refuses only then, through the
single admission point
`hipfire_dispatch::pipeline::sealed_moe::specialized_sealed_moe_retained_admission()`
— so the launch guard and the arming hook read one rule and cannot drift.

**Qwen4 gained the retained-body discipline** at the engine level
(`crates/hipfire-generate`: `ar::retained_body_action` + the Qwen4 AR producer's
closures), keeping `hipfire-arch-qwen4` free of replay code:

- prefill marks itself ineligible, so it can neither record nor consume a tape;
- plain single-token decode is the eligible forward; with admission refused it
  **poisons before running the body** and logs the reason, so this and every later
  forward runs HIP instead of arming a capture the route would refuse per token;
- a capture window that fails inside the body poisons rather than failing every
  following forward;
- a routed state (`Ready`) fails closed while no plan can be prepared for this
  family — it never silently runs HIP behind a plan the engine believes is in use.

Deliberate deviation from the earlier sketch: the "arm once `complete > 0`" gate is
**not** implemented here. Arming is refused wholesale today, so the gate would be
dead code; it belongs with B1, where capture-completeness (the pool launch being
present) is knowable. This is recorded rather than half-built.

**Evidence.**

- `replay::tests::retained_body_scope_is_the_eligible_forward_not_the_backend_choice`
  — armed-but-idle, captured-but-unprepared, ineligible-inside-a-window, and
  poisoned states are all *not* the retained body.
- `ar::route_scope_tests::retained_body_action_keeps_hip_for_every_non_eligible_or_refused_forward`
  — the action matrix across all seven replay states.
- End-to-end, two fresh processes, `.mq4r` artifact with the Redline default
  **armed** (the configuration that previously failed at prefill preflight):

  ```text
  [redline] enabling fail-closed retained default on gfx1151 (model_arch=qwen4, drafter=off, transport=pm4)
  [redline] qwen4 retained body unavailable: specialized sealed MoE has no retained-replay pointer contract; refusing the retained body (the model runs on HIP)
  {"content":"2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37","tokens":116,"tok_s":11.4,"finish_reason":"stop"}
  ```

  The decoded stream and tok/s are identical to the explicit
  `HIPFIRE_REPLAY_BACKEND=hip` baseline, i.e. the scope fix changes nothing about
  the HIP path and restores serving. Capture stays fail-closed: `is_recording()`
  remains part of the refused scope, pinned by the unit test above; no Qwen4 code
  path arms a window while admission is refused, so the guard is a backstop rather
  than a reachable refusal today.

### Acceptance evidence per option

- A2: a `.mq4r` Qwen4 serve with a replay backend armed loads, prefills, and
  generates on HIP, logging a fallback reason; a capture-window failure poisons
  rather than erroring the request; a unit test pins the guard's new predicate
  (`is_recording`/`should_route_*`, not `is_enabled`).
- B1: table-content proof with a negative test (a mutated entry must refuse);
  mapping fingerprint recorded with the tape and re-proved at prepare with a
  negative test; sign tables proven resident before the capture window;
  `HIPFIRE_DUMP_HIDDEN` refused while recording; census (REDLINE §7 gate 2) with a
  reconciled count; then gates 3–8 of the REDLINE ladder.
- C2: the diagnostic override is non-default, cannot install a plan, and any number
  it produces is labelled discovery-only in this record.

Until G4 lands, an explicit `HIPFIRE_REPLAY_BACKEND=hip` run is the healthy
baseline (REDLINE gate 1), and it is also the A/B arm for the later census.

## Verification ladder

| Stage | Route | State |
|---|---|---|
| Unit: G1 funnel entry | GPU test `tensor_ops::tests::recorded_blob_launch_enters_the_tape_with_the_bytes_it_launched` (gfx1151; skips elsewhere) | **passing** |
| Unit: G2 binding vocabulary | 7 tests in `crates/rdna-compute/src/replay.rs` | **passing** |
| Unit: G3 pinned shapes + declarations | `tensor_ops::tests::{pinned_qsa_shapes_are_bit_identical_to_derived_shapes, qsa_position_fields_are_declared_to_the_recorder}` (gfx1151) | **passing** |
| Contract: row-absolute column offset | `test_copy_rows_strided_f32_parity` 7/7 cases bit-exact (needs `--features lab`) | **passing** |
| Real-model shape pin A/B | 5 interleaved fresh-process pairs, greedy 116-token stream bit-identical, tok/s medians equal | **passing** (HIP path only) |
| Structural: no raw launch left in the program's op owner | `tensor_ops.rs` launch-discipline review (0 raw sites) | **passing** |
| Ordinary HIP baseline | `HIPFIRE_REPLAY_BACKEND=hip` daemon probe: loads, generates (`PARIS`), commits | **passing** |
| G4 A2 scope | armed-default `.mq4r` probe: serves on HIP, logs the refusal reason, stream identical to the HIP baseline | **passing** |
| G4 A2 policy | `replay::tests::retained_body_scope_*`, `ar::route_scope_tests::retained_body_action_*` | **passing** |
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
- 2026-09-20 — **G4 A2 landed** (branch-implemented, uncertified) in
  `0c758739b`: the refusal is
  scoped to the retained body (`ReplayController::retained_body_active()` plus one
  admission point in dispatch), and Qwen4 gained the engine-level retained-body
  discipline in `hipfire-generate` (prefill ineligible; decode poisons when the
  route cannot be retained; body failure inside a window poisons; a routed state
  fails closed). Verified by two policy unit tests and two fresh-process armed
  probes that now serve on HIP with the refusal reason logged and a stream
  identical to the HIP baseline. The `complete > 0` arming gate is deferred to B1
  (arming is refused wholesale today, so it would be dead code).
- 2026-09-20 — **G4 analyzed** (no decision): split into D1 (the refusal's blast
  radius — it disables prefill, fallback and every forward, making the `.mq4r`
  artifact unservable) and D2 (the unstated pointer contract). The route is already
  close to conformant: load-time pointer tables that are never rewritten, no
  per-forward host table or H2D, no position-derived scalar anywhere in the route,
  and per-seal pointer-identity re-proof with an existing mapping fingerprint. The
  real gaps are table-*content* proof on the Single path, a plan-pinned mapping
  identity, names for the lifetime hazards (lazy FWHT sign tables, dump-hidden D2H,
  invalidation paths), and the census. Options A1–A3, B1–B4, C1–C2 recorded with
  weights; recommended A2 now and B1+C2 for admission, with B4 as the honest
  fallback.
- 2026-09-20 — **G3 decided and landed** (branch-implemented, uncertified) in
  `360a44ab1`:
  capacity-fixed pool grid + capacity-pinned LDS with a constant symbol + declared
  position fields (including the new `PositionMulU32` for the index-key row offset,
  which replaced a position-shifted destination pointer) + `copy_rows_strided_f32`
  bounded by the destination extent instead of one row pitch. Evidence: 2 new GPU
  tests (shape pin bit-exactness incl. the zero-count symbol swap; declarations
  point at the recorded slots), 7/7 copy-parity cases, 5 interleaved fresh-process
  A/B pairs with a bit-identical 116-token greedy stream and equal tok/s medians,
  263 + 285 regression tests. The `HIPFIRE_QSA_STABLE_SHAPES` measurement gate was
  consumed and deleted. Route arming (`complete > 0`) still belongs with G4.
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
