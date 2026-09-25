// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — gfx1151 QT44 grouped top-10 gate/up O4×R4.
//
// This exact-shape gfx1151 path keeps the static production kernel's F32 input,
// MQ4G256V2 decoding, four static accumulator chains per output row, and full
// wave32 reduction, while pairing four adjacent output rows with four route
// slots.  For each group and slot the eight X scalars are loaded once and
// consumed by all four output rows, so X bytes per FMA halve relative to the
// O2×R8 sibling; the extra weight words per group keep the accumulator budget
// and the per-element arithmetic identical.
//
// Layouts are the live grouped gate/up contract:
//   expert_weight_ptrs: [E] device pointers to [M, K] MQ4G256V2 rows
//   expert_tile_ids:   [ceil(grouped_rows / 16)] expert ID per old 16-slot tile
//   sorted_slot_index: [grouped_rows] flat token/top-k slot, -1 for padding
//   X_src:             [N, K] F32 rotated token rows
//   Y_grouped:         [grouped_rows, M] F32 grouped output
//
// Grid: [ceil(M / 4), ceil(grouped_rows / 4), 1], block 32.  blockIdx.y is a
// four-slot subtile, but the metadata remains at the old sixteen-slot tile
// granularity: subtile s reads expert_tile_ids[s / 4].  Thus the four subtile
// blocks belonging to one old tile use exactly the same expert ID.
//
// Arithmetic invariants:
//   * Each block owns four adjacent output rows and four slots.  It has
//     4 chains × 4 slots × 4 output rows = 64 static accumulators, the same
//     budget as the O2×R8 sibling's 4 chains × 8 slots × 2 rows.
//   * Each output row processes groups in the same complete-quad/tail order as
//     the O2×R8 and O1×R16 kernels.  Its final chain association remains
//     (acc0 + acc1) + (acc2 + acc3), followed by the same full-wave shuffle
//     offsets 16, 8, 4, 2, 1.  Every row follows the identical sequence.
//   * A subtile with no live slot accumulates nothing, so eliding its walk
//     leaves every accumulator at zero and the epilogue writes exactly the
//     zeros the full walk wrote.  The group walk bound is the only dynamic
//     part of the loop structure.
//   * Odd M leaves rows 1..3 possibly invalid.  An invalid row's pointer
//     aliases row0 only to keep pointer formation harmless, its header word and
//     both half constants are forced to zero so every one of its products is
//     exactly zero, and only its epilogue stores are gated.  Each valid row's
//     arithmetic is therefore the production arithmetic verbatim.  A negative
//     expert ID leaves the output sentinel untouched, matching the O2×R8
//     sibling's early return.  Negative sorted slots write zero, and a partial
//     final four-slot subtile writes only grouped_rows-valid slots.
//
// Four rows need four decoded weight states live at once.  The lower residency
// request avoids forcing spills while retaining the production arithmetic and
// output contract.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HF_MQ4_O4_R4_SLOTS 4

// Lane 0 of the `v += __shfl_down(v, off, 32)` tree (off = 16..1) with the
// partner values moved by permlanex16 / in-row DPP instead of LDS bpermute.
// Same additions, operands and order; only lane 0's result (the one stored)
// is defined.
static __device__ __forceinline__ float hf_lane0_tree_sum(float v) {
    v += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x108, 0xf, 0xf, true));  // row_shl:8
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x104, 0xf, 0xf, true));  // row_shl:4
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x102, 0xf, 0xf, true));  // row_shl:2
    v += __int_as_float(__builtin_amdgcn_update_dpp(
        0, __float_as_int(v), 0x101, 0xf, 0xf, true));  // row_shl:1
    return v;
}
// x & 0x0F0F0F0F, opaque to the optimizer so each nibble converts with one
// v_cvt_f32_ubyteN instead of being re-canonicalized into v_bfe + cvt.
static __device__ __forceinline__ unsigned int hf_nibble_bytes(unsigned int x) {
    unsigned int r;
    asm("v_and_b32 %0, 0x0f0f0f0f, %1" : "=v"(r) : "v"(x));
    return r;
}
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_top10_o4_r4(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const float* __restrict__ X_src,
    float* __restrict__ Y_grouped,
    int M,
    int K,
    int x_row_div,
    int grouped_rows
) {
    const int row0 = blockIdx.x * 4;
    const int row1 = row0 + 1;
    const int row2 = row0 + 2;
    const int row3 = row0 + 3;
    const int subtile = blockIdx.y;
    const int slot_start = subtile * HF_MQ4_O4_R4_SLOTS;
    const int tid = threadIdx.x;
    if (row0 >= M || slot_start >= grouped_rows || tid >= 32) return;

    // Slot indices are loaded (clamped in range) up front, alongside the
    // expert ID, instead of one dependent scalar round trip per slot.
    int flat_raw[HF_MQ4_O4_R4_SLOTS];
#pragma unroll
    for (int slot = 0; slot < HF_MQ4_O4_R4_SLOTS; ++slot) {
        const int grouped_slot = slot_start + slot;
        flat_raw[slot] = sorted_slot_index[
            grouped_slot < grouped_rows ? grouped_slot : grouped_rows - 1];
    }

    // Production expert metadata is still one entry per old 16-slot tile, so
    // the four R4 subtiles of a tile share the same expert pointer.
    const int old_tile = subtile >> 2;
    const int expert_id = expert_tile_ids[old_tile];
    // Keep the slot loads here: otherwise they sink below the branches and
    // go back to one round trip each.
    asm volatile("" :: "s"(flat_raw[0]), "s"(flat_raw[1]), "s"(flat_raw[2]),
                 "s"(flat_raw[3]));
    if (expert_id < 0) return;
    const char* __restrict__ A =
        reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const bool row1_valid = row1 < M;
    const bool row2_valid = row2 < M;
    const bool row3_valid = row3 < M;
    const int groups_per_row = K / 256;
    const long long row_stride = (long long)groups_per_row * 136;
    const char* __restrict__ row_ptr0 =
        A + (long long)row0 * row_stride;
    // Do not form out-of-range pointers for odd M.  These aliases are never
    // accumulated or stored when the corresponding row is invalid.
    const char* __restrict__ row_ptr1 = row1_valid
        ? A + (long long)row1 * row_stride
        : row_ptr0;
    const char* __restrict__ row_ptr2 = row2_valid
        ? A + (long long)row2 * row_stride
        : row_ptr0;
    const char* __restrict__ row_ptr3 = row3_valid
        ? A + (long long)row3 * row_stride
        : row_ptr0;

    const int boff = tid * 4;
    const int hoff = (tid < 16) ? 0 : 4;
    const int xoff = tid * 32;

    // Dead slots alias X_src row 0 so their loads are unconditional; their
    // accumulation is still skipped, leaving the +0.0 the store owes.
    bool live[HF_MQ4_O4_R4_SLOTS];
    int x_soff[HF_MQ4_O4_R4_SLOTS];
    bool any_live = false;
#pragma unroll
    for (int slot = 0; slot < HF_MQ4_O4_R4_SLOTS; ++slot) {
        const int grouped_slot = slot_start + slot;
        const int flat = (grouped_slot < grouped_rows) ? flat_raw[slot] : -1;
        live[slot] = flat >= 0;
        const int x_row = !live[slot] ? 0
            : (x_row_div > 1) ? (flat / x_row_div) : flat;
        x_soff[slot] = x_row * K * 4;
        any_live |= live[slot];
    }

    // With no live slot every weight word of this subtile is multiplied by
    // nothing, so the walk and its DRAM fetches are elided.  The fold, shuffle
    // and store still run and still write the zero every sentinel slot owes.
    const int group_walk = any_live ? groups_per_row : 0;

    float acc00[HF_MQ4_O4_R4_SLOTS] = {};
    float acc01[HF_MQ4_O4_R4_SLOTS] = {};
    float acc02[HF_MQ4_O4_R4_SLOTS] = {};
    float acc03[HF_MQ4_O4_R4_SLOTS] = {};
    float acc10[HF_MQ4_O4_R4_SLOTS] = {};
    float acc11[HF_MQ4_O4_R4_SLOTS] = {};
    float acc12[HF_MQ4_O4_R4_SLOTS] = {};
    float acc13[HF_MQ4_O4_R4_SLOTS] = {};
    float acc20[HF_MQ4_O4_R4_SLOTS] = {};
    float acc21[HF_MQ4_O4_R4_SLOTS] = {};
    float acc22[HF_MQ4_O4_R4_SLOTS] = {};
    float acc23[HF_MQ4_O4_R4_SLOTS] = {};
    float acc30[HF_MQ4_O4_R4_SLOTS] = {};
    float acc31[HF_MQ4_O4_R4_SLOTS] = {};
    float acc32[HF_MQ4_O4_R4_SLOTS] = {};
    float acc33[HF_MQ4_O4_R4_SLOTS] = {};

    // Software pipeline: group g+1's header/payload words and X vectors are
    // loaded while group g is decoded and accumulated.  Values and the
    // per-element expressions are the production ones.
    // Buffer loads: the uniform row/slot/group offsets ride in soffset and the
    // lane offset in voffset, so no per-group 64-bit VGPR address arithmetic.
    const __amdgpu_buffer_rsrc_t wrsrc =
        __builtin_amdgcn_make_buffer_rsrc((void*)A, 0, 0x7FFFFFFF, 0x31004000);
    const __amdgpu_buffer_rsrc_t xrsrc =
        __builtin_amdgcn_make_buffer_rsrc((void*)X_src, 0, 0x7FFFFFFF, 0x31004000);
    const int row_soff[4] = {
        (int)(row_ptr0 - A), (int)(row_ptr1 - A), (int)(row_ptr2 - A), (int)(row_ptr3 - A)};
    const bool row_ok[4] = {true, row1_valid, row2_valid, row3_valid};
    unsigned int nh[4], npk[4];
    float4 nxa[HF_MQ4_O4_R4_SLOTS], nxb[HF_MQ4_O4_R4_SLOTS];
#define LOAD_GROUP(G) do { \
        _Pragma("unroll") \
        for (int r = 0; r < 4; ++r) { \
            nh[r] = row_ok[r] ? __builtin_amdgcn_raw_buffer_load_b32( \
                wrsrc, hoff, row_soff[r] + (G) * 136, 0) : 0u; \
            npk[r] = row_ok[r] ? __builtin_amdgcn_raw_buffer_load_b32( \
                wrsrc, 8 + boff, row_soff[r] + (G) * 136, 0) : 0u; \
        } \
        _Pragma("unroll") \
        for (int slot = 0; slot < HF_MQ4_O4_R4_SLOTS; ++slot) { \
            const int xs = x_soff[slot] + (G) * 1024; \
            nxa[slot] = __builtin_bit_cast(float4, \
                __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xoff, xs, 0)); \
            nxb[slot] = __builtin_bit_cast(float4, \
                __builtin_amdgcn_raw_buffer_load_b128(xrsrc, xoff + 16, xs, 0)); \
        } \
    } while (0)
#define DECODE_ROW(SUF, R) \
    const float sc##SUF = __half2float(__ushort_as_half( \
        (unsigned short)(ch[R] & 0xFFFFu))); \
    const float zp##SUF = __half2float(__ushort_as_half( \
        (unsigned short)(ch[R] >> 16))); \
    const unsigned int lo##SUF = hf_nibble_bytes(cpk[R]); \
    const unsigned int hi##SUF = hf_nibble_bytes(cpk[R] >> 4); \
    const float w##SUF##0 = sc##SUF * (float)(lo##SUF & 0xFFu) + zp##SUF; \
    const float w##SUF##1 = sc##SUF * (float)(hi##SUF & 0xFFu) + zp##SUF; \
    const float w##SUF##2 = sc##SUF * (float)((lo##SUF >> 8) & 0xFFu) + zp##SUF; \
    const float w##SUF##3 = sc##SUF * (float)((hi##SUF >> 8) & 0xFFu) + zp##SUF; \
    const float w##SUF##4 = sc##SUF * (float)((lo##SUF >> 16) & 0xFFu) + zp##SUF; \
    const float w##SUF##5 = sc##SUF * (float)((hi##SUF >> 16) & 0xFFu) + zp##SUF; \
    const float w##SUF##6 = sc##SUF * (float)(lo##SUF >> 24) + zp##SUF; \
    const float w##SUF##7 = sc##SUF * (float)(hi##SUF >> 24) + zp##SUF
#define ACCUM_GROUP(G, A0, A1, A2, A3) do { \
        unsigned int ch[4], cpk[4]; \
        float4 cxa[HF_MQ4_O4_R4_SLOTS], cxb[HF_MQ4_O4_R4_SLOTS]; \
        _Pragma("unroll") \
        for (int r = 0; r < 4; ++r) { ch[r] = nh[r]; cpk[r] = npk[r]; } \
        _Pragma("unroll") \
        for (int slot = 0; slot < HF_MQ4_O4_R4_SLOTS; ++slot) { \
            cxa[slot] = nxa[slot]; \
            cxb[slot] = nxb[slot]; \
        } \
        if ((G) + 1 < group_walk) LOAD_GROUP((G) + 1); \
        DECODE_ROW(0, 0); \
        DECODE_ROW(1, 1); \
        DECODE_ROW(2, 2); \
        DECODE_ROW(3, 3); \
        _Pragma("unroll") \
        for (int slot = 0; slot < HF_MQ4_O4_R4_SLOTS; ++slot) { \
            if (!live[slot]) continue; \
            const float x0 = cxa[slot].x; \
            const float x1 = cxa[slot].y; \
            const float x2 = cxa[slot].z; \
            const float x3 = cxa[slot].w; \
            const float x4 = cxb[slot].x; \
            const float x5 = cxb[slot].y; \
            const float x6 = cxb[slot].z; \
            const float x7 = cxb[slot].w; \
            (A0)[slot] += w00 * x0 + w01 * x1 + w02 * x2 + w03 * x3 \
                + w04 * x4 + w05 * x5 + w06 * x6 + w07 * x7; \
            (A1)[slot] += w10 * x0 + w11 * x1 + w12 * x2 + w13 * x3 \
                + w14 * x4 + w15 * x5 + w16 * x6 + w17 * x7; \
            (A2)[slot] += w20 * x0 + w21 * x1 + w22 * x2 + w23 * x3 \
                + w24 * x4 + w25 * x5 + w26 * x6 + w27 * x7; \
            (A3)[slot] += w30 * x0 + w31 * x1 + w32 * x2 + w33 * x3 \
                + w34 * x4 + w35 * x5 + w36 * x6 + w37 * x7; \
        } \
    } while (0)

    if (group_walk > 0) LOAD_GROUP(0);
    int g = 0;
    for (; g + 3 < group_walk; g += 4) {
        ACCUM_GROUP(g, acc00, acc10, acc20, acc30);
        ACCUM_GROUP(g + 1, acc01, acc11, acc21, acc31);
        ACCUM_GROUP(g + 2, acc02, acc12, acc22, acc32);
        ACCUM_GROUP(g + 3, acc03, acc13, acc23, acc33);
    }
    for (; g < group_walk; ++g) {
        switch (g & 3) {
        case 0:
            ACCUM_GROUP(g, acc00, acc10, acc20, acc30);
            break;
        case 1:
            ACCUM_GROUP(g, acc01, acc11, acc21, acc31);
            break;
        case 2:
            ACCUM_GROUP(g, acc02, acc12, acc22, acc32);
            break;
        default:
            ACCUM_GROUP(g, acc03, acc13, acc23, acc33);
            break;
        }
    }
#undef ACCUM_GROUP
#undef DECODE_ROW
#undef LOAD_GROUP

    float sum0[HF_MQ4_O4_R4_SLOTS];
    float sum1[HF_MQ4_O4_R4_SLOTS];
    float sum2[HF_MQ4_O4_R4_SLOTS];
    float sum3[HF_MQ4_O4_R4_SLOTS];
#pragma unroll
    for (int slot = 0; slot < HF_MQ4_O4_R4_SLOTS; ++slot) {
        sum0[slot] = (acc00[slot] + acc01[slot])
            + (acc02[slot] + acc03[slot]);
        sum1[slot] = (acc10[slot] + acc11[slot])
            + (acc12[slot] + acc13[slot]);
        sum2[slot] = (acc20[slot] + acc21[slot])
            + (acc22[slot] + acc23[slot]);
        sum3[slot] = (acc30[slot] + acc31[slot])
            + (acc32[slot] + acc33[slot]);
    }
    // Full-wave reduction is deliberate: each output row has the exact same
    // lane association as the production O2×R8 and O1×R16 kernels (lane 0 of
    // the shuffle-down tree, partners moved by permlane/DPP).
#pragma unroll
    for (int slot = 0; slot < HF_MQ4_O4_R4_SLOTS; ++slot) {
        sum0[slot] = hf_lane0_tree_sum(sum0[slot]);
        sum1[slot] = hf_lane0_tree_sum(sum1[slot]);
        sum2[slot] = hf_lane0_tree_sum(sum2[slot]);
        sum3[slot] = hf_lane0_tree_sum(sum3[slot]);
    }

    if (tid == 0) {
#pragma unroll
        for (int slot = 0; slot < HF_MQ4_O4_R4_SLOTS; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot < grouped_rows) {
                Y_grouped[(long long)grouped_slot * M + row0] = sum0[slot];
                if (row1_valid)
                    Y_grouped[(long long)grouped_slot * M + row1] = sum1[slot];
                if (row2_valid)
                    Y_grouped[(long long)grouped_slot * M + row2] = sum2[slot];
                if (row3_valid)
                    Y_grouped[(long long)grouped_slot * M + row3] = sum3[slot];
            }
        }
    }
}

#undef HF_MQ4_O4_R4_SLOTS
