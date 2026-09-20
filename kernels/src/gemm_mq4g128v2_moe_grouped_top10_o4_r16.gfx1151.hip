// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — Qwen4 QT53 grouped down O4×R16 kernel for gfx1151.
//
// The previous production sibling assigned one wave to two adjacent output
// rows and sixteen grouped route slots. This kernel keeps that exact R16
// mapping, the per-element decode arithmetic, the padded-K handling, and the
// wave32 shuffle association, but one wave computes four adjacent output rows.
// The X value for each lane/slot/group is loaded once and consumed by all four
// row accumulator chains, which halves the X bytes each output element pays
// for; each row still decodes its own QT53 weights and owns its own
// independent F32 accumulator chain.
//
// Measured on the exact routed shape (E=512, M=2560, K=640, grouped_rows=10592,
// N=291) this is 1.83x the previous sibling's time in the focused driver.
// Three structural changes carry that, all arithmetic-invisible:
//
//   * The accumulation statement is the production sibling's text, verbatim,
//     per row: acc += (scale*n0 + zero)*x0 + (scale*n1 + zero)*x1 + ... with
//     the same left-to-right association. Production compiles these kernels
//     without fast-math (hipcc --genco -O3, no -ffast-math), so the source
//     association is the whole contract: no reassociation is permitted and a
//     four-row body cannot disagree with the two-row sibling. Dequantizing
//     with explicit fused intrinsics or re-associating the sum changes the
//     last bits and is therefore wrong here.
//   * Full QT53 groups read each lane's four K values with one aligned
//     float4 load instead of four guarded scalar loads. A lane's slice
//     (group * 128 + 4 * tid) is 16-byte aligned whenever the row stride
//     K * 4 is, so the vector path is admitted only for K % 4 == 0. The
//     arithmetically visible values are identical. The one partial group a
//     K that is not a multiple of 128 can produce is peeled into a tail loop
//     that keeps the production sibling's guarded scalar loads and
//     padded-value handling exactly.
//   * Loads are batched, because the sibling's codegen issued one X load per
//     slot with an immediate full wait. Four slots' X loads are now issued
//     together before their FMA blocks, and each row's raw weight bytes are
//     loaded for all four rows before any row is decoded.
//
// Launch contract (same arguments and output layout as the production sibling):
//   grid = [ceil(M / 4), ceil(grouped_rows / 16), 1], block = [32, 1, 1]
//   expert_tile_ids[tile] is -1 for a sentinel tile; slots beyond
//   grouped_rows are padding and are never written.
//   Y_grouped[grouped_slot * M + row] is unweighted grouped output.
//   Rows at or beyond M are never written and their accumulator chains stay
//   zero. Their weight loads are clamped to the last valid row of the same
//   expert, so no read leaves the expert matrix.
//
// QT53 layout: each logical K=128 group occupies 68 bytes: fp16 scale,
// fp16 zero, and 64 packed nibbles. The final group is logically padded to
// 128 values, but values at k >= K are multiplied by zero exactly as in the
// production sibling. The caller owns the unchanged BF16 round-trip boundary.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__gfx1151__)

__launch_bounds__(32, 8)
extern "C" __global__ void gemm_mq4g128v2_moe_grouped_top10_o4_r16_gfx1151(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const float* __restrict__ X_src,
    float* __restrict__ Y_grouped,
    int M,
    int K,
    int x_row_div,
    int grouped_rows,
    int x_src_rows,
    int n_exp
) {
    const int tid = threadIdx.x;
    const int row0 = blockIdx.x * 4;
    const int tile = blockIdx.y;
    const int slot_start = tile * 16;
    if (row0 >= M || slot_start >= grouped_rows || tid >= 32) return;

    const int expert = expert_tile_ids[tile];
    float acc[4][16];
    const float* x_rows[16];
    bool row_valid[4];
    #pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) acc[r][slot] = 0.0f;
        x_rows[slot] = nullptr;
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) row_valid[r] = (row0 + r) < M;

    if (expert >= 0 && expert < n_exp && K > 0 && x_row_div > 0) {
        #pragma unroll
        for (int slot = 0; slot < 16; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot >= grouped_rows) continue;
            const int flat = sorted_slot_index[grouped_slot];
            if (flat < 0) continue;
            // The Qwen4 grouped-down caller passes x_row_div == 1: every route
            // slot owns its own X row.  Skip the runtime integer division for
            // that case; the general divisor keeps its own path.
            const int x_row = (x_row_div == 1) ? flat : flat / x_row_div;
            if (x_row >= 0 && x_row < x_src_rows)
                x_rows[slot] = X_src + (long long)x_row * K;
        }

        const char* A = reinterpret_cast<const char*>(expert_ptrs[expert]);
        const int groups = (K + 127) / 128;
        const long long row_stride = (long long)groups * 68;
        const char* row_ptr[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = row_valid[r] ? (row0 + r) : (M - 1);
            row_ptr[r] = A + (long long)row * row_stride;
        }
        // A full group needs all four lane K values in range and a 16-byte
        // aligned row stride. Anything else keeps the production sibling's
        // guarded scalar path.
        const int vector_groups = (K % 4 == 0) ? (K / 128) : 0;

        // Pinned production contract: dequantize first, then accumulate the
        // four K values in order, every product fused into its accumulation:
        //   (((acc + w0*x0) + w1*x1) + w2*x2) + w3*x3
        // Clang's fast-math reassociation is free to permute an unpinned
        // chain, which changes the last bits of every output.
        auto accumulate = [&](int slot, float x0, float x1, float x2, float x3,
                              const float (&scale)[4], const float (&zero)[4],
                              const unsigned char (&b0)[4],
                              const unsigned char (&b1)[4]) {
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                acc[r][slot] += (scale[r] * (float)(b0[r] & 0xF) + zero[r]) * x0
                    + (scale[r] * (float)(b0[r] >> 4) + zero[r]) * x1
                    + (scale[r] * (float)(b1[r] & 0xF) + zero[r]) * x2
                    + (scale[r] * (float)(b1[r] >> 4) + zero[r]) * x3;
            }
        };

        for (int group = 0; group < vector_groups; ++group) {
            // Issue every row's raw group bytes before decoding any of them:
            // eight independent loads in flight instead of four dependent
            // load/decode pairs.
            unsigned int header[4];
            unsigned short payload[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (row_valid[r]) {
                    const char* gp = row_ptr[r] + (long long)group * 68;
                    header[r] = *(const unsigned int*)(gp + 0);
                    payload[r] = *(const unsigned short*)(gp + 4 + tid * 2);
                } else {
                    header[r] = 0U;
                    payload[r] = static_cast<unsigned short>(0);
                }
            }
            float scale[4];
            float zero[4];
            unsigned char b0[4];
            unsigned char b1[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                scale[r] = __half2float(__ushort_as_half(
                    (unsigned short)(header[r] & 0xFFFFU)));
                zero[r] = __half2float(__ushort_as_half(
                    (unsigned short)(header[r] >> 16)));
                b0[r] = (unsigned char)(payload[r] & 0xFFU);
                b1[r] = (unsigned char)(payload[r] >> 8);
            }
            const int base = group * 128 + tid * 4;
            // Four slots per batch: the four X loads are independent, so
            // issuing them together gives the scheduler four in-flight loads
            // before the first FMA needs one. Slots with no X row keep the
            // exact previous behaviour of contributing nothing.
            #pragma unroll
            for (int batch = 0; batch < 16; batch += 4) {
                float4 values[4];
                bool live[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float* x = x_rows[batch + i];
                    live[i] = x != nullptr;
                    values[i] = live[i]
                        ? *reinterpret_cast<const float4*>(x + base)
                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    if (!live[i]) continue;
                    const float4 xv = values[i];
                    accumulate(batch + i, xv.x, xv.y, xv.z, xv.w, scale, zero, b0, b1);
                }
            }
        }

        // The single partial group a K that is not a multiple of 128 can
        // produce keeps the production sibling's guarded scalar loads.
        for (int group = vector_groups; group < groups; ++group) {
            const int base = group * 128 + tid * 4;
            const int i0 = base;
            const int i1 = base + 1;
            const int i2 = base + 2;
            const int i3 = base + 3;
            // Issue every row's raw group bytes before decoding any of them:
            // eight independent loads in flight instead of four dependent
            // load/decode pairs.
            unsigned int header[4];
            unsigned short payload[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (row_valid[r]) {
                    const char* gp = row_ptr[r] + (long long)group * 68;
                    header[r] = *(const unsigned int*)(gp + 0);
                    payload[r] = *(const unsigned short*)(gp + 4 + tid * 2);
                } else {
                    header[r] = 0U;
                    payload[r] = static_cast<unsigned short>(0);
                }
            }
            float scale[4];
            float zero[4];
            unsigned char b0[4];
            unsigned char b1[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                scale[r] = __half2float(__ushort_as_half(
                    (unsigned short)(header[r] & 0xFFFFU)));
                zero[r] = __half2float(__ushort_as_half(
                    (unsigned short)(header[r] >> 16)));
                b0[r] = (unsigned char)(payload[r] & 0xFFU);
                b1[r] = (unsigned char)(payload[r] >> 8);
            }
            #pragma unroll
            for (int slot = 0; slot < 16; ++slot) {
                const float* x = x_rows[slot];
                if (x == nullptr) continue;
                const float x0 = i0 < K ? x[i0] : 0.0f;
                const float x1 = i1 < K ? x[i1] : 0.0f;
                const float x2 = i2 < K ? x[i2] : 0.0f;
                const float x3 = i3 < K ? x[i3] : 0.0f;
                accumulate(slot, x0, x1, x2, x3, scale, zero, b0, b1);
            }
        }
    }

    // Match the production sibling's wave32 association independently for all
    // four output rows. A slot with no X row never accumulated anything, so
    // its accumulator is still exactly +0.0 and needs no shuffle rounds: the
    // reduction is skipped for those slots and for rows past M, both of which
    // are uniform (whole-wave) decisions. Every reduced accumulator keeps the
    // production sibling's offsets 16, 8, 4, 2, 1 and association.
    #pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        if (x_rows[slot] == nullptr) continue;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (!row_valid[r]) continue;
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[r][slot] += __shfl_down(acc[r][slot], offset, 32);
            }
        }
    }

    if (tid == 0) {
        #pragma unroll
        for (int slot = 0; slot < 16; ++slot) {
            const int grouped_slot = slot_start + slot;
            if (grouped_slot >= grouped_rows) continue;
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (row_valid[r])
                    Y_grouped[(long long)grouped_slot * M + row0 + r] =
                        acc[r][slot];
            }
        }
    }
}

#endif
