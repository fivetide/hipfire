// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact BF16-weight × F32-input gfx1151 R16 GEMM.
//
// This gfx1151-specific kernel owns sixteen output-channel rows and two token
// rows, with no output-row cooperation or LDS staging.

#include <hip/hip_runtime.h>

#define HF_BF16_LO(u) __uint_as_float((u) << 16)
#define HF_BF16_HI(u) __uint_as_float((u) & 0xFFFF0000u)
#define HF_R16_ROWS 16
#define HF_R16_STREAMS 2

// Layouts match kernels/src/gemm_bf16_xf32_multirow.hip:
//   W: [M, K] BF16, row-major
//   X: [N, K] F32, row-major
//   Y: [N, M] F32, row-major
//
// Mapping and arithmetic invariants:
//   * Grid is [ceil(N / 2), ceil(M / 16), 1], block is [32, 1, 1].  One wave
//     owns sixteen adjacent output-channel rows and two adjacent token rows,
//     so the token tile is the fast-varying grid dimension: the sixteen weight
//     rows a block group reads stay resident while the token tiles of that
//     group sweep past, which is what keeps the weight stream off DRAM.
//   * The stream loop is statically unrolled so each a0/a1/a2/a3[row][stream]
//     accumulator retains the original R4 association while each BF16 weight
//     vector is reused across both X rows and each X vector across all sixteen
//     output rows.  Sixteen output rows per wave make the activation traffic
//     per FMA four times smaller than the four-row tile's and sixteen times
//     smaller than the single-row gemv's.
//   * The wide K path is the reference lane*8 / +256 traversal.  For every
//     wide step, one lane loads and widens one uint4 of BF16 weights per
//     output row, then loads and consumes each X stream's eight floats before
//     advancing to the next output row; no array of live float4 X vectors is
//     required.  The scalar path is the reference lane / +32 traversal used
//     whenever K % 8 != 0.
//   * Every (row, stream) pair uses four accumulators, combines as
//     `(a0 + a1) + (a2 + a3)`, then performs the same shuffle offsets
//     16, 8, 4, 2, 1.  BF16 weights are widened by the same raw-bit shifts and
//     are never converted or staged through another precision, so each output
//     element is bitwise identical to the four-row
//     `gemm_bf16_xf32_multirow` result.
//   * Final partial N streams alias batch_base, the first valid input row.
//     Their results are not stored.  The `row_base + row >= M` guard keeps an
//     output row out of range at the top of M from being read or written
//     (every production M on this route is a multiple of sixteen); M/N guards
//     occur before any dereference; this kernel has no inter-thread barrier or
//     shared state.

__launch_bounds__(32, 4)
extern "C" __global__ void gemm_bf16_xf32_multirow_r16_gfx1151(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row_base = blockIdx.y * HF_R16_ROWS;
    const int batch_base = blockIdx.x * HF_R16_STREAMS;
    const int lane = threadIdx.x;
    if (row_base >= M || batch_base >= N || lane >= 32) return;

    int eff_row[HF_R16_ROWS];
    const unsigned short* w_row[HF_R16_ROWS];
#pragma unroll
    for (int row = 0; row < HF_R16_ROWS; ++row) {
        eff_row[row] = (row_base + row < M) ? row_base + row : row_base;
        w_row[row] = W + (size_t)eff_row[row] * K;
    }

    float a0[HF_R16_ROWS][HF_R16_STREAMS];
    float a1[HF_R16_ROWS][HF_R16_STREAMS];
    float a2[HF_R16_ROWS][HF_R16_STREAMS];
    float a3[HF_R16_ROWS][HF_R16_STREAMS];
#pragma unroll
    for (int row = 0; row < HF_R16_ROWS; ++row) {
#pragma unroll
        for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
            a0[row][stream] = 0.0f;
            a1[row][stream] = 0.0f;
            a2[row][stream] = 0.0f;
            a3[row][stream] = 0.0f;
        }
    }

    if ((K & 7) == 0) {
        // Eight BF16 values per lane per step, exactly matching R4's global
        // K order.  X vectors are loaded and consumed one stream and one
        // output row at a time.
        for (int i = lane * 8; i + 8 <= K; i += 32 * 8) {
            // Widen every output row's weight vector before consuming any of
            // them: the sixteen loads are independent, so the whole group
            // issues before the first dependent accumulate stalls.
            float w[HF_R16_ROWS][8];
#pragma unroll
            for (int row = 0; row < HF_R16_ROWS; ++row) {
                const uint4 wv = *(const uint4*)(w_row[row] + i);
                w[row][0] = HF_BF16_LO(wv.x);
                w[row][1] = HF_BF16_HI(wv.x);
                w[row][2] = HF_BF16_LO(wv.y);
                w[row][3] = HF_BF16_HI(wv.y);
                w[row][4] = HF_BF16_LO(wv.z);
                w[row][5] = HF_BF16_HI(wv.z);
                w[row][6] = HF_BF16_LO(wv.w);
                w[row][7] = HF_BF16_HI(wv.w);
            }
#pragma unroll
            for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
                const int batch = (batch_base + stream < N)
                    ? batch_base + stream
                    : batch_base;
                const float* x_row = X + (size_t)batch * K;
                const float4 xa = *(const float4*)(x_row + i);
                const float4 xb = *(const float4*)(x_row + i + 4);
#pragma unroll
                for (int row = 0; row < HF_R16_ROWS; ++row) {
                    a0[row][stream] += w[row][0] * xa.x;
                    a1[row][stream] += w[row][1] * xa.y;
                    a2[row][stream] += w[row][2] * xa.z;
                    a3[row][stream] += w[row][3] * xa.w;
                    a0[row][stream] += w[row][4] * xb.x;
                    a1[row][stream] += w[row][5] * xb.y;
                    a2[row][stream] += w[row][6] * xb.z;
                    a3[row][stream] += w[row][7] * xb.w;
                }
            }
        }
    } else {
        // For K%8 != 0 the reference intentionally uses scalar traversal for
        // every element, not only the final partial eight-value group.
        for (int i = lane; i < K; i += 32) {
#pragma unroll
            for (int row = 0; row < HF_R16_ROWS; ++row) {
                const float weight =
                    __uint_as_float(((unsigned int)w_row[row][i]) << 16);
#pragma unroll
                for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
                    const int batch = (batch_base + stream < N)
                        ? batch_base + stream
                        : batch_base;
                    const float* x_row = X + (size_t)batch * K;
                    const float x = x_row[i];
                    a0[row][stream] += weight * x;
                }
            }
        }
    }

    float sum[HF_R16_ROWS][HF_R16_STREAMS];
#pragma unroll
    for (int row = 0; row < HF_R16_ROWS; ++row) {
#pragma unroll
        for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
            sum[row][stream] = (a0[row][stream] + a1[row][stream])
                + (a2[row][stream] + a3[row][stream]);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int row = 0; row < HF_R16_ROWS; ++row) {
#pragma unroll
            for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
                sum[row][stream] += __shfl_down(sum[row][stream], offset, 32);
            }
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int row = 0; row < HF_R16_ROWS; ++row) {
            if (row_base + row >= M) continue;
#pragma unroll
            for (int stream = 0; stream < HF_R16_STREAMS; ++stream) {
                if (batch_base + stream < N) {
                    Y[(size_t)(batch_base + stream) * M + row_base + row] =
                        sum[row][stream];
                }
            }
        }
    }
}
