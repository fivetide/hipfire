// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — exact BF16-weight × F32-input multirow prefill GEMM.

#include <hip/hip_runtime.h>

#define HF_BF16_LO(u) __uint_as_float((u) << 16)
#define HF_BF16_HI(u) __uint_as_float((u) & 0xFFFF0000u)

// Layouts:
//   W: [M, K] BF16, row-major
//   X: [N, K] F32, row-major
//   Y: [N, M] F32, row-major
//
// One wave owns one output-channel row and four adjacent token rows.  The
// per-token lane accumulators intentionally match gemv_bf16_xf32: each lane
// loads eight BF16 values with uint4, distributes them over four accumulators,
// reduces with (a0+a1)+(a2+a3), then performs the same wave shuffle.  The
// widened weights are reused across the four token rows, but no arithmetic
// association is changed from the batch-1 reference path.
//
// Grid: [M, ceil(N / 4), 1].  A partial final token tile aliases its last
// valid input row for masked lanes, so no out-of-range activation is read.
__launch_bounds__(32, 16)
extern "C" __global__ void gemm_bf16_xf32_multirow(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row = blockIdx.x;
    const int batch_base = blockIdx.y * 4;
    const int lane = threadIdx.x;
    if (row >= M || lane >= 32 || batch_base >= N) return;

    const unsigned short* w_row = W + (size_t)row * K;
    const int b0 = batch_base;
    const int b1 = (batch_base + 1 < N) ? batch_base + 1 : b0;
    const int b2 = (batch_base + 2 < N) ? batch_base + 2 : b0;
    const int b3 = (batch_base + 3 < N) ? batch_base + 3 : b0;
    const float* x0 = X + (size_t)b0 * K;
    const float* x1 = X + (size_t)b1 * K;
    const float* x2 = X + (size_t)b2 * K;
    const float* x3 = X + (size_t)b3 * K;

    float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
    float a20 = 0.0f, a21 = 0.0f, a22 = 0.0f, a23 = 0.0f;
    float a30 = 0.0f, a31 = 0.0f, a32 = 0.0f, a33 = 0.0f;

    // Match gemv_bf16_xf32's wide K traversal and association exactly:
    // each lane handles eight consecutive BF16 values, with four accumulators
    // reused for positions 0/4, 1/5, 2/6, and 3/7 in that eight-value group.
    if ((K & 7) == 0) {
        for (int i = lane * 8; i + 8 <= K; i += 32 * 8) {
            const uint4 wv = *(const uint4*)(w_row + i);
            const float4 xa0 = *(const float4*)(x0 + i);
            const float4 xb0 = *(const float4*)(x0 + i + 4);
            const float4 xa1 = *(const float4*)(x1 + i);
            const float4 xb1 = *(const float4*)(x1 + i + 4);
            const float4 xa2 = *(const float4*)(x2 + i);
            const float4 xb2 = *(const float4*)(x2 + i + 4);
            const float4 xa3 = *(const float4*)(x3 + i);
            const float4 xb3 = *(const float4*)(x3 + i + 4);

            const float w0 = HF_BF16_LO(wv.x);
            const float w1 = HF_BF16_HI(wv.x);
            const float w2 = HF_BF16_LO(wv.y);
            const float w3 = HF_BF16_HI(wv.y);
            const float w4 = HF_BF16_LO(wv.z);
            const float w5 = HF_BF16_HI(wv.z);
            const float w6 = HF_BF16_LO(wv.w);
            const float w7 = HF_BF16_HI(wv.w);

            a00 += w0 * xa0.x;
            a01 += w1 * xa0.y;
            a02 += w2 * xa0.z;
            a03 += w3 * xa0.w;
            a00 += w4 * xb0.x;
            a01 += w5 * xb0.y;
            a02 += w6 * xb0.z;
            a03 += w7 * xb0.w;

            a10 += w0 * xa1.x;
            a11 += w1 * xa1.y;
            a12 += w2 * xa1.z;
            a13 += w3 * xa1.w;
            a10 += w4 * xb1.x;
            a11 += w5 * xb1.y;
            a12 += w6 * xb1.z;
            a13 += w7 * xb1.w;

            a20 += w0 * xa2.x;
            a21 += w1 * xa2.y;
            a22 += w2 * xa2.z;
            a23 += w3 * xa2.w;
            a20 += w4 * xb2.x;
            a21 += w5 * xb2.y;
            a22 += w6 * xb2.z;
            a23 += w7 * xb2.w;

            a30 += w0 * xa3.x;
            a31 += w1 * xa3.y;
            a32 += w2 * xa3.z;
            a33 += w3 * xa3.w;
            a30 += w4 * xb3.x;
            a31 += w5 * xb3.y;
            a32 += w6 * xb3.z;
            a33 += w7 * xb3.w;
        }
    }

    // Match gemv_bf16_xf32's scalar tail for K % 8 != 0 (and its complete
    // traversal when the wide path is unavailable).
    for (int i = ((K & 7) == 0 ? (K & ~7) : 0) + lane; i < K; i += 32) {
        const float weight = __uint_as_float(((unsigned int)w_row[i]) << 16);
        a00 += weight * x0[i];
        a10 += weight * x1[i];
        a20 += weight * x2[i];
        a30 += weight * x3[i];
    }

    float s0 = (a00 + a01) + (a02 + a03);
    float s1 = (a10 + a11) + (a12 + a13);
    float s2 = (a20 + a21) + (a22 + a23);
    float s3 = (a30 + a31) + (a32 + a33);
    for (int offset = 16; offset > 0; offset >>= 1) {
        s0 += __shfl_down(s0, offset, 32);
        s1 += __shfl_down(s1, offset, 32);
        s2 += __shfl_down(s2, offset, 32);
        s3 += __shfl_down(s3, offset, 32);
    }

    if (lane == 0) {
        Y[(size_t)b0 * M + row] = s0;
        if (batch_base + 1 < N) Y[(size_t)b1 * M + row] = s1;
        if (batch_base + 2 < N) Y[(size_t)b2 * M + row] = s2;
        if (batch_base + 3 < N) Y[(size_t)b3 * M + row] = s3;
    }
}

// gfx1151 candidate: one wave owns one output-channel row and eight adjacent
// token rows. This is intentionally a separate symbol so the measured route
// can fall back to gemm_bf16_xf32_multirow without changing its contract.
// Every token row keeps the four accumulator streams, scalar K traversal,
// pairwise reduction, and wave32 shuffle of the four-row reference kernel.
__launch_bounds__(32, 8)
extern "C" __global__ void gemm_bf16_xf32_multirow_n8_gfx1151(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int M,
    int K,
    int N
) {
    const int row = blockIdx.x;
    const int batch_base = blockIdx.y * 8;
    const int lane = threadIdx.x;
    if (row >= M || lane >= 32 || batch_base >= N) return;

    const unsigned short* w_row = W + (size_t)row * K;
    const int b0 = batch_base;
    const int b1 = (batch_base + 1 < N) ? batch_base + 1 : b0;
    const int b2 = (batch_base + 2 < N) ? batch_base + 2 : b0;
    const int b3 = (batch_base + 3 < N) ? batch_base + 3 : b0;
    const int b4 = (batch_base + 4 < N) ? batch_base + 4 : b0;
    const int b5 = (batch_base + 5 < N) ? batch_base + 5 : b0;
    const int b6 = (batch_base + 6 < N) ? batch_base + 6 : b0;
    const int b7 = (batch_base + 7 < N) ? batch_base + 7 : b0;
    const float* x0 = X + (size_t)b0 * K;
    const float* x1 = X + (size_t)b1 * K;
    const float* x2 = X + (size_t)b2 * K;
    const float* x3 = X + (size_t)b3 * K;
    const float* x4 = X + (size_t)b4 * K;
    const float* x5 = X + (size_t)b5 * K;
    const float* x6 = X + (size_t)b6 * K;
    const float* x7 = X + (size_t)b7 * K;

    float a00 = 0.0f, a01 = 0.0f, a02 = 0.0f, a03 = 0.0f;
    float a10 = 0.0f, a11 = 0.0f, a12 = 0.0f, a13 = 0.0f;
    float a20 = 0.0f, a21 = 0.0f, a22 = 0.0f, a23 = 0.0f;
    float a30 = 0.0f, a31 = 0.0f, a32 = 0.0f, a33 = 0.0f;
    float a40 = 0.0f, a41 = 0.0f, a42 = 0.0f, a43 = 0.0f;
    float a50 = 0.0f, a51 = 0.0f, a52 = 0.0f, a53 = 0.0f;
    float a60 = 0.0f, a61 = 0.0f, a62 = 0.0f, a63 = 0.0f;
    float a70 = 0.0f, a71 = 0.0f, a72 = 0.0f, a73 = 0.0f;

    if ((K & 7) == 0) {
        for (int i = lane * 8; i + 8 <= K; i += 32 * 8) {
            const uint4 wv = *(const uint4*)(w_row + i);
            const float4 xa0 = *(const float4*)(x0 + i);
            const float4 xb0 = *(const float4*)(x0 + i + 4);
            const float4 xa1 = *(const float4*)(x1 + i);
            const float4 xb1 = *(const float4*)(x1 + i + 4);
            const float4 xa2 = *(const float4*)(x2 + i);
            const float4 xb2 = *(const float4*)(x2 + i + 4);
            const float4 xa3 = *(const float4*)(x3 + i);
            const float4 xb3 = *(const float4*)(x3 + i + 4);
            const float4 xa4 = *(const float4*)(x4 + i);
            const float4 xb4 = *(const float4*)(x4 + i + 4);
            const float4 xa5 = *(const float4*)(x5 + i);
            const float4 xb5 = *(const float4*)(x5 + i + 4);
            const float4 xa6 = *(const float4*)(x6 + i);
            const float4 xb6 = *(const float4*)(x6 + i + 4);
            const float4 xa7 = *(const float4*)(x7 + i);
            const float4 xb7 = *(const float4*)(x7 + i + 4);

            const float w0 = HF_BF16_LO(wv.x);
            const float w1 = HF_BF16_HI(wv.x);
            const float w2 = HF_BF16_LO(wv.y);
            const float w3 = HF_BF16_HI(wv.y);
            const float w4 = HF_BF16_LO(wv.z);
            const float w5 = HF_BF16_HI(wv.z);
            const float w6 = HF_BF16_LO(wv.w);
            const float w7 = HF_BF16_HI(wv.w);

            a00 += w0 * xa0.x; a01 += w1 * xa0.y; a02 += w2 * xa0.z; a03 += w3 * xa0.w;
            a00 += w4 * xb0.x; a01 += w5 * xb0.y; a02 += w6 * xb0.z; a03 += w7 * xb0.w;
            a10 += w0 * xa1.x; a11 += w1 * xa1.y; a12 += w2 * xa1.z; a13 += w3 * xa1.w;
            a10 += w4 * xb1.x; a11 += w5 * xb1.y; a12 += w6 * xb1.z; a13 += w7 * xb1.w;
            a20 += w0 * xa2.x; a21 += w1 * xa2.y; a22 += w2 * xa2.z; a23 += w3 * xa2.w;
            a20 += w4 * xb2.x; a21 += w5 * xb2.y; a22 += w6 * xb2.z; a23 += w7 * xb2.w;
            a30 += w0 * xa3.x; a31 += w1 * xa3.y; a32 += w2 * xa3.z; a33 += w3 * xa3.w;
            a30 += w4 * xb3.x; a31 += w5 * xb3.y; a32 += w6 * xb3.z; a33 += w7 * xb3.w;
            a40 += w0 * xa4.x; a41 += w1 * xa4.y; a42 += w2 * xa4.z; a43 += w3 * xa4.w;
            a40 += w4 * xb4.x; a41 += w5 * xb4.y; a42 += w6 * xb4.z; a43 += w7 * xb4.w;
            a50 += w0 * xa5.x; a51 += w1 * xa5.y; a52 += w2 * xa5.z; a53 += w3 * xa5.w;
            a50 += w4 * xb5.x; a51 += w5 * xb5.y; a52 += w6 * xb5.z; a53 += w7 * xb5.w;
            a60 += w0 * xa6.x; a61 += w1 * xa6.y; a62 += w2 * xa6.z; a63 += w3 * xa6.w;
            a60 += w4 * xb6.x; a61 += w5 * xb6.y; a62 += w6 * xb6.z; a63 += w7 * xb6.w;
            a70 += w0 * xa7.x; a71 += w1 * xa7.y; a72 += w2 * xa7.z; a73 += w3 * xa7.w;
            a70 += w4 * xb7.x; a71 += w5 * xb7.y; a72 += w6 * xb7.z; a73 += w7 * xb7.w;
        }
    }

    for (int i = ((K & 7) == 0 ? (K & ~7) : 0) + lane; i < K; i += 32) {
        const float weight = __uint_as_float(((unsigned int)w_row[i]) << 16);
        a00 += weight * x0[i]; a10 += weight * x1[i];
        a20 += weight * x2[i]; a30 += weight * x3[i];
        a40 += weight * x4[i]; a50 += weight * x5[i];
        a60 += weight * x6[i]; a70 += weight * x7[i];
    }

    float s0 = (a00 + a01) + (a02 + a03);
    float s1 = (a10 + a11) + (a12 + a13);
    float s2 = (a20 + a21) + (a22 + a23);
    float s3 = (a30 + a31) + (a32 + a33);
    float s4 = (a40 + a41) + (a42 + a43);
    float s5 = (a50 + a51) + (a52 + a53);
    float s6 = (a60 + a61) + (a62 + a63);
    float s7 = (a70 + a71) + (a72 + a73);
    for (int offset = 16; offset > 0; offset >>= 1) {
        s0 += __shfl_down(s0, offset, 32); s1 += __shfl_down(s1, offset, 32);
        s2 += __shfl_down(s2, offset, 32); s3 += __shfl_down(s3, offset, 32);
        s4 += __shfl_down(s4, offset, 32); s5 += __shfl_down(s5, offset, 32);
        s6 += __shfl_down(s6, offset, 32); s7 += __shfl_down(s7, offset, 32);
    }
    if (lane == 0) {
        Y[(size_t)b0 * M + row] = s0;
        if (batch_base + 1 < N) Y[(size_t)b1 * M + row] = s1;
        if (batch_base + 2 < N) Y[(size_t)b2 * M + row] = s2;
        if (batch_base + 3 < N) Y[(size_t)b3 * M + row] = s3;
        if (batch_base + 4 < N) Y[(size_t)b4 * M + row] = s4;
        if (batch_base + 5 < N) Y[(size_t)b5 * M + row] = s5;
        if (batch_base + 6 < N) Y[(size_t)b6 * M + row] = s6;
        if (batch_base + 7 < N) Y[(size_t)b7 * M + row] = s7;
    }
}
