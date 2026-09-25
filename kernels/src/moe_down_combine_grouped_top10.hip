// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — fixed top-10 BF16 grouped-down combine.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ __hip_bfloat16 moe_top10_bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float moe_top10_bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

// `inverse_perm[token * 10 + rank]` maps a route slot to the sorted grouped
// row.  The grouped path intentionally mirrors
// moe_down_combine_top10_batched: route ranks are selected by ascending expert
// ID, and residual, weight, output, product, and every accumulation are each
// rounded through the source BF16 contract before the next rank.
static __device__ __forceinline__ float moe_down_widen(float v) { return v; }
static __device__ __forceinline__ float moe_down_widen(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

template <typename T>
__device__ __forceinline__ void moe_down_combine_grouped(
    const T* __restrict__ grouped_down,     // [grouped_rows * H], unweighted
    const int* __restrict__ inverse_perm,    // [N * 10], flat -> grouped row
    const int* __restrict__ topk_indices,     // [N * 10], expert IDs
    const float* __restrict__ topk_weights,   // [N * 10]
    float* __restrict__ residual,             // [N * H], in-place +=
    int hidden,
    int grouped_rows,
    int tokens
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (token >= tokens || column >= hidden) return;

    const size_t route_base = (size_t)token * 10;
    __hip_bfloat16 accumulated = moe_top10_bf16_round(
        residual[(size_t)token * hidden + column]);
    int used_ranks = 0;
    for (int count = 0; count < 10; ++count) {
        int selected_rank = -1;
        int selected_expert = 0x7fffffff;
        for (int rank = 0; rank < 10; ++rank) {
            if ((used_ranks & (1 << rank)) != 0) continue;
            const int expert = topk_indices[route_base + rank];
            if (expert < selected_expert) {
                selected_expert = expert;
                selected_rank = rank;
            }
        }
        if (selected_rank < 0) break;
        used_ranks |= 1 << selected_rank;

        const __hip_bfloat16 weight =
            moe_top10_bf16_round(topk_weights[route_base + selected_rank]);
        const int grouped_row = inverse_perm[route_base + selected_rank];
        const float raw_output =
            (grouped_row >= 0 && grouped_row < grouped_rows)
                ? moe_down_widen(grouped_down[(size_t)grouped_row * hidden + column])
                : 0.0f;
        const __hip_bfloat16 output = moe_top10_bf16_round(raw_output);
        const __hip_bfloat16 weighted = moe_top10_bf16_round(
            moe_top10_bf16_to_f32(output) * moe_top10_bf16_to_f32(weight));
        accumulated = moe_top10_bf16_round(
            moe_top10_bf16_to_f32(accumulated) + moe_top10_bf16_to_f32(weighted));
    }
    residual[(size_t)token * hidden + column] =
        moe_top10_bf16_to_f32(accumulated);
}

__launch_bounds__(256, 4)
extern "C" __global__ void moe_down_combine_grouped_top10(
    const float* __restrict__ grouped_down, const int* __restrict__ inverse_perm,
    const int* __restrict__ topk_indices, const float* __restrict__ topk_weights,
    float* __restrict__ residual, int hidden, int grouped_rows, int tokens) {
    moe_down_combine_grouped(grouped_down, inverse_perm, topk_indices, topk_weights,
                             residual, hidden, grouped_rows, tokens);
}

// Grouped down stored as BF16 bits (the values the BF16 round above yields),
// eight columns per thread: the ten route ranks' order (ascending expert ID,
// block-uniform) is resolved first, every rank's 16-byte slice is loaded
// before the in-order accumulation, and each column runs the exact
// moe_down_combine_grouped arithmetic.  Grid [ceil(hidden / 512), tokens],
// block 64, hidden % 8 == 0.
extern "C" __global__ __launch_bounds__(64) void moe_down_combine_grouped_top10_bf16in(
    const unsigned short* __restrict__ grouped_down, const int* __restrict__ inverse_perm,
    const int* __restrict__ topk_indices, const float* __restrict__ topk_weights,
    float* __restrict__ residual, int hidden, int grouped_rows, int tokens) {
    const int token = blockIdx.y;
    const int col0 = (blockIdx.x * 64 + threadIdx.x) * 8;
    if (token >= tokens || col0 >= hidden) return;
    const size_t route_base = (size_t)token * 10;
    int order[10];
    int used_ranks = 0, n = 0;
    for (int count = 0; count < 10; ++count) {
        int selected_rank = -1, selected_expert = 0x7fffffff;
        for (int rank = 0; rank < 10; ++rank) {
            if ((used_ranks & (1 << rank)) != 0) continue;
            const int expert = topk_indices[route_base + rank];
            if (expert < selected_expert) { selected_expert = expert; selected_rank = rank; }
        }
        if (selected_rank < 0) break;
        used_ranks |= 1 << selected_rank;
        order[n++] = selected_rank;
    }
    uint4 rows[10];
    float wts[10];
#pragma unroll
    for (int c = 0; c < 10; ++c) {
        rows[c] = make_uint4(0, 0, 0, 0);
        wts[c] = 0.0f;
        if (c < n) {
            const int rank = order[c];
            wts[c] = moe_top10_bf16_to_f32(moe_top10_bf16_round(topk_weights[route_base + rank]));
            const int grouped_row = inverse_perm[route_base + rank];
            if (grouped_row >= 0 && grouped_row < grouped_rows)
                rows[c] = *(const uint4*)(grouped_down + (size_t)grouped_row * hidden + col0);
        }
    }
    float* res = residual + (size_t)token * hidden + col0;
    const float4 r0 = *(const float4*)res, r1 = *(const float4*)(res + 4);
    float acc[8] = {r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w};
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        __hip_bfloat16 accumulated = moe_top10_bf16_round(acc[e]);
#pragma unroll
        for (int c = 0; c < 10; ++c) {
            if (c >= n) break;
            const unsigned int w = (&rows[c].x)[e >> 1];
            const float raw_output = __uint_as_float((e & 1) ? (w & 0xFFFF0000u) : (w << 16));
            const __hip_bfloat16 output = moe_top10_bf16_round(raw_output);
            const __hip_bfloat16 weighted = moe_top10_bf16_round(
                moe_top10_bf16_to_f32(output) * wts[c]);
            accumulated = moe_top10_bf16_round(
                moe_top10_bf16_to_f32(accumulated) + moe_top10_bf16_to_f32(weighted));
        }
        acc[e] = moe_top10_bf16_to_f32(accumulated);
    }
    *(float4*)res = make_float4(acc[0], acc[1], acc[2], acc[3]);
    *(float4*)(res + 4) = make_float4(acc[4], acc[5], acc[6], acc[7]);
}
