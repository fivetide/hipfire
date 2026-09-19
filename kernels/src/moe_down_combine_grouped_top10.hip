// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — Qwen4 fixed top-10 grouped-down combine.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ __hip_bfloat16 qwen4_bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ float qwen4_bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

// `inverse_perm[token * 10 + rank]` maps a route slot to the sorted grouped
// row.  The grouped path intentionally mirrors
// moe_down_combine_top10_batched: route ranks are selected by ascending expert
// ID, and residual, weight, output, product, and every accumulation are each
// rounded through the source BF16 contract before the next rank.
__launch_bounds__(256, 4)
extern "C" __global__ void moe_down_combine_grouped_top10(
    const float* __restrict__ grouped_down, // [grouped_rows * H], unweighted
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
    __hip_bfloat16 accumulated = qwen4_bf16_round(
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
            qwen4_bf16_round(topk_weights[route_base + selected_rank]);
        const int grouped_row = inverse_perm[route_base + selected_rank];
        const float raw_output =
            (grouped_row >= 0 && grouped_row < grouped_rows)
                ? grouped_down[(size_t)grouped_row * hidden + column]
                : 0.0f;
        const __hip_bfloat16 output = qwen4_bf16_round(raw_output);
        const __hip_bfloat16 weighted = qwen4_bf16_round(
            qwen4_bf16_to_f32(output) * qwen4_bf16_to_f32(weight));
        accumulated = qwen4_bf16_round(
            qwen4_bf16_to_f32(accumulated) + qwen4_bf16_to_f32(weighted));
    }
    residual[(size_t)token * hidden + column] =
        qwen4_bf16_to_f32(accumulated);
}
