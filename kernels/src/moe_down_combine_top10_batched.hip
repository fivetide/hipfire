// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — Qwen4 fixed top-10 expanded-down combine.

#include <hip/hip_runtime.h>

// `expert_outputs` is deliberately unweighted [N * 10 * H].  This kernel is
// the sole routed-weight application: each expanded row is multiplied exactly
// once, then accumulated into the token residual.
__launch_bounds__(256, 4)
extern "C" __global__ void moe_down_combine_top10_batched(
    const float* __restrict__ expert_outputs, // [N * 10 * H]
    const float* __restrict__ topk_weights,    // [N * 10]
    float* __restrict__ residual,              // [N * H], in-place +=
    int hidden,
    int tokens
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (token >= tokens || column >= hidden) return;

    float sum = 0.0f;
    const size_t token_slot = (size_t)token * 10;
    for (int rank = 0; rank < 10; ++rank) {
        sum += topk_weights[token_slot + rank] *
               expert_outputs[(token_slot + rank) * hidden + column];
    }
    residual[(size_t)token * hidden + column] += sum;
}
