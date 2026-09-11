// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — Qwen4 fixed top-10 grouped-down combine.

#include <hip/hip_runtime.h>

// `inverse_perm[token * 10 + rank]` maps a route slot to the sorted grouped
// row.  Padding rows are never selected; the explicit bounds check keeps a
// malformed permutation from turning into an out-of-bounds read.
__launch_bounds__(256, 4)
extern "C" __global__ void moe_down_combine_grouped_top10(
    const float* __restrict__ grouped_down, // [grouped_rows * H], unweighted
    const int* __restrict__ inverse_perm,    // [N * 10], flat -> grouped row
    const float* __restrict__ topk_weights,  // [N * 10]
    float* __restrict__ residual,            // [N * H], in-place +=
    int hidden,
    int grouped_rows,
    int tokens
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (token >= tokens || column >= hidden) return;

    float sum = 0.0f;
    const size_t route_base = (size_t)token * 10;
    for (int rank = 0; rank < 10; ++rank) {
        const int grouped_row = inverse_perm[route_base + rank];
        if (grouped_row >= 0 && grouped_row < grouped_rows) {
            sum += topk_weights[route_base + rank] *
                   grouped_down[(size_t)grouped_row * hidden + column];
        }
    }
    residual[(size_t)token * hidden + column] += sum;
}
