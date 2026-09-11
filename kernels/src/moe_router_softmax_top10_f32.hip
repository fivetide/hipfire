// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — Qwen4's fixed 512-way top-10 router.

#include <hip/hip_runtime.h>
#include <math.h>

// One workgroup handles one token.  The Qwen4 router has exactly 512 experts,
// so one lane owns one logit and thread 0 performs the deterministic selection.
// This is intentionally a new symbol: the incumbent k=8 router is not widened.
//
// topk_idx is an i32 bit-pattern stored in the project's F32-typed scratch
// convention.  `normalize_topk_prob != 0` applies the Qwen4 selected-probability
// renormalization after the float32 softmax.
__launch_bounds__(512, 1)
extern "C" __global__ void moe_router_softmax_top10_f32(
    const float* __restrict__ logits, // [tokens, n_exp]
    int* __restrict__ topk_idx,       // [tokens, 10]
    float* __restrict__ topk_w,       // [tokens, 10]
    int n_exp,
    int top_k,
    int normalize_topk_prob
) {
    __shared__ float values[512];
    const int lane = threadIdx.x;
    const int token = blockIdx.x;

    if (lane < n_exp && lane < 512)
        values[lane] = logits[(size_t)token * n_exp + lane];
    else if (lane < 512)
        values[lane] = -INFINITY;
    __syncthreads();

    if (lane != 0) return;

    // Stable float32 softmax.  The host-side seal fixes n_exp=512/top_k=10;
    // the guards keep this kernel bounds-safe if a caller violates that
    // contract before dispatch.
    if (n_exp <= 0 || n_exp > 512 || top_k <= 0 || top_k > 10) return;
    float max_logit = -INFINITY;
    for (int e = 0; e < n_exp; ++e) {
        if (values[e] > max_logit) max_logit = values[e];
    }
    float denom = 0.0f;
    for (int e = 0; e < n_exp; ++e) {
        values[e] = expf(values[e] - max_logit);
        denom += values[e];
    }
    if (!(denom > 0.0f) || !isfinite(denom)) return;
    const float inv_denom = 1.0f / denom;
    for (int e = 0; e < n_exp; ++e) values[e] *= inv_denom;

    bool selected[512];
    for (int e = 0; e < n_exp; ++e) selected[e] = false;
    for (int rank = 0; rank < 10; ++rank) {
        int best = 0;
        float best_value = -INFINITY;
        bool found = false;
        if (rank < top_k) {
            for (int e = 0; e < n_exp; ++e) {
                const float value = values[e];
                // Strict value ordering plus lower-index tie breaking matches
                // the reference router and is independent of wave scheduling.
                if (selected[e]) continue;
                if (!found || value > best_value ||
                    (value == best_value && e < best)) {
                    best = e;
                    best_value = value;
                    found = true;
                }
            }
            selected[best] = true;
            topk_idx[(size_t)token * 10 + rank] = best;
            topk_w[(size_t)token * 10 + rank] = best_value;
        } else {
            // The fixed Qwen4 path never takes this arm, but zeroing the tail
            // prevents stale route data from becoming a hidden fallback.
            topk_idx[(size_t)token * 10 + rank] = 0;
            topk_w[(size_t)token * 10 + rank] = 0.0f;
        }
    }

    if (normalize_topk_prob != 0) {
        float selected_sum = 0.0f;
        for (int rank = 0; rank < top_k; ++rank)
            selected_sum += topk_w[(size_t)token * 10 + rank];
        if (selected_sum > 0.0f && isfinite(selected_sum)) {
            const float inv_selected = 1.0f / selected_sum;
            for (int rank = 0; rank < top_k; ++rank)
                topk_w[(size_t)token * 10 + rank] *= inv_selected;
        }
    }
}
