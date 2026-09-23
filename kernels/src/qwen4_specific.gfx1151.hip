// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Q/K input values are source BF16 activations.  The recurrent state and
// projections remain F32, but source l2norm intentionally performs its
// multiply, reduction, rsqrt, and output multiply in BF16 before promotion.
// Keep these helpers explicit so the boundary is not widened by F32 operators.
// The host side owns bounded arenas and passes explicit active lengths; no kernel
// performs host I/O or allocates memory.

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

using __hip_bfloat16 = hip_bfloat16;

static __device__ __forceinline__ float __bfloat162float(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

static __device__ __forceinline__ __hip_bfloat16 qwen4_bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ __hip_bfloat16 qwen4_bf16_add(
    __hip_bfloat16 lhs,
    __hip_bfloat16 rhs) {
    return qwen4_bf16_round(__bfloat162float(lhs) + __bfloat162float(rhs));
}

extern "C" __global__ void qwen4_hc_prepare_f32(
    const float* x,
    const float* branch_weight,
    float* normalized,
    float* branch,
    int hidden,
    float inv_rms) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= hidden) return;
    normalized[i] = x[i] * inv_rms;
    branch[i] = normalized[i] * (1.0f + branch_weight[i]);
}

extern "C" __global__ void qwen4_hc_mix_f32(
    const float* branches,
    float* output,
    int branch_count,
    int hidden) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= hidden) return;
    float sum = 0.0f;
    for (int branch = 0; branch < branch_count; ++branch) {
        sum += branches[branch * hidden + i];
    }
    output[i] = sum / (float)branch_count;
}

extern "C" __global__ void qwen4_ple_depthwise_f32(
    const float* input,
    const float* history,
    const float* kernel,
    float* output,
    float* next_history,
    int channels,
    int history_rows,
    int kernel_size,
    int dilation,
    int cursor) {
    int channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= channels) return;
    float value = kernel[channel] * input[channel];
    for (int tap = 1; tap < kernel_size; ++tap) {
        int delay = tap * dilation;
        int row = (cursor + history_rows - (delay % history_rows)) % history_rows;
        value += kernel[tap * channels + channel] * history[row * channels + channel];
    }
    output[channel] = value;
    for (int row = 0; row < history_rows; ++row) {
        next_history[row * channels + channel] = history[row * channels + channel];
    }
    next_history[cursor * channels + channel] = input[channel];
}
