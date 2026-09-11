// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//
// Qwen4 ordinary-HIP correctness kernels.  All accumulations are F32.  The
// host side owns bounded arenas and passes explicit active lengths; no kernel
// performs host I/O or allocates memory.

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

// q/k contain 16 heads of width 128, v contains 48 heads of width 128.  Each
// value head repeats one q/k head three times.  state is [48, value_dim, key_dim].
extern "C" __global__ void qwen4_gdn_step_f32(
    const float* q,
    const float* k,
    const float* v,
    const float* gate,
    const float* beta,
    float* state,
    float* output,
    int key_heads,
    int value_heads,
    int key_dim,
    int value_dim) {
    int value_head = (int)blockIdx.x;
    int value_index = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (value_head >= value_heads || value_index >= value_dim) return;
    int repeat = value_heads / key_heads;
    int key_head = value_head / repeat;
    float gate_value = 1.0f / (1.0f + expf(-gate[value_head]));
    float beta_value = beta[value_head];
    int state_row = (value_head * value_dim + value_index) * key_dim;
    float acc = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
        float qv = q[key_head * key_dim + kd];
        float kv = k[key_head * key_dim + kd];
        float next = gate_value * state[state_row + kd] + beta_value * v[value_head * value_dim + value_index] * kv;
        state[state_row + kd] = next;
        acc += next * qv;
    }
    output[value_head * value_dim + value_index] = acc / (1.0f + expf(-acc));
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

// F32 block-of-four average and L2 normalization.  RoPE remains a separate
// launch so block-start positions are explicit and testable.
extern "C" __global__ void qwen4_qsa_pool_f32(
    const float* raw_keys,
    float* pooled,
    int block_count,
    int head_dim) {
    int block = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (block >= block_count || channel >= head_dim) return;
    float value = 0.0f;
    for (int row = 0; row < 4; ++row) {
        value += raw_keys[(block * 4 + row) * head_dim + channel];
    }
    pooled[block * head_dim + channel] = value * 0.25f;
}
extern "C" __global__ void qwen4_qsa_attention_f32(
    const float* q_with_gate,
    const float* full_keys,
    const float* full_values,
    const float* selected_indices,
    float* output,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int selected_len,
    int full_capacity) {
    int head = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (head >= n_heads || channel >= head_dim) return;
    int kv_head = head / (n_heads / n_kv_heads);
    float max_score = -3.402823466e+38f;
    for (int row = 0; row < selected_len; ++row) {
        int token = (int)selected_indices[row];
        if (token < 0 || token >= full_capacity) continue;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_with_gate[head * head_dim * 2 + d]
                * full_keys[token * n_kv_heads * head_dim + kv_head * head_dim + d];
        }
        max_score = fmaxf(max_score, score / sqrtf((float)head_dim));
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (int row = 0; row < selected_len; ++row) {
        int token = (int)selected_indices[row];
        if (token < 0 || token >= full_capacity) continue;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_with_gate[head * head_dim * 2 + d]
                * full_keys[token * n_kv_heads * head_dim + kv_head * head_dim + d];
        }
        float weight = expf(score / sqrtf((float)head_dim) - max_score);
        denominator += weight;
        numerator += weight * full_values[token * n_kv_heads * head_dim + kv_head * head_dim + channel];
    }
    float gate = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        gate += q_with_gate[head * head_dim * 2 + head_dim + d];
    }
    gate = 1.0f / (1.0f + expf(-gate / (float)head_dim));
    output[head * head_dim + channel] = denominator > 0.0f ? gate * numerator / denominator : 0.0f;
}
