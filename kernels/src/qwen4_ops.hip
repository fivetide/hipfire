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

// q/k/v are source BF16 activations promoted to the explicit F32 recurrent
// state.  Each value head repeats one q/k head.  The source l2norm call runs
// before that promotion, so its intermediate products, reduction, rsqrt, and
// output are BF16.  Query scale is a separate F32 multiply after promotion.
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
    int value_dim,
    float query_scale) {
    int value_head = (int)blockIdx.x;
    int value_index = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (value_head >= value_heads || value_index >= value_dim) return;
    int repeat = value_heads / key_heads;
    int key_head = value_head / repeat;
    const int key_base = key_head * key_dim;
    float q_sum_f32 = 0.0f;
    float k_sum_f32 = 0.0f;
    const __hip_bfloat16 eps = qwen4_bf16_round(1.0e-6f);
    for (int kd = 0; kd < key_dim; ++kd) {
        const __hip_bfloat16 q_value = qwen4_bf16_round(q[key_base + kd]);
        const __hip_bfloat16 k_value = qwen4_bf16_round(k[key_base + kd]);
        const float q_float = __bfloat162float(q_value);
        const float k_float = __bfloat162float(k_value);
        q_sum_f32 += __bfloat162float(qwen4_bf16_round(q_float * q_float));
        k_sum_f32 += __bfloat162float(qwen4_bf16_round(k_float * k_float));
    }
    const __hip_bfloat16 q_sum = qwen4_bf16_round(q_sum_f32);
    const __hip_bfloat16 k_sum = qwen4_bf16_round(k_sum_f32);
    const __hip_bfloat16 q_inv = qwen4_bf16_round(
        rsqrtf(__bfloat162float(q_sum) + __bfloat162float(eps)));
    const __hip_bfloat16 k_inv = qwen4_bf16_round(
        rsqrtf(__bfloat162float(k_sum) + __bfloat162float(eps)));
    float decay = expf(gate[value_head]);
    float beta_value = beta[value_head];
    int state_base = value_head * key_dim * value_dim;
    for (int kd = 0; kd < key_dim; ++kd) {
        state[state_base + kd * value_dim + value_index] *= decay;
    }
    float kv_mem = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
        const __hip_bfloat16 k_value = qwen4_bf16_round(k[key_base + kd]);
        const __hip_bfloat16 k_normalized = qwen4_bf16_round(
            __bfloat162float(k_value) * __bfloat162float(k_inv));
        float kv = __bfloat162float(k_normalized);
        kv_mem += state[state_base + kd * value_dim + value_index] * kv;
    }
    const __hip_bfloat16 v_value = qwen4_bf16_round(
        v[value_head * value_dim + value_index]);
    float delta = (__bfloat162float(v_value) - kv_mem) * beta_value;
    for (int kd = 0; kd < key_dim; ++kd) {
        const __hip_bfloat16 k_value = qwen4_bf16_round(k[key_base + kd]);
        const __hip_bfloat16 k_normalized = qwen4_bf16_round(
            __bfloat162float(k_value) * __bfloat162float(k_inv));
        float kv = __bfloat162float(k_normalized);
        state[state_base + kd * value_dim + value_index] += kv * delta;
    }
    float out = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
        const __hip_bfloat16 q_value = qwen4_bf16_round(q[key_base + kd]);
        const __hip_bfloat16 q_normalized = qwen4_bf16_round(
            __bfloat162float(q_value) * __bfloat162float(q_inv));
        float qv = __bfloat162float(q_normalized);
        qv *= query_scale;
        out += state[state_base + kd * value_dim + value_index] * qv;
    }
    output[value_head * value_dim + value_index] = out;
}

// Source HF executes the recurrent/chunk result through a hidden BF16 cast
// before RMSNormGated.  Keep the compact production buffer F32 for the
// existing gate kernel while forcing the exact bounded BF16 storage boundary
// through a caller-owned device scratch tensor.
extern "C" __global__ void qwen4_gdn_bf16_roundtrip_f32(
    const float* input,
    __hip_bfloat16* scratch,
    float* output,
    int elements) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= elements) return;
    __hip_bfloat16 rounded = static_cast<__hip_bfloat16>(input[i]);
    scratch[i] = rounded;
    output[i] = __bfloat162float(scratch[i]);
}

// Learned HC read/mix. `norm_weight` is resident BF16; down/up projections
// have already run through the native quantized GEMV path.
extern "C" __global__ void qwen4_hc_read_f32(
    const float* input,
    const __hip_bfloat16* norm_weight,
    const float* low,
    const float* up,
    float* normalized,
    float* mixed,
    int branches,
    int hidden,
    int rank) {
    int column = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int wide = branches * hidden;
    if (column >= wide) return;
    int branch = column / hidden;
    int branch_start = branch * hidden;
    float sum = 0.0f;
    for (int i = 0; i < hidden; ++i) {
        float value = input[branch_start + i];
        sum += value * value;
    }
    float inv = rsqrtf(sum / (float)hidden + 1.0e-6f);
    float normed = input[column] * inv * (1.0f + __bfloat162float(norm_weight[column]));
    normalized[column] = normed;
    if (column < hidden) {
        float value = 0.0f;
        for (int branch = 0; branch < branches; ++branch) {
            int offset = branch * hidden + column;
            float gate = 0.0f;
            for (int r = 0; r < rank; ++r) {
                gate += up[offset * rank + r] * low[r];
            }
            gate = 1.0f / (1.0f + expf(-gate));
            value += gate * normalized[offset];
        }
        mixed[column] = value / (float)branches;
    }
}

extern "C" __global__ void qwen4_hc_write_f32(
    const float* input,
    const float* normalized,
    const float* mixed,
    const float* gates,
    float* output,
    int branches,
    int hidden) {
    int column = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int wide = branches * hidden;
    if (column >= wide) return;
    int branch = column / hidden;
    float gate = 2.0f / (1.0f + expf(-gates[branch] / (float)branches));
    output[column] = input[column] + gate * mixed[column % hidden];
}

extern "C" __global__ void qwen4_hc_final_f32(
    const float* normalized,
    float* output,
    int branches,
    int hidden) {
    int column = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (column >= hidden) return;
    float sum = 0.0f;
    for (int branch = 0; branch < branches; ++branch) {
        sum += normalized[branch * hidden + column];
    }
    output[column] = sum / (float)branches;
}
extern "C" __global__ void qwen4_hc_norm_f32(
    const float* input,
    const __hip_bfloat16* norm_weight,
    float* normalized,
    int branches,
    int hidden) {
    int column = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int wide = branches * hidden;
    if (column >= wide) return;
    int branch = column / hidden;
    int branch_start = branch * hidden;
    float sum = 0.0f;
    for (int i = 0; i < hidden; ++i) {
        float value = input[branch_start + i];
        sum += value * value;
    }
    float inv = rsqrtf(sum / (float)hidden + 1.0e-6f);
    normalized[column] = input[column] * inv * (1.0f + __bfloat162float(norm_weight[column]));
}
extern "C" __global__ void qwen4_gdn_conv_bf16_f32(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    float* output,
    float* next_history,
    int channels,
    int history_rows,
    int kernel_size,
    int cursor) {
    int channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= channels) return;
    float value = __bfloat162float(kernel[channel]) * input[channel];
    for (int tap = 1; tap < kernel_size; ++tap) {
        int delay = tap;
        if (history_rows > 0) {
            int row = (cursor + history_rows - (delay % history_rows)) % history_rows;
            value += __bfloat162float(kernel[tap * channels + channel])
                * history[row * channels + channel];
        }
    }
    output[channel] = value * (1.0f / (1.0f + expf(-value)));
    for (int row = 0; row < history_rows; ++row) {
        next_history[row * channels + channel] = history[row * channels + channel];
    }
    if (history_rows > 0) {
        next_history[cursor * channels + channel] = input[channel];
    }
}

extern "C" __global__ void qwen4_gdn_params_bf16_f32(
    const float* a,
    const float* b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    float* gate,
    float* beta,
    int heads) {
    int head = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (head >= heads) return;
    float al = __bfloat162float(a_log[head]);
    float dt = __bfloat162float(dt_bias[head]);
    float value = a[head] + dt;
    float softplus = value > 20.0f ? value : (value < -20.0f ? expf(value) : log1pf(expf(value)));
    gate[head] = -expf(al) * softplus;
    beta[head] = 1.0f / (1.0f + expf(-b[head]));
}

extern "C" __global__ void qwen4_gdn_params_f32(
    const float* a,
    const float* b,
    const float* a_log,
    const float* dt_bias,
    float* gate,
    float* beta,
    int heads) {
    int head = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (head >= heads) return;
    float value = a[head] + dt_bias[head];
    float softplus = value > 20.0f
        ? value
        : (value < -20.0f ? expf(value) : log1pf(expf(value)));
    gate[head] = -expf(a_log[head]) * softplus;
    beta[head] = 1.0f / (1.0f + expf(-b[head]));
}

extern "C" __global__ void qwen4_gdn_gate_bf16_f32(
    const float* recurrent,
    const float* z,
    const __hip_bfloat16* norm,
    float* output,
    int value_heads,
    int value_dim) {
    int head = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (head >= value_heads || channel >= value_dim) return;
    int width = value_heads * value_dim;
    int start = head * value_dim;
    float sum = 0.0f;
    for (int i = 0; i < value_dim; ++i) {
        float value = recurrent[start + i];
        sum += value * value;
    }
    float inv = rsqrtf(sum / (float)value_dim + 1.0e-6f);
    float normalized = recurrent[start + channel] * inv
        * (1.0f + __bfloat162float(norm[channel]));
    // HF RMSNormGated casts the normalized hidden back to the input BF16
    // dtype before applying the output sigmoid gate.
    __hip_bfloat16 normalized_bf16 = static_cast<__hip_bfloat16>(normalized);
    float normalized_hidden = __bfloat162float(normalized_bf16);
    float gate = z[start + channel];
    output[start + channel] = normalized_hidden / (1.0f + expf(-gate));
    (void)width;
}
extern "C" __global__ void qwen4_qsa_norm_rope_f32(
    float* values,
    const __hip_bfloat16* norm,
    int heads,
    int head_dim,
    int position,
    int rotary_dim) {
    int head = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (head >= heads || channel >= head_dim) return;
    float sum = 0.0f;
    int start = head * head_dim;
    for (int i = 0; i < head_dim; ++i) {
        float value = values[start + i];
        sum += value * value;
    }
    float inv = rsqrtf(sum / (float)head_dim + 1.0e-6f);
    float value = values[start + channel] * inv
        * (1.0f + __bfloat162float(norm[channel]));
    if (channel < rotary_dim) {
        int half = rotary_dim / 2;
        int partner = channel < half ? channel + half : channel - half;
        // Recompute the partner from the immutable input row.  The old
        // implementation read `values[partner]` after another lane could
        // already have overwritten it with its normalized value, making the
        // result scheduling-dependent.
        float other = values[start + partner] * inv
            * (1.0f + __bfloat162float(norm[partner]));
        int base = channel < half ? channel : partner;
        float angle = (float)position
            * powf(1000000.0f, -((float)(2 * base)) / (float)rotary_dim);
        float sine = sinf(angle);
        float cosine = cosf(angle);
        value = channel < half ? value * cosine - other * sine
            : other * sine + value * cosine;
    }
    values[start + channel] = value;
}

extern "C" __global__ void qwen4_qsa_cache_append_f32(
    const float* key,
    const float* value,
    float* full_keys,
    float* full_values,
    int position,
    int kv_width) {
    int channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= kv_width) return;
    full_keys[position * kv_width + channel] = key[channel];
    full_values[position * kv_width + channel] = value[channel];
}

// Complete-block QSA pooling: mean over `compress` raw index keys, grouped
// RMS, then the same HalfSplit RoPE used by the index key stream.  The output
// remains in the resident pooled-key arena; no host selection metadata is
// involved.
extern "C" __global__ void qwen4_qsa_pool_rope_f32(
    const float* raw_keys,
    float* pooled,
    int block_count,
    int compress,
    int index_dim) {
    int block = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (block >= block_count || compress <= 0) return;
    bool active = channel < index_dim;
    float value = 0.0f;
    int base = block * compress * index_dim;
    if (active) {
        for (int row = 0; row < compress; ++row) {
            value += raw_keys[base + row * index_dim + channel] / (float)compress;
        }
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = active ? value * value : 0.0f;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (!active) return;
    float inv = rsqrtf(partial[0] / (float)index_dim + 1.0e-6f);
    value *= inv;
    int half = index_dim / 2;
    int rotary_dim = index_dim < 64 ? index_dim : 64;
    if (channel < rotary_dim) {
        int partner = channel < half ? channel + half : channel - half;
        float other = 0.0f;
        for (int row = 0; row < compress; ++row) {
            other += raw_keys[base + row * index_dim + partner] / (float)compress;
        }
        float other_sum = 0.0f;
        for (int d = 0; d < index_dim; ++d) {
            float v = 0.0f;
            for (int row = 0; row < compress; ++row) {
                v += raw_keys[base + row * index_dim + d] / (float)compress;
            }
            other_sum += v * v;
        }
        other *= rsqrtf(other_sum / (float)index_dim + 1.0e-6f);
        float angle = (float)(block * compress)
            * powf(1000000.0f, -((float)(2 * (channel < half ? channel : partner))) / (float)index_dim);
        float s = sinf(angle);
        float c = cosf(angle);
        value = channel < half ? value * c - other * s : other * s + value * c;
    }
    pooled[block * index_dim + channel] = value;
}

// Select complete four-token blocks by the learned index score and append the
// causal tail. One workgroup owns the current query row; no score is copied to
// host memory.
extern "C" __global__ void qwen4_qsa_select_f32(
    const float* query,
    const float* pooled,
    int* selected,
    int block_count,
    int index_heads,
    int index_dim,
    int budget_blocks,
    int compress,
    int visible,
    int capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int i = 0; i < capacity; ++i) selected[i] = -1;
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    for (int slot = 0; slot < chosen; ++slot) {
        float best = -3.402823466e+38f;
        int best_block = -1;
        for (int block = 0; block < block_count; ++block) {
            bool used = false;
            for (int prior = 0; prior < slot; ++prior) {
                if (selected[prior * compress] == block * compress) used = true;
            }
            if (used) continue;
            float score = 0.0f;
            for (int head = 0; head < index_heads; ++head) {
                float dot = 0.0f;
                for (int d = 0; d < index_dim; ++d) {
                    dot += query[head * index_dim + d]
                        * pooled[block * index_dim + d];
                }
                score += fmaxf(dot / sqrtf((float)index_dim), 0.0f);
            }
            if (score > best) {
                best = score;
                best_block = block;
            }
        }
        if (best_block >= 0) {
            for (int row = 0; row < compress && slot * compress + row < capacity; ++row) {
                int token = best_block * compress + row;
                if (token < visible) selected[slot * compress + row] = token;
            }
        }
    }
    int tail_start = block_count * compress;
    int offset = chosen * compress;
    for (int token = tail_start; token < visible && offset < capacity; ++token) {
        selected[offset++] = token;
    }
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

extern "C" __global__ void qwen4_qsa_attention_f32(
    const float* q_with_gate,
    const float* full_keys,
    const float* full_values,
    const int* selected_indices,
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
        int token = selected_indices[row];
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
        int token = selected_indices[row];
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
    float gate = q_with_gate[head * head_dim * 2 + head_dim + channel];
    output[head * head_dim + channel] =
        denominator > 0.0f ? numerator / denominator / (1.0f + expf(-gate)) : 0.0f;
}
extern "C" __global__ void qwen4_scale_f32(
    float* values,
    int elements,
    float scale) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) values[index] *= scale;
}

// Deterministic device-side greedy top-1 over one or more contiguous logits
// rows.  Ties retain the lowest vocabulary id, matching the reference route.
extern "C" __global__ void qwen4_argmax_f32(
    const float* logits,
    int* indices,
    int rows,
    int vocab) {
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    __shared__ float values[256];
    __shared__ int ids[256];
    float best = -3.402823466e+38f;
    int best_id = 0x7fffffff;
    int base = row * vocab;
    for (int id = (int)threadIdx.x; id < vocab; id += (int)blockDim.x) {
        float value = logits[base + id];
        if (value > best || (value == best && id < best_id)) {
            best = value;
            best_id = id;
        }
    }
    values[threadIdx.x] = best;
    ids[threadIdx.x] = best_id;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = values[threadIdx.x + stride];
            int other_id = ids[threadIdx.x + stride];
            if (other > values[threadIdx.x]
                || (other == values[threadIdx.x] && other_id < ids[threadIdx.x])) {
                values[threadIdx.x] = other;
                ids[threadIdx.x] = other_id;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) indices[row] = ids[0];
}
