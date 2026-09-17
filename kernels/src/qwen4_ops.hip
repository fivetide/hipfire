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
// Source Qwen4 shared-expert folding is BF16 at both multiply and add:
// routed output, shared down output, and sigmoid gate are each stored as
// BF16, then the product and residual sum are rounded independently.
extern "C" __global__ void qwen4_bf16_scaled_add_f32(
    float* residual,
    const float* value,
    const float* scalar,
    int elements) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= elements) return;
    const __hip_bfloat16 lhs = qwen4_bf16_round(residual[i]);
    const __hip_bfloat16 rhs = qwen4_bf16_round(value[i]);
    const __hip_bfloat16 gate = qwen4_bf16_round(scalar[0]);
    const __hip_bfloat16 product = qwen4_bf16_round(
        __bfloat162float(gate) * __bfloat162float(rhs));
    residual[i] = __bfloat162float(qwen4_bf16_round(
        __bfloat162float(lhs) + __bfloat162float(product)));
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
    const int tid = (int)threadIdx.x;
    __shared__ float rms_sum[256];
    for (int branch = 0; branch < branches; ++branch) {
        const int offset = branch * hidden;
        float partial = 0.0f;
        for (int j = tid; j < hidden; j += (int)blockDim.x) {
            const float value =
                __bfloat162float(qwen4_bf16_round(input[offset + j]));
            partial += value * value;
        }
        rms_sum[tid] = partial;
        __syncthreads();
        for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                rms_sum[tid] += rms_sum[tid + stride];
            }
            __syncthreads();
        }
        const float inv = rsqrtf(rms_sum[0] / (float)hidden + 1.0e-6f);
        for (int j = tid; j < hidden; j += (int)blockDim.x) {
            const int column = offset + j;
            normalized[column] = __bfloat162float(qwen4_bf16_round(
                __bfloat162float(qwen4_bf16_round(input[column]))
                * inv * (1.0f + __bfloat162float(norm_weight[column]))));
        }
        __syncthreads();
    }
    for (int j = tid; j < hidden; j += (int)blockDim.x) {
        float value = 0.0f;
        for (int branch = 0; branch < branches; ++branch) {
            const int offset = branch * hidden + j;
            float gate = 0.0f;
            for (int r = 0; r < rank; ++r) {
                gate += up[offset * rank + r] * low[r];
            }
            gate = __bfloat162float(qwen4_bf16_round(
                1.0f / (1.0f + expf(-gate))));
            value += __bfloat162float(qwen4_bf16_round(
                gate * normalized[offset]));
        }
        mixed[j] = __bfloat162float(qwen4_bf16_round(value / (float)branches));
    }
}
// Learned HC read/mix when the up projection has already been materialized.
// The production path runs the native quantized GEMV before this kernel, so
// `projected_up` is one gate logit per branch/hidden column rather than the
// raw `[branches*hidden, rank]` weight matrix consumed above.
// Learned HC read/mix after the grouped RMSNorm has materialized `normalized`.
// The production path runs the native quantized GEMV before this kernel, so
// `projected_up` is one gate logit per branch/hidden column.
extern "C" __global__ void qwen4_hc_read_projected_f32(
    const float* input,
    const __hip_bfloat16* norm_weight,
    const float* projected_up,
    float* normalized,
    float* mixed,
    int branches,
    int hidden) {
    const int column = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int wide = branches * hidden;
    if (column >= wide) return;
    if (column < hidden) {
        float value = 0.0f;
        for (int branch = 0; branch < branches; ++branch) {
            const int offset = branch * hidden + column;
            const float gate_input = __bfloat162float(qwen4_bf16_round(projected_up[offset]));
            const float gate = __bfloat162float(qwen4_bf16_round(
                1.0f / (1.0f + expf(-gate_input))));
            const float normed = __bfloat162float(qwen4_bf16_round(normalized[offset]));
            value += __bfloat162float(qwen4_bf16_round(gate * normed));
        }
        mixed[column] = __bfloat162float(qwen4_bf16_round(value / (float)branches));
    }
    (void)input;
    (void)norm_weight;
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
    const float gate_logit = __bfloat162float(qwen4_bf16_round(gates[branch]));
    const float gate_scaled = __bfloat162float(
        qwen4_bf16_round(gate_logit / (float)branches));
    const float gate = __bfloat162float(qwen4_bf16_round(
        2.0f * __bfloat162float(qwen4_bf16_round(
            1.0f / (1.0f + expf(-gate_scaled))))));
    const float injection = __bfloat162float(qwen4_bf16_round(
        gate * __bfloat162float(qwen4_bf16_round(mixed[column % hidden]))));
    const float value = __bfloat162float(qwen4_bf16_round(input[column])) + injection;
    output[column] = __bfloat162float(qwen4_bf16_round(value));
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
    const int branch = (int)blockIdx.x;
    if (branch >= branches) return;
    const int tid = (int)threadIdx.x;
    const int offset = branch * hidden;
    __shared__ float rms_sum[256];
    float partial = 0.0f;
    for (int j = tid; j < hidden; j += (int)blockDim.x) {
        const float value = __bfloat162float(qwen4_bf16_round(input[offset + j]));
        partial += value * value;
    }
    rms_sum[tid] = partial;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            rms_sum[tid] += rms_sum[tid + stride];
        }
        __syncthreads();
    }
    const float inv = rsqrtf(rms_sum[0] / (float)hidden + 1.0e-6f);
    for (int j = tid; j < hidden; j += (int)blockDim.x) {
        const int column = offset + j;
        normalized[column] = __bfloat162float(qwen4_bf16_round(
            __bfloat162float(qwen4_bf16_round(input[column]))
            * inv * (1.0f + __bfloat162float(norm_weight[column]))));
    }
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
    const float current = __bfloat162float(qwen4_bf16_round(input[channel]));
    float value = __bfloat162float(kernel[channel * kernel_size + history_rows]) * current;
    // causal_conv1d uses the raw HF weight order: oldest history first,
    // newest history last, and the current token at the final tap.
    for (int tap = 0; tap < history_rows; ++tap) {
        int row = (cursor + tap) % history_rows;
        value += __bfloat162float(kernel[channel * kernel_size + tap])
            * __bfloat162float(qwen4_bf16_round(history[row * channels + channel]));
    }
    output[channel] = __bfloat162float(qwen4_bf16_round(
        value * (1.0f / (1.0f + expf(-value)))));
    for (int row = 0; row < history_rows; ++row) {
        next_history[row * channels + channel] =
            __bfloat162float(qwen4_bf16_round(history[row * channels + channel]));
    }
    if (history_rows > 0) {
        next_history[cursor * channels + channel] = current;
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
    float value = __bfloat162float(qwen4_bf16_round(a[head])) + dt;
    float softplus = value > 20.0f ? value : (value < -20.0f ? expf(value) : log1pf(expf(value)));
    gate[head] = -expf(al) * softplus;
    beta[head] = 1.0f / (1.0f + expf(-__bfloat162float(qwen4_bf16_round(b[head]))));
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
        float value = __bfloat162float(qwen4_bf16_round(recurrent[start + i]));
        sum += value * value;
    }
    float inv = rsqrtf(sum / (float)value_dim + 1.0e-6f);
    float normalized = __bfloat162float(qwen4_bf16_round(
        recurrent[start + channel] * inv * __bfloat162float(norm[channel])));
    float gate = __bfloat162float(qwen4_bf16_round(z[start + channel]));
    output[start + channel] = __bfloat162float(qwen4_bf16_round(
        normalized / (1.0f + expf(-gate))));
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
// Complete-block QSA pooling: mean over raw BF16 index keys, followed by the
// learned source key RMSNorm and HalfSplit RoPE at the block start. A null
// `norm` is used only by synthetic parity buffers.
extern "C" __global__ void qwen4_qsa_pool_rope_f32(
    const float* raw_keys,
    float* pooled,
    const __hip_bfloat16* norm,
    int block_count,
    int compress,
    int index_dim) {
    int block = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (block >= block_count || compress <= 0) return;
    bool active = channel < index_dim;
    int base = block * compress * index_dim;
    float mean = 0.0f;
    if (active) {
        for (int row = 0; row < compress; ++row) {
            mean += raw_keys[base + row * index_dim + channel]
                / (float)compress;
        }
    }
    __hip_bfloat16 mean_bf16 = qwen4_bf16_round(mean);
    __shared__ float partial[256];
    float squared = active
        ? __bfloat162float(qwen4_bf16_round(__bfloat162float(mean_bf16)
            * __bfloat162float(mean_bf16)))
        : 0.0f;
    partial[threadIdx.x] = squared;
    __syncthreads();
    for (int stride = (int)blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (!active) return;
    float inv = rsqrtf(partial[0] / (float)index_dim + 1.0e-6f);
    float norm_value = norm == nullptr
        ? 0.0f
        : __bfloat162float(norm[channel]);
    float value = __bfloat162float(qwen4_bf16_round(
        __bfloat162float(mean_bf16) * inv * (1.0f + norm_value)));
    int half = index_dim / 2;
    int rotary_dim = index_dim < 64 ? index_dim : 64;
    if (channel < rotary_dim) {
        int partner = channel < half ? channel + half : channel - half;
        float other_mean = 0.0f;
        for (int row = 0; row < compress; ++row) {
            other_mean += raw_keys[base + row * index_dim + partner]
                / (float)compress;
        }
        __hip_bfloat16 other_bf16 = qwen4_bf16_round(other_mean);
        float other_norm = norm == nullptr
            ? 0.0f
            : __bfloat162float(norm[partner]);
        float other = __bfloat162float(qwen4_bf16_round(
            __bfloat162float(other_bf16) * inv * (1.0f + other_norm)));
        float angle = (float)(block * compress)
            * powf(1000000.0f, -((float)(2 * (channel < half ? channel : partner)))
                / (float)index_dim);
        float sine = sinf(angle);
        float cosine = cosf(angle);
        value = channel < half ? value * cosine - other * sine
            : other * sine + value * cosine;
    }
    pooled[block * index_dim + channel] = __bfloat162float(
        qwen4_bf16_round(value));
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
// Reuse the previous MTP QSA sparse row after step zero. The selected buffer
// stores i32 indices in a byte-addressed Raw allocation. One workgroup owns
// the row so compaction is stable and requires no host round-trip or scratch
// allocation.
extern "C" __global__ void qwen4_qsa_reuse_selection(
    int* selected,
    int* selected_len_out,
    int selected_len,
    int position,
    int capacity) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int write = 0;
    bool has_current = false;
    for (int read = 0; read < selected_len && read < capacity; ++read) {
        int token = selected[read];
        if (token < 0 || token > position) continue;
        if (token == position) has_current = true;
        selected[write++] = token;
    }
    if (!has_current && write < capacity) {
        selected[write++] = position;
    }
    for (int index = write; index < capacity; ++index) {
        selected[index] = -1;
    }
    *selected_len_out = write;
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
