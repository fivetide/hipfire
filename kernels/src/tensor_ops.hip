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

static __device__ __forceinline__ float bf16_to_f32(__hip_bfloat16 value) {
    return static_cast<float>(value);
}

static __device__ __forceinline__ __hip_bfloat16 bf16_round(float value) {
    return static_cast<__hip_bfloat16>(value);
}

static __device__ __forceinline__ __hip_bfloat16 bf16_add(
    __hip_bfloat16 lhs,
    __hip_bfloat16 rhs) {
    return bf16_round(bf16_to_f32(lhs) + bf16_to_f32(rhs));
}



// q/k/v are source BF16 activations promoted to the explicit F32 recurrent
// state.  Each value head repeats one q/k head.  The source l2norm call runs
// before that promotion, so its intermediate products, reduction, rsqrt, and
// output are BF16.  Query scale is a separate F32 multiply after promotion.
extern "C" __global__ void gated_delta_step_f32(
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
    const __hip_bfloat16 eps = bf16_round(1.0e-6f);
    for (int kd = 0; kd < key_dim; ++kd) {
        const __hip_bfloat16 q_value = bf16_round(q[key_base + kd]);
        const __hip_bfloat16 k_value = bf16_round(k[key_base + kd]);
        const float q_float = bf16_to_f32(q_value);
        const float k_float = bf16_to_f32(k_value);
        q_sum_f32 += bf16_to_f32(bf16_round(q_float * q_float));
        k_sum_f32 += bf16_to_f32(bf16_round(k_float * k_float));
    }
    const __hip_bfloat16 q_sum = bf16_round(q_sum_f32);
    const __hip_bfloat16 k_sum = bf16_round(k_sum_f32);
    const __hip_bfloat16 q_inv = bf16_round(
        rsqrtf(bf16_to_f32(q_sum) + bf16_to_f32(eps)));
    const __hip_bfloat16 k_inv = bf16_round(
        rsqrtf(bf16_to_f32(k_sum) + bf16_to_f32(eps)));
    float decay = expf(gate[value_head]);
    float beta_value = beta[value_head];
    int state_base = value_head * key_dim * value_dim;
    for (int kd = 0; kd < key_dim; ++kd) {
        state[state_base + kd * value_dim + value_index] *= decay;
    }
    float kv_mem = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
        const __hip_bfloat16 k_value = bf16_round(k[key_base + kd]);
        const __hip_bfloat16 k_normalized = bf16_round(
            bf16_to_f32(k_value) * bf16_to_f32(k_inv));
        float kv = bf16_to_f32(k_normalized);
        kv_mem += state[state_base + kd * value_dim + value_index] * kv;
    }
    const __hip_bfloat16 v_value = bf16_round(
        v[value_head * value_dim + value_index]);
    float delta = (bf16_to_f32(v_value) - kv_mem) * beta_value;
    for (int kd = 0; kd < key_dim; ++kd) {
        const __hip_bfloat16 k_value = bf16_round(k[key_base + kd]);
        const __hip_bfloat16 k_normalized = bf16_round(
            bf16_to_f32(k_value) * bf16_to_f32(k_inv));
        float kv = bf16_to_f32(k_normalized);
        state[state_base + kd * value_dim + value_index] += kv * delta;
    }
    float out = 0.0f;
    for (int kd = 0; kd < key_dim; ++kd) {
        const __hip_bfloat16 q_value = bf16_round(q[key_base + kd]);
        const __hip_bfloat16 q_normalized = bf16_round(
            bf16_to_f32(q_value) * bf16_to_f32(q_inv));
        float qv = bf16_to_f32(q_normalized);
        qv *= query_scale;
        out += state[state_base + kd * value_dim + value_index] * qv;
    }
    output[value_head * value_dim + value_index] = out;
}
// gfx1151-only exact-128 path.  Q/K normalization is shared by every value
// channel in the block, but the recurrence remains one independent row per
// value channel and keeps the original decay -> kv_mem -> delta -> update ->
// output order.
extern "C" __global__ void gated_delta_step_shared_norm128_gfx1151(
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
    const int value_head = (int)blockIdx.x;
    const int value_index = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    const bool exact_shape = key_dim == 128 && value_dim == 128;
    const bool active = exact_shape
        && value_head < value_heads
        && value_index < 128;
    __shared__ __hip_bfloat16 q_norm[128];
    __shared__ __hip_bfloat16 k_norm[128];
    __shared__ __hip_bfloat16 q_inv_shared;
    __shared__ __hip_bfloat16 k_inv_shared;

    // Load the BF16-rounded q/k vectors cooperatively.  Threads beyond the
    // exact 128-wide vector remain barrier participants so the old 256-wide
    // launch remains safe for direct callers.
    if (exact_shape && value_head < value_heads && threadIdx.x < 128) {
        const int repeat = value_heads / key_heads;
        const int key_head = value_head / repeat;
        const int key_base = key_head * 128;
        q_norm[threadIdx.x] = bf16_round(q[key_base + threadIdx.x]);
        k_norm[threadIdx.x] = bf16_round(k[key_base + threadIdx.x]);
    }
    __syncthreads();

    if (threadIdx.x == 0 && value_head < value_heads && exact_shape) {
        float q_sum_f32 = 0.0f;
        float k_sum_f32 = 0.0f;
        const __hip_bfloat16 eps = bf16_round(1.0e-6f);
        for (int kd = 0; kd < 128; ++kd) {
            const __hip_bfloat16 q_value = q_norm[kd];
            const __hip_bfloat16 k_value = k_norm[kd];
            const float q_float = bf16_to_f32(q_value);
            const float k_float = bf16_to_f32(k_value);
            q_sum_f32 += bf16_to_f32(bf16_round(q_float * q_float));
            k_sum_f32 += bf16_to_f32(bf16_round(k_float * k_float));
        }
        const __hip_bfloat16 q_sum = bf16_round(q_sum_f32);
        const __hip_bfloat16 k_sum = bf16_round(k_sum_f32);
        q_inv_shared = bf16_round(
            rsqrtf(bf16_to_f32(q_sum) + bf16_to_f32(eps)));
        k_inv_shared = bf16_round(
            rsqrtf(bf16_to_f32(k_sum) + bf16_to_f32(eps)));
    }
    __syncthreads();

    if (threadIdx.x < 128) {
        const __hip_bfloat16 q_inv = q_inv_shared;
        const __hip_bfloat16 k_inv = k_inv_shared;
        q_norm[threadIdx.x] = bf16_round(
            bf16_to_f32(q_norm[threadIdx.x]) * bf16_to_f32(q_inv));
        k_norm[threadIdx.x] = bf16_round(
            bf16_to_f32(k_norm[threadIdx.x]) * bf16_to_f32(k_inv));
    }
    __syncthreads();
    if (!active) return;

    const int repeat = value_heads / key_heads;
    const int key_head = value_head / repeat;
    const float decay = expf(gate[value_head]);
    const float beta_value = beta[value_head];
    const int state_base = value_head * 128 * 128;
    for (int kd = 0; kd < 128; ++kd) {
        state[state_base + kd * 128 + value_index] *= decay;
    }
    float kv_mem = 0.0f;
    for (int kd = 0; kd < 128; ++kd) {
        const float kv = bf16_to_f32(k_norm[kd]);
        kv_mem += state[state_base + kd * 128 + value_index] * kv;
    }
    const __hip_bfloat16 v_value = bf16_round(
        v[value_head * 128 + value_index]);
    const float delta = (bf16_to_f32(v_value) - kv_mem) * beta_value;
    for (int kd = 0; kd < 128; ++kd) {
        const float kv = bf16_to_f32(k_norm[kd]);
        state[state_base + kd * 128 + value_index] += kv * delta;
    }
    float out = 0.0f;
    for (int kd = 0; kd < 128; ++kd) {
        float qv = bf16_to_f32(q_norm[kd]);
        qv *= query_scale;
        out += state[state_base + kd * 128 + value_index] * qv;
    }
    output[value_head * 128 + value_index] = out;
}
// Wave-size-independent persistent row-batched path for exact 128x128 geometry.
// A block owns one value head and 256 threads.  tid%128 is the output channel
// and tid/128 selects the half; each thread retains 64 state values in
// registers.  Shared handoffs serialize the kd=0..127 kv and output chains:
// half 1 continues half 0's accumulator rather than combining partial sums.
//
// Nothing but the state recurrence is serial across rows, so the per-row Q/K
// norm factors, decay and beta of up to 256 rows are computed up front (one
// row per thread, each with the exact serial sum the per-row path uses) and
// the next row's raw q/k/v are prefetched while the current row runs.  The
// normalized q/k vectors are double-buffered so a row needs no trailing
// barrier.  Every value is produced by the same expression sequence as
// gated_delta_step_f32, so the result is bitwise unchanged.
__launch_bounds__(256)
extern "C" __global__ void gated_delta_step_halves_state128_persistent256_f32(
    const float* projection,
    const float* gate,
    const float* beta,
    float* state,
    float* output,
    int rows,
    int qkv_width,
    int key_heads,
    int value_heads,
    int key_dim,
    int value_dim,
    float query_scale) {
    const int value_head = (int)blockIdx.x;
    const int thread = (int)threadIdx.x;
    const int value_index = thread & 127;
    const int half = thread >> 7;
    const bool exact_shape = key_dim == 128 && value_dim == 128;
    if (!exact_shape || value_head >= value_heads) return;

    const int qk_width = key_heads * 128;
    const int value_width = value_heads * 128;
    const int repeat = value_heads / key_heads;
    const int key_head = value_head / repeat;
    const int kd_base = half * 64;
    const int state_base = value_head * 128 * 128;
    const int q_offset = key_head * 128;
    const int k_offset = qk_width + key_head * 128;
    const int v_offset = 2 * qk_width + value_head * 128;

    __shared__ __hip_bfloat16 q_norm[2][128];
    __shared__ __hip_bfloat16 k_norm[2][128];
    __shared__ __hip_bfloat16 q_inv_rows[256];
    __shared__ __hip_bfloat16 k_inv_rows[256];
    __shared__ float decay_rows[256];
    __shared__ float beta_rows[256];
    __shared__ float handoff_kv[128];
    __shared__ float handoff_delta[128];
    __shared__ float handoff_out[128];

    float state_half[64];
    #pragma unroll
    for (int kd = 0; kd < 64; ++kd) {
        state_half[kd] =
            state[state_base + (kd_base + kd) * 128 + value_index];
    }

    // Half 0 carries q/k channel value_index, half 1 carries v.
    float raw_a = 0.0f;
    float raw_b = 0.0f;
    if (half == 0) {
        raw_a = projection[q_offset + value_index];
        raw_b = projection[k_offset + value_index];
    } else {
        raw_a = projection[v_offset + value_index];
    }

    for (int block = 0; block < rows; block += 256) {
        const int block_rows = min(256, rows - block);
        __syncthreads();
        if (thread < block_rows) {
            const int row = block + thread;
            const int row_base = row * qkv_width;
            float q_sum_f32 = 0.0f;
            float k_sum_f32 = 0.0f;
            const __hip_bfloat16 eps = bf16_round(1.0e-6f);
            for (int kd = 0; kd < 128; ++kd) {
                const __hip_bfloat16 q_value =
                    bf16_round(projection[row_base + q_offset + kd]);
                const __hip_bfloat16 k_value =
                    bf16_round(projection[row_base + k_offset + kd]);
                const float q_float = bf16_to_f32(q_value);
                const float k_float = bf16_to_f32(k_value);
                q_sum_f32 += bf16_to_f32(bf16_round(q_float * q_float));
                k_sum_f32 += bf16_to_f32(bf16_round(k_float * k_float));
            }
            const __hip_bfloat16 q_sum = bf16_round(q_sum_f32);
            const __hip_bfloat16 k_sum = bf16_round(k_sum_f32);
            q_inv_rows[thread] = bf16_round(
                rsqrtf(bf16_to_f32(q_sum) + bf16_to_f32(eps)));
            k_inv_rows[thread] = bf16_round(
                rsqrtf(bf16_to_f32(k_sum) + bf16_to_f32(eps)));
            decay_rows[thread] = expf(gate[row * value_heads + value_head]);
            beta_rows[thread] = beta[row * value_heads + value_head];
        }
        __syncthreads();

        for (int r = 0; r < block_rows; ++r) {
            const int row = block + r;
            const int buf = r & 1;
            const float cur_a = raw_a;
            const float cur_b = raw_b;
            if (row + 1 < rows) {
                const int next_base = (row + 1) * qkv_width;
                if (half == 0) {
                    raw_a = projection[next_base + q_offset + value_index];
                    raw_b = projection[next_base + k_offset + value_index];
                } else {
                    raw_a = projection[next_base + v_offset + value_index];
                }
            }
            // The first row of a parameter block stages its own normalized
            // q/k; every later row's were staged by half 0 at the end of the
            // previous row, behind that row's last barrier.
            if (r == 0) {
                if (half == 0) {
                    q_norm[buf][value_index] = bf16_round(
                        bf16_to_f32(bf16_round(cur_a)) * bf16_to_f32(q_inv_rows[r]));
                    k_norm[buf][value_index] = bf16_round(
                        bf16_to_f32(bf16_round(cur_b)) * bf16_to_f32(k_inv_rows[r]));
                }
                __syncthreads();
            }

            const float decay = decay_rows[r];
            const float beta_value = beta_rows[r];

            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
                state_half[kd] *= decay;
            }

            float kv_mem = 0.0f;
            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
                const float kv = bf16_to_f32(k_norm[buf][kd_base + kd]);
                kv_mem += state_half[kd] * kv;
            }
            if (half == 0) {
                handoff_kv[value_index] = kv_mem;
            }
            __syncthreads();

            if (half == 1) {
                kv_mem = handoff_kv[value_index];
                #pragma unroll
                for (int kd = 0; kd < 64; ++kd) {
                    const float kv = bf16_to_f32(k_norm[buf][64 + kd]);
                    kv_mem += state_half[kd] * kv;
                }
                const __hip_bfloat16 v_value = bf16_round(cur_a);
                const float delta =
                    (bf16_to_f32(v_value) - kv_mem) * beta_value;
                handoff_delta[value_index] = delta;
            }
            __syncthreads();

            const float delta = handoff_delta[value_index];
            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
                const float kv = bf16_to_f32(k_norm[buf][kd_base + kd]);
                state_half[kd] += kv * delta;
            }

            float out = 0.0f;
            #pragma unroll
            for (int kd = 0; kd < 64; ++kd) {
                float qv = bf16_to_f32(q_norm[buf][kd_base + kd]);
                qv *= query_scale;
                out += state_half[kd] * qv;
            }
            if (half == 0) {
                handoff_out[value_index] = out;
                if (r + 1 < block_rows) {
                    q_norm[buf ^ 1][value_index] = bf16_round(
                        bf16_to_f32(bf16_round(raw_a)) * bf16_to_f32(q_inv_rows[r + 1]));
                    k_norm[buf ^ 1][value_index] = bf16_round(
                        bf16_to_f32(bf16_round(raw_b)) * bf16_to_f32(k_inv_rows[r + 1]));
                }
            }
            __syncthreads();

            if (half == 1) {
                out = handoff_out[value_index];
                #pragma unroll
                for (int kd = 0; kd < 64; ++kd) {
                    float qv = bf16_to_f32(q_norm[buf][64 + kd]);
                    qv *= query_scale;
                    out += state_half[kd] * qv;
                }
                output[row * value_width + value_head * 128 + value_index] = out;
            }
        }
    }

    #pragma unroll
    for (int kd = 0; kd < 64; ++kd) {
        state[state_base + (kd_base + kd) * 128 + value_index] =
            state_half[kd];
    }
}



// Source HF executes the recurrent/chunk result through a hidden BF16 cast
// before RMSNormGated.  Keep the compact production buffer F32 for the
// existing gate kernel while forcing the exact bounded BF16 storage boundary
// through a caller-owned device scratch tensor.
extern "C" __global__ void bf16_roundtrip_f32(
    const float* input,
    __hip_bfloat16* scratch,
    float* output,
    int elements) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= elements) return;
    __hip_bfloat16 rounded = static_cast<__hip_bfloat16>(input[i]);
    scratch[i] = rounded;
    output[i] = bf16_to_f32(scratch[i]);
}
// Source shared-expert folding is BF16 at both multiply and add:
// routed output, shared down output, and sigmoid gate are each stored as
// BF16, then the product and residual sum are rounded independently.
extern "C" __global__ void bf16_scaled_add_f32(
    float* residual,
    const float* value,
    const float* scalar,
    int elements) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= elements) return;
    const __hip_bfloat16 lhs = bf16_round(residual[i]);
    const __hip_bfloat16 rhs = bf16_round(value[i]);
    const __hip_bfloat16 gate = bf16_round(scalar[0]);
    const __hip_bfloat16 product = bf16_round(
        bf16_to_f32(gate) * bf16_to_f32(rhs));
    residual[i] = bf16_to_f32(bf16_round(
        bf16_to_f32(lhs) + bf16_to_f32(product)));
}
// Row-batched sibling of bf16_scaled_add_f32.  The scalar gate is one
// BF16 value per row; every product and residual sum keeps the same source
// rounding boundaries as the decode operation.
extern "C" __global__ void bf16_scaled_add_batched_f32(
    float* residual,
    const float* value,
    const float* scalar,
    int rows,
    int elements) {
    const int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = rows * elements;
    if (flat >= total) return;
    const int row = flat / elements;
    const __hip_bfloat16 lhs = bf16_round(residual[flat]);
    const __hip_bfloat16 rhs = bf16_round(value[flat]);
    const __hip_bfloat16 gate = bf16_round(scalar[row]);
    const __hip_bfloat16 product = bf16_round(
        bf16_to_f32(gate) * bf16_to_f32(rhs));
    residual[flat] = bf16_to_f32(bf16_round(
        bf16_to_f32(lhs) + bf16_to_f32(product)));
}

// Learned HC read/mix. `norm_weight` is resident BF16; down/up projections
// have already run through the native quantized GEMV path.
extern "C" __global__ void hyper_read_f32(
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
                bf16_to_f32(bf16_round(input[offset + j]));
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
            normalized[column] = bf16_to_f32(bf16_round(
                bf16_to_f32(bf16_round(input[column]))
                * inv * (1.0f + bf16_to_f32(norm_weight[column]))));
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
            gate = bf16_to_f32(bf16_round(
                1.0f / (1.0f + expf(-gate))));
            value += bf16_to_f32(bf16_round(
                gate * normalized[offset]));
        }
        mixed[j] = bf16_to_f32(bf16_round(value / (float)branches));
    }
}
// Learned HC read/mix when the up projection has already been materialized.
// The production path runs the native quantized GEMV before this kernel, so
// `projected_up` is one gate logit per branch/hidden column rather than the
// raw `[branches*hidden, rank]` weight matrix consumed above.
// Learned HC read/mix after the grouped RMSNorm has materialized `normalized`.
// The production path runs the native quantized GEMV before this kernel, so
// `projected_up` is one gate logit per branch/hidden column.
extern "C" __global__ void hyper_read_projected_f32(
    const float* input,
    const __hip_bfloat16* norm_weight,
    const float* projected_up,
    float* normalized,
    float* mixed,
    int branches,
    int hidden,
    int rows) {
    const int column = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = (int)blockIdx.y;
    const int wide = branches * hidden;
    if (row >= rows || column >= wide) return;
    if (column < hidden) {
        const int row_offset = row * wide;
        const int mixed_offset = row * hidden;
        float value = 0.0f;
        for (int branch = 0; branch < branches; ++branch) {
            const int offset = row_offset + branch * hidden + column;
            const float gate_input = bf16_to_f32(bf16_round(projected_up[offset]));
            const float gate = bf16_to_f32(bf16_round(
                1.0f / (1.0f + expf(-gate_input))));
            const float normed = bf16_to_f32(bf16_round(normalized[offset]));
            value += bf16_to_f32(bf16_round(gate * normed));
        }
        mixed[mixed_offset + column] =
            bf16_to_f32(bf16_round(value / (float)branches));
    }
    (void)input;
    (void)norm_weight;
}


extern "C" __global__ void hyper_write_f32(
    const float* input,
    const float* normalized,
    const float* mixed,
    const float* gates,
    float* output,

    int branches,
    int hidden,
    int rows) {
    const int column = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = (int)blockIdx.y;
    const int wide = branches * hidden;
    if (row >= rows || column >= wide) return;
    const int row_offset = row * wide;
    const int branch = column / hidden;
    const float gate_logit =
        bf16_to_f32(bf16_round(gates[row * branches + branch]));
    const float gate_scaled = bf16_to_f32(
        bf16_round(gate_logit / (float)branches));
    const float gate = bf16_to_f32(bf16_round(
        2.0f * bf16_to_f32(bf16_round(
            1.0f / (1.0f + expf(-gate_scaled))))));
    const float injection = bf16_to_f32(bf16_round(
        gate * bf16_to_f32(
            bf16_round(mixed[row * hidden + column % hidden]))));
    const float value =
        bf16_to_f32(bf16_round(input[row_offset + column])) + injection;
    output[row_offset + column] = bf16_to_f32(bf16_round(value));
}

extern "C" __global__ void hyper_norm_f32(
    const float* input,
    const __hip_bfloat16* norm_weight,
    float* normalized,
    int branches,
    int hidden,
    int rows) {
    const int branch = (int)blockIdx.x;
    const int row = (int)blockIdx.y;
    if (branch >= branches || row >= rows) return;
    const int tid = (int)threadIdx.x;
    const int offset = (row * branches + branch) * hidden;
    const int weight_offset = branch * hidden;
    __shared__ float rms_sum[256];
    float partial = 0.0f;
    for (int j = tid; j < hidden; j += (int)blockDim.x) {
        const float value = bf16_to_f32(bf16_round(input[offset + j]));
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
        normalized[column] = bf16_to_f32(bf16_round(
            bf16_to_f32(bf16_round(input[column]))
            * inv * (1.0f + bf16_to_f32(norm_weight[weight_offset + j]))));
    }
}
// Lane 0 of `v += __shfl_down(v, off, 32)` for off = 16..1, with the
// partners moved by permlanex16 / row DPP on gfx11/gfx12 (same additions and
// order as gemm_bf16_xf32_multirow's reduction).
static __device__ __forceinline__ float hc_lane0_tree_sum(float v) {
#if defined(__GFX11__) || defined(__GFX12__)
    v += __int_as_float(__builtin_amdgcn_permlanex16(
        __float_as_int(v), __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x108, 0xf, 0xf, true));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x104, 0xf, 0xf, true));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x102, 0xf, 0xf, true));
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x101, 0xf, 0xf, true));
#else
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down(v, offset, 32);
#endif
    return v;
}

// hyper_norm_f32 followed by the BF16 [branches, wide] gate projection of
// that normalized row (gemm_bf16_xf32_multirow / gemv_bf16_xf32), for one
// token per block, without materializing `normalized` in global memory.
// Every branch uses hyper_norm_f32's exact partial sums and 256-wide LDS tree;
// the normalized row lands in LDS and wave w < branches replays the BF16
// kernels' lane traversal (lane*8, +256), four chains, (a0+a1)+(a2+a3) fold
// and shuffle tree for gate row w, so both results are bitwise unchanged.
// Grid [rows], block 256, dynamic LDS (branches * hidden + 256) * 4 bytes.
// Requires (branches * hidden) % 8 == 0 and branches <= 8.
extern "C" __global__ void hyper_norm_gate_f32(
    const float* input,
    const __hip_bfloat16* norm_weight,
    const unsigned short* gate_weight,
    float* gates,
    int branches,
    int hidden) {
    extern __shared__ float hc_lds[];
    const int wide = branches * hidden;
    float* xs = hc_lds;
    float* rms_sum = hc_lds + wide;
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const float* in = input + (size_t)row * wide;
    for (int branch = 0; branch < branches; ++branch) {
        const int offset = branch * hidden;
        float partial = 0.0f;
        for (int j = tid; j < hidden; j += (int)blockDim.x) {
            const float value = bf16_to_f32(bf16_round(in[offset + j]));
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
            xs[offset + j] = bf16_to_f32(bf16_round(
                bf16_to_f32(bf16_round(in[offset + j]))
                * inv * (1.0f + bf16_to_f32(norm_weight[offset + j]))));
        }
        __syncthreads();
    }
    const int wave = tid >> 5;
    const int lane = tid & 31;
    if (wave >= branches) return;
    const unsigned short* w_row = gate_weight + (size_t)wave * wide;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    // Weights are fetched ten steps at a time before they are consumed: one
    // load per step waits a full memory round trip each.
    constexpr int BATCH = 10;
    int i = lane * 8;
    for (; i + 8 + (BATCH - 1) * 256 <= wide; i += BATCH * 256) {
        uint4 wb[BATCH];
#pragma unroll
        for (int u = 0; u < BATCH; ++u) wb[u] = *(const uint4*)(w_row + i + u * 256);
#pragma unroll
        for (int u = 0; u < BATCH; ++u) {
            const uint4 wv = wb[u];
            const float4 xa = *(const float4*)(xs + i + u * 256);
            const float4 xb = *(const float4*)(xs + i + u * 256 + 4);
            a0 += __uint_as_float(wv.x << 16) * xa.x;
            a1 += __uint_as_float(wv.x & 0xFFFF0000u) * xa.y;
            a2 += __uint_as_float(wv.y << 16) * xa.z;
            a3 += __uint_as_float(wv.y & 0xFFFF0000u) * xa.w;
            a0 += __uint_as_float(wv.z << 16) * xb.x;
            a1 += __uint_as_float(wv.z & 0xFFFF0000u) * xb.y;
            a2 += __uint_as_float(wv.w << 16) * xb.z;
            a3 += __uint_as_float(wv.w & 0xFFFF0000u) * xb.w;
        }
    }
    for (; i + 8 <= wide; i += 32 * 8) {
        const uint4 wv = *(const uint4*)(w_row + i);
        const float4 xa = *(const float4*)(xs + i);
        const float4 xb = *(const float4*)(xs + i + 4);
        a0 += __uint_as_float(wv.x << 16) * xa.x;
        a1 += __uint_as_float(wv.x & 0xFFFF0000u) * xa.y;
        a2 += __uint_as_float(wv.y << 16) * xa.z;
        a3 += __uint_as_float(wv.y & 0xFFFF0000u) * xa.w;
        a0 += __uint_as_float(wv.z << 16) * xb.x;
        a1 += __uint_as_float(wv.z & 0xFFFF0000u) * xb.y;
        a2 += __uint_as_float(wv.w << 16) * xb.z;
        a3 += __uint_as_float(wv.w & 0xFFFF0000u) * xb.w;
    }
    const float sum = hc_lane0_tree_sum((a0 + a1) + (a2 + a3));
    if (lane == 0) gates[(size_t)row * branches + wave] = sum;
}
// HC read tail on gfx1151: the BF16 [wide, low_rank] up projection
// (gemm_bf16_xf32_multirow's per-thread route, 256 < low_rank <= 512) fused
// with hyper_read_projected_f32, so the projected gate logits never reach
// global memory.  A block owns hidden columns j0..j0+7 of all four branches:
// lane l computes wide row (l >> 3) * hidden + j0 + (l & 7) exactly as the
// per-thread GEMM does (32 replayed lane partials, shuffle tree), then the
// four branch lanes of a column combine in branch order exactly as
// hyper_read_projected_f32 does.  Eight waves take 2 tokens each, 4 times.
// Grid [hidden / 8, ceil(rows / 64)], block 256, dynamic LDS
// 32 * (low_rank / 2 + 1) * 4 bytes.  branches == 4, hidden % 8 == 0.
extern "C" __global__ __launch_bounds__(256, 1) void hyper_read_up_fused_f32(
    const unsigned short* __restrict__ W,
    const float* __restrict__ X,
    const float* __restrict__ normalized,
    float* __restrict__ mixed,
    int hidden,
    int K,
    int N) {
    constexpr int TPW = 2;
    constexpr int TGROUPS = 4;
    const int branches = 4;
    const int wide = branches * hidden;
    extern __shared__ unsigned int ws[];
    const int stride = K / 2 + 1;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int j0 = blockIdx.x * 8;
    const int vpr = K / 8;
    for (int v = tid; v < 32 * vpr; v += 256) {
        const int r = v / vpr, c = v - r * vpr;
        const int row = (r >> 3) * hidden + j0 + (r & 7);
        const uint4 u = *(const uint4*)(W + (size_t)row * K + c * 8);
        unsigned int* d = ws + r * stride + c * 4;
        d[0] = u.x; d[1] = u.y; d[2] = u.z; d[3] = u.w;
    }
    __syncthreads();
    const unsigned int* wr = ws + lane * stride;
    const int my_row = (lane >> 3) * hidden + j0 + (lane & 7);
    for (int tg = 0; tg < TGROUPS; ++tg) {
        const int tbase = ((blockIdx.y * TGROUPS + tg) * 8 + wave) * TPW;
        if (tbase >= N) break;
        int tok[TPW];
#pragma unroll
        for (int t = 0; t < TPW; ++t) tok[t] = tbase + t < N ? tbase + t : tbase;
        float h[TPW][16];
#pragma unroll
        for (int l = 0; l < 16; ++l) {
            float sp[2][TPW];
#pragma unroll
            for (int p = 0; p < 2; ++p) {
                const int vl = l + 16 * p;
                float a[TPW][4];
#pragma unroll
                for (int t = 0; t < TPW; ++t) a[t][0] = a[t][1] = a[t][2] = a[t][3] = 0.0f;
#pragma unroll
                for (int it = 0; it < 2; ++it) {
                    const int i = vl * 8 + it * 256;
                    if (i + 8 > K) break;
                    const unsigned int u0 = wr[i / 2], u1 = wr[i / 2 + 1], u2 = wr[i / 2 + 2], u3 = wr[i / 2 + 3];
                    const float w0 = __uint_as_float(u0 << 16), w1 = __uint_as_float(u0 & 0xFFFF0000u);
                    const float w2 = __uint_as_float(u1 << 16), w3 = __uint_as_float(u1 & 0xFFFF0000u);
                    const float w4 = __uint_as_float(u2 << 16), w5 = __uint_as_float(u2 & 0xFFFF0000u);
                    const float w6 = __uint_as_float(u3 << 16), w7 = __uint_as_float(u3 & 0xFFFF0000u);
#pragma unroll
                    for (int t = 0; t < TPW; ++t) {
                        const float* xr = X + (size_t)tok[t] * K + i;
                        a[t][0] += w0 * xr[0];
                        a[t][1] += w1 * xr[1];
                        a[t][2] += w2 * xr[2];
                        a[t][3] += w3 * xr[3];
                        a[t][0] += w4 * xr[4];
                        a[t][1] += w5 * xr[5];
                        a[t][2] += w6 * xr[6];
                        a[t][3] += w7 * xr[7];
                    }
                }
#pragma unroll
                for (int t = 0; t < TPW; ++t) sp[p][t] = (a[t][0] + a[t][1]) + (a[t][2] + a[t][3]);
            }
#pragma unroll
            for (int t = 0; t < TPW; ++t) {
                h[t][l] = sp[0][t] + sp[1][t];
                asm volatile("" : "+v"(h[t][l]) :: "memory");
            }
        }
#pragma unroll
        for (int t = 0; t < TPW; ++t) {
#pragma unroll
            for (int off = 8; off > 0; off >>= 1)
#pragma unroll
                for (int l = 0; l < off; ++l) h[t][l] = h[t][l] + h[t][l + off];
            // hyper_read_projected_f32 for (token, column) from the branch
            // lanes; the sum runs in branch order from +0.0.
            const size_t token = (size_t)tok[t];
            const float gate_input = bf16_to_f32(bf16_round(h[t][0]));
            const float gate = bf16_to_f32(bf16_round(1.0f / (1.0f + expf(-gate_input))));
            const float normed = bf16_to_f32(bf16_round(normalized[token * wide + my_row]));
            const float term = bf16_to_f32(bf16_round(gate * normed));
            float value = 0.0f;
#pragma unroll
            for (int branch = 0; branch < 4; ++branch)
                value += __shfl(term, (lane & 7) + 8 * branch, 32);
            if (lane < 8 && tbase + t < N)
                mixed[token * hidden + j0 + lane] = bf16_to_f32(bf16_round(value / (float)branches));
        }
    }
}
extern "C" __global__ void gated_delta_conv_bf16_f32(
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
    const float current = bf16_to_f32(bf16_round(input[channel]));
    float value = bf16_to_f32(kernel[channel * kernel_size + history_rows]) * current;
    // causal_conv1d uses the raw HF weight order: oldest history first,
    // newest history last, and the current token at the final tap.
    for (int tap = 0; tap < history_rows; ++tap) {
        int row = (cursor + tap) % history_rows;
        value += bf16_to_f32(kernel[channel * kernel_size + tap])
            * bf16_to_f32(bf16_round(history[row * channels + channel]));
    }
    output[channel] = bf16_to_f32(bf16_round(
        value * (1.0f / (1.0f + expf(-value)))));
    for (int row = 0; row < history_rows; ++row) {
        next_history[row * channels + channel] =
            bf16_to_f32(bf16_round(history[row * channels + channel]));
    }
    if (history_rows > 0) {
        next_history[cursor * channels + channel] = current;
    }
}
// K=4 ordered causal convolution. Each channel owns a local BF16 ring and
// advances it through all rows, preserving the rowwise source boundaries.
extern "C" __global__ void gated_delta_conv_bf16_f32_batched_k4(
    const float* input,
    const __hip_bfloat16* kernel,
    const float* history,
    float* output,
    float* next_history,
    int rows,
    int channels,
    int history_rows,
    int kernel_size,
    int start_cursor) {
    const int channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (channel >= channels || rows <= 0
        || history_rows != 3 || kernel_size != 4) {
        return;
    }

    __hip_bfloat16 history_ring[3];
    for (int history_row = 0; history_row < 3; ++history_row) {
        history_ring[history_row] = bf16_round(
            history[history_row * channels + channel]);
    }

    int cursor = start_cursor;
    for (int row = 0; row < rows; ++row) {
        const int input_offset = row * channels + channel;
        const __hip_bfloat16 current_bf16 = bf16_round(input[input_offset]);
        const float current = bf16_to_f32(current_bf16);
        float value = bf16_to_f32(kernel[channel * 4 + 3]) * current;
        for (int tap = 0; tap < 3; ++tap) {
            const int history_index = (cursor + tap) % 3;
            value += bf16_to_f32(kernel[channel * 4 + tap])
                * bf16_to_f32(history_ring[history_index]);
        }
        output[input_offset] = bf16_to_f32(bf16_round(
            value * (1.0f / (1.0f + expf(-value)))));
        history_ring[cursor] = current_bf16;
        cursor = cursor == 2 ? 0 : cursor + 1;
    }

    for (int history_row = 0; history_row < 3; ++history_row) {
        next_history[history_row * channels + channel] =
            bf16_to_f32(history_ring[history_row]);
    }
}

// Row-batched parameter expansion; source BF16 boundaries match decode.
extern "C" __global__ void gated_delta_params_bf16_f32_batched(
    const float* a,
    const float* b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    float* gate,
    float* beta,
    int rows,
    int heads) {
    const int flat = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = rows * heads;
    if (flat >= total) return;
    const int head = flat % heads;
    const float al = bf16_to_f32(a_log[head]);
    const float dt = bf16_to_f32(dt_bias[head]);
    const float value = bf16_to_f32(bf16_round(a[flat])) + dt;
    const float softplus = value > 20.0f
        ? value
        : (value < -20.0f ? expf(value) : log1pf(expf(value)));
    gate[flat] = -expf(al) * softplus;
    beta[flat] = 1.0f / (1.0f + expf(-bf16_to_f32(bf16_round(b[flat]))));
}

extern "C" __global__ void gated_delta_params_bf16_f32(
    const float* a,
    const float* b,
    const __hip_bfloat16* a_log,
    const __hip_bfloat16* dt_bias,
    float* gate,
    float* beta,
    int heads) {
    int head = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (head >= heads) return;
    float al = bf16_to_f32(a_log[head]);
    float dt = bf16_to_f32(dt_bias[head]);
    float value = bf16_to_f32(bf16_round(a[head])) + dt;
    float softplus = value > 20.0f ? value : (value < -20.0f ? expf(value) : log1pf(expf(value)));
    gate[head] = -expf(al) * softplus;
    beta[head] = 1.0f / (1.0f + expf(-bf16_to_f32(bf16_round(b[head]))));
}

extern "C" __global__ void gated_delta_params_f32(
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

extern "C" __global__ void gated_delta_gate_bf16_f32(
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
        float value = bf16_to_f32(bf16_round(recurrent[start + i]));
        sum += value * value;
    }
    float inv = rsqrtf(sum / (float)value_dim + 1.0e-6f);
    float normalized = bf16_to_f32(bf16_round(
        recurrent[start + channel] * inv * bf16_to_f32(norm[channel])));
    float gate = bf16_to_f32(bf16_round(z[start + channel]));
    output[start + channel] = bf16_to_f32(bf16_round(
        normalized / (1.0f + expf(-gate))));
    (void)width;
}
// Row-batched gated RMSNorm. The recurrent output is rounded in place before
// the norm reduction, matching the scalar BF16 boundary.  The rounded head is
// staged in LDS so every channel's (identical, serial) sum reads LDS instead
// of re-fetching the head from global memory.
extern "C" __global__ void gated_delta_gate_bf16_f32_batched(
    float* recurrent,
    const float* z,
    const __hip_bfloat16* norm,
    float* output,
    int rows,
    int value_heads,
    int value_dim) {
    const int flat_head = (int)blockIdx.x;
    const int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    const int total_heads = rows * value_heads;
    if (flat_head >= total_heads || value_dim != 128) return;
    const int row = flat_head / value_heads;
    const int head = flat_head % value_heads;
    const int width = value_heads * value_dim;
    const int start = row * width + head * value_dim;
    __shared__ float rounded[128];

    if (channel < value_dim) {
        const float value = bf16_to_f32(bf16_round(recurrent[start + channel]));
        rounded[channel] = value;
        recurrent[start + channel] = value;
    }
    __syncthreads();
    if (channel >= value_dim) return;

    float sum = 0.0f;
    for (int i = 0; i < value_dim; ++i) {
        const float value = bf16_to_f32(bf16_round(rounded[i]));
        sum += value * value;
    }
    const float inv = rsqrtf(sum / (float)value_dim + 1.0e-6f);
    const float normalized = bf16_to_f32(bf16_round(
        rounded[channel] * inv * bf16_to_f32(norm[channel])));
    const float gate = bf16_to_f32(bf16_round(z[start + channel]));
    output[start + channel] = bf16_to_f32(bf16_round(
        normalized / (1.0f + expf(-gate))));
}

extern "C" __global__ void indexed_attention_norm_rope_f32(
    float* values,
    const __hip_bfloat16* norm,
    int heads,
    int head_dim,
    int head_stride,
    int position,
    int rotary_dim) {
    int head = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (head >= heads) return;
    int start = head * head_stride;
    __shared__ float input_row[256];
    if (channel < head_dim) {
        input_row[channel] = values[start + channel];
    }
    __syncthreads();
    if (channel >= head_dim) return;
    float sum = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        float row_value = input_row[i];
        sum += row_value * row_value;
    }
    float inv = rsqrtf(sum / (float)head_dim + 1.0e-6f);
    float value = input_row[channel] * inv
        * (1.0f + bf16_to_f32(norm[channel]));
    if (channel < rotary_dim) {
        int half = rotary_dim / 2;
        int partner = channel < half ? channel + half : channel - half;
        float other = input_row[partner] * inv
            * (1.0f + bf16_to_f32(norm[partner]));
        int base = channel < half ? channel : partner;
        float angle = (float)position
            * powf(10000000.0f, -((float)(2 * base)) / (float)rotary_dim);
        float sine = sinf(angle);
        float cosine = cosf(angle);
        value = channel < half ? value * cosine - other * sine
            : other * sine + value * cosine;
    }
    values[start + channel] = value;
}
 
extern "C" __global__ void indexed_attention_norm_rope_f32_batched(
    float* values,
    const __hip_bfloat16* norm,
    int rows,
    int row_stride,
    int heads,
    int head_dim,
    int head_stride,
    int position_start,
    int rotary_dim) {
    int row = (int)blockIdx.z;
    int head = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (row >= rows || head >= heads) return;
    int start = row * row_stride + head * head_stride;
    __shared__ float input_row[256];
    if (channel < head_dim) {
        input_row[channel] = values[start + channel];
    }
    __syncthreads();
    if (channel >= head_dim) return;
    float sum = 0.0f;
    for (int i = 0; i < head_dim; ++i) {
        float row_value = input_row[i];
        sum += row_value * row_value;
    }
    float inv = rsqrtf(sum / (float)head_dim + 1.0e-6f);
    float value = input_row[channel] * inv
        * (1.0f + bf16_to_f32(norm[channel]));
    if (channel < rotary_dim) {
        int half = rotary_dim / 2;
        int partner = channel < half ? channel + half : channel - half;
        float other = input_row[partner] * inv
            * (1.0f + bf16_to_f32(norm[partner]));
        int base = channel < half ? channel : partner;
        float angle = (float)(position_start + row)
            * powf(10000000.0f, -((float)(2 * base)) / (float)rotary_dim);
        float sine = sinf(angle);
        float cosine = cosf(angle);
        value = channel < half ? value * cosine - other * sine
            : other * sine + value * cosine;
    }
    values[start + channel] = value;
}
 

extern "C" __global__ void indexed_attention_cache_append_f32(
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
 
extern "C" __global__ void indexed_attention_cache_append_f32_batched(
    const float* key,
    const float* value,
    float* full_keys,
    float* full_values,
    int rows,
    int position_start,
    int kv_width) {
    int row = (int)blockIdx.y;
    int channel = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows || channel >= kv_width) return;
    int position = position_start + row;
    full_keys[position * kv_width + channel] = key[row * kv_width + channel];
    full_values[position * kv_width + channel] = value[row * kv_width + channel];
}

// Complete-block QSA pooling: mean over `compress` raw index keys, grouped
// RMS, then the same HalfSplit RoPE used by the index key stream.  The output
// remains in the resident pooled-key arena; no host selection metadata is
// involved.
// Complete-block QSA pooling: mean over raw BF16 index keys, followed by the
// learned source key RMSNorm and HalfSplit RoPE at the block start. A null
// `norm` is used only by synthetic parity buffers.
extern "C" __global__ void indexed_attention_pool_rope_f32(
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
    __hip_bfloat16 mean_bf16 = bf16_round(mean);
    __shared__ float partial[256];
    float squared = active
        ? bf16_to_f32(bf16_round(bf16_to_f32(mean_bf16)
            * bf16_to_f32(mean_bf16)))
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
        : bf16_to_f32(norm[channel]);
    float value = bf16_to_f32(bf16_round(
        bf16_to_f32(mean_bf16) * inv * (1.0f + norm_value)));
    int rotary_dim = index_dim < 64 ? index_dim : 64;
    int half = rotary_dim / 2;
    if (channel < rotary_dim) {
        int partner = channel < half ? channel + half : channel - half;
        float other_mean = 0.0f;
        for (int row = 0; row < compress; ++row) {
            other_mean += raw_keys[base + row * index_dim + partner]
                / (float)compress;
        }
        __hip_bfloat16 other_bf16 = bf16_round(other_mean);
        float other_norm = norm == nullptr
            ? 0.0f
            : bf16_to_f32(norm[partner]);
        float other = bf16_to_f32(bf16_round(
            bf16_to_f32(other_bf16) * inv * (1.0f + other_norm)));
        float angle = (float)(block * compress)
            * powf(10000000.0f, -((float)(2 * (channel < half ? channel : partner)))
                / (float)rotary_dim);
        float sine = sinf(angle);
        float cosine = cosf(angle);
        value = channel < half ? value * cosine - other * sine
            : other * sine + value * cosine;
    }
    pooled[block * index_dim + channel] = bf16_to_f32(
        bf16_round(value));
}

// Compute one block score with the original head/dimension traversal. Keeping
// this as one thread's serial reduction preserves the F32 association and the
// `fmaxf` NaN/tie behavior while allowing independent blocks to run in parallel.
static __device__ __forceinline__ float indexed_attention_select_score_f32(
    const float* query,
    const float* pooled,
    int block,
    int index_heads,
    int index_dim) {
    float score = 0.0f;
    for (int head = 0; head < index_heads; ++head) {
        float dot = 0.0f;
        for (int d = 0; d < index_dim; ++d) {
            dot += query[head * index_dim + d]
                * pooled[block * index_dim + d];
        }
        score += fmaxf(dot / sqrtf((float)index_dim), 0.0f);
    }
    return score;
}

// Serial fallback off gfx1151 or when the score array exceeds its LDS limit.
// Keep stable ascending-block tie selection and the causal tail unchanged.
// Batched selection keeps one independent workgroup per query row. The row's
// visible prefix derives from position_start, so complete blocks that belong
// only to later rows can never enter an earlier row's candidate set.
extern "C" __global__ void indexed_attention_select_f32_batched_serial(
    const float* query,
    const float* pooled,
    int* selected,
    int rows,
    int query_row_stride,
    int block_count,
    int index_heads,
    int index_dim,
    int budget_blocks,
    int compress,
    int position_start,
    int capacity) {
    int row = (int)blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    int visible = position_start + row + 1;
    int row_block_count = visible / compress;
    if (row_block_count > block_count) row_block_count = block_count;
    int* row_selected = selected + row * capacity;
    const float* row_query = query + row * query_row_stride;
    for (int i = 0; i < capacity; ++i) row_selected[i] = -1;
    int chosen = budget_blocks < row_block_count ? budget_blocks : row_block_count;
    for (int slot = 0; slot < chosen; ++slot) {
        float best = -3.402823466e+38f;
        int best_block = -1;
        for (int block = 0; block < row_block_count; ++block) {
            bool used = false;
            for (int prior = 0; prior < slot; ++prior) {
                if (row_selected[prior * compress] == block * compress) used = true;
            }
            if (used) continue;
            float score = indexed_attention_select_score_f32(
                row_query, pooled, block, index_heads, index_dim);
            if (score > best) {
                best = score;
                best_block = block;
            }
        }
        if (best_block >= 0) {
            for (int slot_row = 0;
                 slot_row < compress && slot * compress + slot_row < capacity;
                 ++slot_row) {
                int token = best_block * compress + slot_row;
                if (token < visible) row_selected[slot * compress + slot_row] = token;
            }
        }
    }
    int tail_start = row_block_count * compress;
    int offset = chosen * compress;
    for (int token = tail_start; token < visible && offset < capacity; ++token) {
        row_selected[offset++] = token;
    }
}

extern "C" __global__ void indexed_attention_select_f32_batched(
    const float* query,
    const float* pooled,
    int* selected,
    int rows,
    int query_row_stride,
    int block_count,
    int index_heads,
    int index_dim,
    int budget_blocks,
    int compress,
    int position_start,
    int capacity) {
    int row = (int)blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float scores[];
    int visible = position_start + row + 1;
    int row_block_count = visible / compress;
    if (row_block_count > block_count) row_block_count = block_count;
    const float* row_query = query + row * query_row_stride;
    int* row_selected = selected + row * capacity;
    // Every slot the caller can read owes a -1: the ones above `chosen`, the
    // tail a short row never reaches, and the whole row when it has no
    // complete block. This is the fill the old single-threaded path did.
    for (int i = (int)threadIdx.x; i < capacity; i += (int)blockDim.x) {
        row_selected[i] = -1;
    }
    if (row_block_count > 0) {
        for (int block = (int)threadIdx.x;
             block < row_block_count;
             block += (int)blockDim.x) {
            scores[block] = indexed_attention_select_score_f32(
                row_query, pooled, block, index_heads, index_dim);
        }
    }
    __syncthreads();
    int chosen = budget_blocks < row_block_count ? budget_blocks : row_block_count;
    // rank(block) counts the blocks that sort before it under
    // (score desc, index asc) — the exact order the serial selection sort
    // produced — so writing block `b` at slot `rank(b)` emits the same bytes.
    // Live scores are >= 0 (fmaxf against 0, summed per head), so no block can
    // hold the old sentinel and the order is strict: the ranks are distinct.
    for (int block = (int)threadIdx.x;
         block < row_block_count;
         block += (int)blockDim.x) {
        float score = scores[block];
        int rank = 0;
        for (int other = 0; other < row_block_count; ++other) {
            float value = scores[other];
            if (value > score || (value == score && other < block)) ++rank;
        }
        if (rank < chosen) {
            for (int slot = 0; slot < compress; ++slot) {
                int index = rank * compress + slot;
                int token = block * compress + slot;
                if (index < capacity && token < visible) row_selected[index] = token;
            }
        }
    }
    int tail_start = row_block_count * compress;
    int offset = chosen * compress;
    for (int token = tail_start + (int)threadIdx.x;
         token < visible && offset + (token - tail_start) < capacity;
         token += (int)blockDim.x) {
        row_selected[offset + (token - tail_start)] = token;
    }
}

extern "C" __global__ void indexed_attention_select_f32_serial(
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
            float score = indexed_attention_select_score_f32(
                query, pooled, block, index_heads, index_dim);
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

// Select complete four-token blocks by the learned index score and append the
// causal tail. One workgroup owns the current query row; independent block
// scores are staged in dynamic LDS and the stable selection reuses a sentinel
// in that score array for the old first-slot membership predicate.
extern "C" __global__ void indexed_attention_select_f32(
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
    if (blockIdx.x != 0) return;
    extern __shared__ float scores[];
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    if (chosen > 0) {
        for (int block = (int)threadIdx.x;
             block < block_count;
             block += (int)blockDim.x) {
            scores[block] = indexed_attention_select_score_f32(
                query, pooled, block, index_heads, index_dim);
        }
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    for (int i = 0; i < capacity; ++i) selected[i] = -1;
    for (int slot = 0; slot < chosen; ++slot) {
        float best = -3.402823466e+38f;
        int best_block = -1;
        for (int block = 0; block < block_count; ++block) {
            float score = scores[block];
            if (score == -3.402823466e+38f) continue;
            if (score > best) {
                best = score;
                best_block = block;
            }
        }
        if (best_block >= 0) {
            if (slot * compress < capacity && best_block * compress < visible) {
                scores[best_block] = -3.402823466e+38f;
            }
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
extern "C" __global__ void indexed_attention_reuse_selection(
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


// Serial fallback off gfx1151 or when score/token rows exceed its LDS limit.
// Keep the body identical to the original per-channel implementation.
extern "C" __global__ void indexed_attention_attention_f32_batched_serial(
    const float* q_with_gate,
    const float* full_keys,
    const float* full_values,
    const int* selected_indices,
    float* output,
    int rows,
    int position_start,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity) {
    int row = (int)blockIdx.z;
    int head = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (row >= rows || head >= n_heads || channel >= head_dim) return;
    int visible = position_start + row + 1;
    int block_count = visible / compress;
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int selected_len = chosen * compress + visible - block_count * compress;
    if (selected_len > capacity) selected_len = capacity;
    int kv_head = head / (n_heads / n_kv_heads);
    int q_row_stride = 2 * n_heads * head_dim;
    int q_head_offset = row * q_row_stride + head * head_dim * 2;
    const int* row_selected = selected_indices + row * capacity;
    float max_score = -3.402823466e+38f;
    for (int selected_row = 0; selected_row < selected_len; ++selected_row) {
        int token = row_selected[selected_row];
        if (token < 0 || token >= full_capacity || token >= visible) continue;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_with_gate[q_head_offset + d]
                * full_keys[token * n_kv_heads * head_dim + kv_head * head_dim + d];
        }
        max_score = fmaxf(max_score, score / sqrtf((float)head_dim));
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (int selected_row = 0; selected_row < selected_len; ++selected_row) {
        int token = row_selected[selected_row];
        if (token < 0 || token >= full_capacity || token >= visible) continue;
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_with_gate[q_head_offset + d]
                * full_keys[token * n_kv_heads * head_dim + kv_head * head_dim + d];
        }
        float weight = expf(score / sqrtf((float)head_dim) - max_score);
        denominator += weight;
        numerator += weight
            * full_values[token * n_kv_heads * head_dim + kv_head * head_dim + channel];
    }
    float gate = q_with_gate[q_head_offset + head_dim + channel];
    int output_offset = row * n_heads * head_dim + head * head_dim + channel;
    output[output_offset] =
        denominator > 0.0f ? numerator / denominator / (1.0f + expf(-gate)) : 0.0f;
}

extern "C" __global__ void indexed_attention_attention_f32_batched(
    const float* q_with_gate,
    const float* full_keys,
    const float* full_values,
    const int* selected_indices,
    float* output,
    int rows,
    int position_start,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity) {
    int row = (int)blockIdx.z;
    if (row >= rows) return;
    int visible = position_start + row + 1;
    int block_count = visible / compress;
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int selected_len = chosen * compress + visible - block_count * compress;
    if (selected_len > capacity) selected_len = capacity;
    extern __shared__ float row_scores[];
    int* row_tokens = reinterpret_cast<int*>(row_scores + selected_len);
    const int* row_selected = selected_indices + row * capacity;
    int q_row_stride = 2 * n_heads * head_dim;
    int q_head_offset = row * q_row_stride + (int)blockIdx.x * head_dim * 2;
    for (int selected_row = (int)threadIdx.x;
         selected_row < selected_len;
         selected_row += (int)blockDim.x) {
        int token = row_selected[selected_row];
        row_tokens[selected_row] = token;
        float score = 0.0f;
        if (token >= 0 && token < full_capacity && token < visible) {
            int kv_head = (int)blockIdx.x / (n_heads / n_kv_heads);
            for (int d = 0; d < head_dim; ++d) {
                score += q_with_gate[q_head_offset + d]
                    * full_keys[token * n_kv_heads * head_dim + kv_head * head_dim + d];
            }
            score /= sqrtf((float)head_dim);
        }
        row_scores[selected_row] = score;
    }
    __syncthreads();
    int head = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (head >= n_heads || channel >= head_dim) return;
    int kv_head = head / (n_heads / n_kv_heads);
    float max_score = -3.402823466e+38f;
    for (int selected_row = 0; selected_row < selected_len; ++selected_row) {
        int token = row_tokens[selected_row];
        if (token < 0 || token >= full_capacity || token >= visible) continue;
        max_score = fmaxf(max_score, row_scores[selected_row]);
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (int selected_row = 0; selected_row < selected_len; ++selected_row) {
        int token = row_tokens[selected_row];
        if (token < 0 || token >= full_capacity || token >= visible) continue;
        float weight = expf(row_scores[selected_row] - max_score);
        denominator += weight;
        numerator += weight
            * full_values[token * n_kv_heads * head_dim + kv_head * head_dim + channel];
    }
    int output_offset = row * n_heads * head_dim + head * head_dim + channel;
    float gate = q_with_gate[q_head_offset + head_dim + channel];
    output[output_offset] =
        denominator > 0.0f ? numerator / denominator / (1.0f + expf(-gate)) : 0.0f;
}

// Bitwise-equal to indexed_attention_attention_f32_batched_serial for
// head_dim == 256: every score is the same serial d-order dot and every
// softmax sum runs in selected-row order per channel. One workgroup owns four
// query heads of one KV group, so each key/value row is fetched once for all
// four; keys are staged through a padded LDS tile for coalesced reads and
// expf runs once per (head, token) instead of once per channel.
#define QSA_HG 4
#define QSA_TT 64
#define QSA_DC 64
#define QSA_KS (QSA_DC + 4)
#define QSA_HD 256
extern "C" __global__ __launch_bounds__(256) void indexed_attention_attention_f32_batched_hg4(
    const float* q_with_gate,
    const float* full_keys,
    const float* full_values,
    const int* selected_indices,
    float* output,
    int rows,
    int position_start,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int budget_blocks,
    int compress,
    int capacity,
    int full_capacity) {
    int row = (int)blockIdx.z;
    if (row >= rows) return;
    int visible = position_start + row + 1;
    int block_count = visible / compress;
    int chosen = budget_blocks < block_count ? budget_blocks : block_count;
    int selected_len = chosen * compress + visible - block_count * compress;
    if (selected_len > capacity) selected_len = capacity;
    // Every region is a multiple of 16 bytes so float4 LDS accesses stay aligned.
    extern __shared__ float4 lds4[];
    float* lds = reinterpret_cast<float*>(lds4);
    float* weights = lds;                              // [selected_len][QSA_HG]
    float* qs = weights + QSA_HG * selected_len;       // [QSA_HG][QSA_HD]
    float* kt = qs + QSA_HG * QSA_HD;                  // [QSA_TT][QSA_KS]
    float* maxes = kt + QSA_TT * QSA_KS;               // [QSA_HG]
    int* tokens = reinterpret_cast<int*>(maxes + QSA_HG);  // [selected_len]
    int tid = (int)threadIdx.x;
    int head0 = (int)blockIdx.x * QSA_HG;
    int kv_head = head0 / (n_heads / n_kv_heads);
    int kv_stride = n_kv_heads * QSA_HD;
    const float* keys = full_keys + kv_head * QSA_HD;
    const float* values = full_values + kv_head * QSA_HD;
    const int* row_selected = selected_indices + row * capacity;
    int q_row = row * 2 * n_heads * QSA_HD;
    for (int i = tid; i < selected_len; i += 256) {
        int token = row_selected[i];
        tokens[i] = (token >= 0 && token < full_capacity && token < visible) ? token : -1;
    }
    for (int i = tid; i < QSA_HG * QSA_HD; i += 256) {
        qs[i] = q_with_gate[q_row + (head0 + i / QSA_HD) * QSA_HD * 2 + i % QSA_HD];
    }
    __syncthreads();

    // Scores: each thread owns one selected row and all four heads, reading
    // its key straight from global memory (four independent serial d-order
    // chains; q comes from LDS as a broadcast).  No key tile, no barriers.
    float partial_max[QSA_HG];
#pragma unroll
    for (int h = 0; h < QSA_HG; ++h) partial_max[h] = -3.402823466e+38f;
    for (int base = 0; base < selected_len; base += 256) {
        const int index = base + tid;
        const int token = index < selected_len ? tokens[index] : -1;
        float score[QSA_HG];
#pragma unroll
        for (int h = 0; h < QSA_HG; ++h) score[h] = 0.0f;
        if (token >= 0) {
            const float4* key4 = reinterpret_cast<const float4*>(keys + token * kv_stride);
#pragma unroll 4
            for (int d4 = 0; d4 < QSA_HD / 4; ++d4) {
                const float4 k4 = key4[d4];
#pragma unroll
                for (int h = 0; h < QSA_HG; ++h) {
                    const float4 q4 = reinterpret_cast<const float4*>(qs + h * QSA_HD)[d4];
                    score[h] += q4.x * k4.x;
                    score[h] += q4.y * k4.y;
                    score[h] += q4.z * k4.z;
                    score[h] += q4.w * k4.w;
                }
            }
        }
        if (index < selected_len) {
#pragma unroll
            for (int h = 0; h < QSA_HG; ++h) {
                const float scaled = score[h] / sqrtf((float)QSA_HD);
                weights[index * QSA_HG + h] = scaled;
                if (token >= 0) partial_max[h] = fmaxf(partial_max[h], scaled);
            }
        }
    }
    // fmaxf is exact and order-independent, so the tree matches the serial scan.
#pragma unroll
    for (int h = 0; h < QSA_HG; ++h) kt[h * 256 + tid] = partial_max[h];
    __syncthreads();
    if (tid < QSA_HG) {
        float head_max = -3.402823466e+38f;
        for (int i = 0; i < 256; ++i) head_max = fmaxf(head_max, kt[tid * 256 + i]);
        maxes[tid] = head_max;
    }
    __syncthreads();
    for (int i = tid; i < QSA_HG * selected_len; i += 256) {
        if (tokens[i / QSA_HG] >= 0) weights[i] = expf(weights[i] - maxes[i % QSA_HG]);
    }
    __syncthreads();

    int channel = tid;
    float denominator[QSA_HG];
    float numerator[QSA_HG];
#pragma unroll
    for (int h = 0; h < QSA_HG; ++h) {
        denominator[h] = 0.0f;
        numerator[h] = 0.0f;
    }
    // Invalid (-1) rows hold weight 0 (their score was never exponentiated)
    // and read value 0, so they add +0.0 to sums that are never -0.0: the
    // accumulation is branch-free and still bitwise the skipping version.
    // Eight value loads are issued before their in-order accumulation.
    int i = 0;
    for (; i + 8 <= selected_len; i += 8) {
        float value[8];
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const int token = tokens[i + u];
            value[u] = token >= 0 ? values[token * kv_stride + channel] : 0.0f;
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const float4 w4 = *reinterpret_cast<const float4*>(weights + (i + u) * QSA_HG);
            denominator[0] += w4.x;
            numerator[0] += w4.x * value[u];
            denominator[1] += w4.y;
            numerator[1] += w4.y * value[u];
            denominator[2] += w4.z;
            numerator[2] += w4.z * value[u];
            denominator[3] += w4.w;
            numerator[3] += w4.w * value[u];
        }
    }
    for (; i < selected_len; ++i) {
        const int token = tokens[i];
        const float value = token >= 0 ? values[token * kv_stride + channel] : 0.0f;
        const float4 w4 = *reinterpret_cast<const float4*>(weights + i * QSA_HG);
        denominator[0] += w4.x;
        numerator[0] += w4.x * value;
        denominator[1] += w4.y;
        numerator[1] += w4.y * value;
        denominator[2] += w4.z;
        numerator[2] += w4.z * value;
        denominator[3] += w4.w;
        numerator[3] += w4.w * value;
    }
#pragma unroll
    for (int h = 0; h < QSA_HG; ++h) {
        int head = head0 + h;
        float gate = q_with_gate[q_row + head * QSA_HD * 2 + QSA_HD + channel];
        output[row * n_heads * QSA_HD + head * QSA_HD + channel] = denominator[h] > 0.0f
            ? numerator[h] / denominator[h] / (1.0f + expf(-gate))
            : 0.0f;
    }
}
#undef QSA_KS
#undef QSA_HG
#undef QSA_TT
#undef QSA_DC
#undef QSA_HD

extern "C" __global__ void indexed_attention_attention_f32_serial(
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

// Compute each selected row's token and normalized Q/K score once per
// channel tile, then keep the max/weight/numerator scans serial per output
// channel in the original selected-row order.
extern "C" __global__ void indexed_attention_attention_f32(
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
    extern __shared__ float row_scores[];
    int* row_tokens = reinterpret_cast<int*>(row_scores + selected_len);
    for (int row = (int)threadIdx.x;
         row < selected_len;
         row += (int)blockDim.x) {
        int token = selected_indices[row];
        row_tokens[row] = token;
        float score = 0.0f;
        if (token >= 0 && token < full_capacity) {
            for (int d = 0; d < head_dim; ++d) {
                score += q_with_gate[(int)blockIdx.x * head_dim * 2 + d]
                    * full_keys[token * n_kv_heads * head_dim
                        + ((int)blockIdx.x / (n_heads / n_kv_heads)) * head_dim + d];
            }
            score /= sqrtf((float)head_dim);
        }
        row_scores[row] = score;
    }
    __syncthreads();
    int head = (int)blockIdx.x;
    int channel = (int)(blockIdx.y * blockDim.x + threadIdx.x);
    if (head >= n_heads || channel >= head_dim) return;
    int kv_head = head / (n_heads / n_kv_heads);
    float max_score = -3.402823466e+38f;
    for (int row = 0; row < selected_len; ++row) {
        int token = row_tokens[row];
        if (token < 0 || token >= full_capacity) continue;
        max_score = fmaxf(max_score, row_scores[row]);
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (int row = 0; row < selected_len; ++row) {
        int token = row_tokens[row];
        if (token < 0 || token >= full_capacity) continue;
        float weight = expf(row_scores[row] - max_score);
        denominator += weight;
        numerator += weight * full_values[token * n_kv_heads * head_dim + kv_head * head_dim + channel];
    }
    float gate = q_with_gate[head * head_dim * 2 + head_dim + channel];
    output[head * head_dim + channel] =
        denominator > 0.0f ? numerator / denominator / (1.0f + expf(-gate)) : 0.0f;
}
extern "C" __global__ void scale_f32(
    float* values,
    int elements,
    float scale) {
    int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < elements) values[index] *= scale;
}
// HC-specific pointwise fusion.  This preserves the source order of the
// three F32->BF16->F32 boundaries around scale and SiLU while replacing the
// per-row roundtrip, flat scale, flat SiLU, and per-row roundtrip launches
// with one launch over the contiguous low-rank buffer.
extern "C" __global__ void hc_activation_fused_f32(
    float* values,
    int elements,
    float scale) {
    const int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= elements) return;

    float value = bf16_to_f32(bf16_round(values[index]));
    value *= scale;
    value = bf16_to_f32(bf16_round(value));
    value = value / (1.0f + expf(-value));
    values[index] = bf16_to_f32(bf16_round(value));
}

// Deterministic device-side greedy top-1 over one or more contiguous logits
// rows.  Ties retain the lowest vocabulary id, matching the reference route.
extern "C" __global__ void argmax_f32(
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
