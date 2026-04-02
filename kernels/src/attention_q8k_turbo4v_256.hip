#include <hip/hip_runtime.h>
#include "turbo_common.h"

// Asymmetric attention: Q8 K cache + turbo4 V cache, head_dim=256.
// Phase 1: Q @ K^T (Q8 dequant for K) → softmax scores
// Phase 2: Sparse V accumulate — skip positions with score < threshold
// Grid: [n_heads]. Block: [32]. Shared: scores[seq_len] + workspace[32].
extern "C" __global__ void attention_q8k_turbo4v_256(
    const float* __restrict__ q,           // [n_heads * 256]
    const unsigned char* __restrict__ k_q8, // Q8_0 format K cache
    const unsigned char* __restrict__ v_t4, // turbo4 format V cache
    float* __restrict__ out,               // [n_heads * 256]
    const int* __restrict__ pos_buf,
    const float* __restrict__ signs1,      // [256]
    const float* __restrict__ signs2,      // [256]
    int seq_len_hint,
    int n_heads, int n_kv_heads, int head_dim, int max_seq
) {
    const int seq_len = pos_buf[0] + 1;
    extern __shared__ float sdata[];
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const int kv_group = n_heads / n_kv_heads;
    const int kv_h = h / kv_group;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;  // 32

    const float scale = 1.0f / sqrtf((float)head_dim);

    // Q8 K cache layout: [max_seq * n_kv_heads * (34 bytes per 32 elements)]
    // Each Q8_0 block: [f16 scale (2B)] [32 x int8 (32B)] = 34 bytes per 32 elements
    // For head_dim=256: 8 blocks per head per position
    const int q8_blocks_per_head = head_dim / 32;  // 8
    const int q8_bytes_per_head = q8_blocks_per_head * 34;  // 272
    const int q8_bytes_per_pos = n_kv_heads * q8_bytes_per_head;

    // Turbo4 V cache layout: [max_seq * n_kv_heads * (4 + head_dim/2)]
    const int t4_bytes_per_head = 4 + head_dim / 2;  // 132
    const int t4_bytes_per_pos = n_kv_heads * t4_bytes_per_head;

    const float* q_head = q + h * head_dim;
    float* scores = sdata;
    float* ws = sdata + seq_len;

    // Phase 1: Q @ K^T with Q8 dequant
    float local_max = -1e30f;
    for (int t = tid; t < seq_len; t += nthreads) {
        const unsigned char* kb = k_q8 + (size_t)t * q8_bytes_per_pos + kv_h * q8_bytes_per_head;
        float dot = 0.0f;
        // Each thread processes all 8 Q8 blocks (256 dims) for this position
        for (int blk = 0; blk < q8_blocks_per_head; blk++) {
            const unsigned char* block = kb + blk * 34;
            unsigned short s16 = *(const unsigned short*)block;
            float bscale = (float)(*(_Float16*)&s16);
            const signed char* vals = (const signed char*)(block + 2);
            int base = blk * 32;
            for (int j = 0; j < 32; j++) {
                dot += q_head[base + j] * (bscale * (float)vals[j]);
            }
        }
        float s = dot * scale;
        scores[t] = s;
        local_max = fmaxf(local_max, s);
    }

    // Max reduction
    ws[tid] = local_max;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) ws[tid] = fmaxf(ws[tid], ws[tid + s]);
        __syncthreads();
    }
    float max_val = ws[0];
    __syncthreads();

    // Exp + sum
    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += nthreads) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        local_sum += e;
    }
    ws[tid] = local_sum;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) {
        if (tid < s) ws[tid] += ws[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / ws[0];
    __syncthreads();

    // Normalize scores
    for (int t = tid; t < seq_len; t += nthreads) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // Phase 2: Sparse turbo4 V accumulate
    // Each thread handles 8 dims of output (tid*8 .. tid*8+7)
    float mv[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int t = 0; t < seq_len; t++) {
        float w = scores[t];
        if (w < 1e-6f) continue;  // Sparse V skip (+22.8% at 32K)

        const unsigned char* vb = v_t4 + (size_t)t * t4_bytes_per_pos + kv_h * t4_bytes_per_head;
        float cnorm = *(const float*)vb;
        float wn = w * cnorm;

        // Read 4 packed bytes (8 nibbles) for this thread's 8 dims
        unsigned char p0 = vb[4 + tid * 4];
        unsigned char p1 = vb[4 + tid * 4 + 1];
        unsigned char p2 = vb[4 + tid * 4 + 2];
        unsigned char p3 = vb[4 + tid * 4 + 3];

        mv[0] += wn * TURBO_C4_256[(p0)       & 0xF];
        mv[1] += wn * TURBO_C4_256[(p0 >> 4)  & 0xF];
        mv[2] += wn * TURBO_C4_256[(p1)       & 0xF];
        mv[3] += wn * TURBO_C4_256[(p1 >> 4)  & 0xF];
        mv[4] += wn * TURBO_C4_256[(p2)       & 0xF];
        mv[5] += wn * TURBO_C4_256[(p2 >> 4)  & 0xF];
        mv[6] += wn * TURBO_C4_256[(p3)       & 0xF];
        mv[7] += wn * TURBO_C4_256[(p3 >> 4)  & 0xF];
    }

    // Inverse FWHT-256 to undo rotation
    fwht_shfl_inverse_256(mv[0], mv[1], mv[2], mv[3], mv[4], mv[5], mv[6], mv[7],
                          signs1, signs2, tid);

    float* oh = out + h * head_dim;
    int d0 = tid * 8;
    oh[d0] = mv[0]; oh[d0+1] = mv[1]; oh[d0+2] = mv[2]; oh[d0+3] = mv[3];
    oh[d0+4] = mv[4]; oh[d0+5] = mv[5]; oh[d0+6] = mv[6]; oh[d0+7] = mv[7];
}
