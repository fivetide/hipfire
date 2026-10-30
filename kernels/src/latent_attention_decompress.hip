// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <stdint.h>

// Latent-attention decompression: cache rows [latent || tail] -> per-head
// expanded K_nope / V rows (BF16), token-major mirrors
//     knope[t][h][nope]   vmir[t][h][v]      (a row is contiguous across heads,
//                                             so the mirror grows by appending rows)
// The shared tail channels are NOT copied per head: attention reads them
// directly from the latent cache row (shared across heads).
//
// Source rows are `row = (base_ptr ? *base_ptr : 0) + row0 + i`, i < n_rows,
// so a decode/prefill step can decompress exactly the rows it just cached with
// a device-resident position.
//
// kv_b row r = h*(nope+v)+e, e<nope -> K_nope[h], e>=nope -> V[h].
//
// Three entry points:
//   latent_attention_gather_latent_f32      cache rows -> F32 plane [n,latent]
//                                           (input of an external projector)
//   latent_attention_scatter_expanded_bf16  F32 projector output [n, H*(nope+v)]
//                                           -> BF16 mirrors
//   latent_attention_decompress_dense       fused dense projector (BF16/F32
//                                           weights) + BF16 round + scatter
// The dense projector accumulates in FP32 strictly in ascending latent index,
// independent of tile composition, so one row decompressed in any tiling or at
// any time has identical bits.

static __device__ __forceinline__ uint16_t la_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static __device__ __forceinline__ float la_bf16_to_f32(uint16_t h) {
    return __uint_as_float(((uint32_t)h) << 16);
}

extern "C" __global__ void latent_attention_gather_latent_f32(
    const uint16_t* __restrict__ cache,  // [cap, latent+tail] BF16
    const int* __restrict__ base_ptr,    // nullable device scalar
    float* __restrict__ out,             // [n_rows, latent]
    int row0, int n_rows, int latent, int tail, int cap
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)n_rows * latent;
    if (idx >= total) return;
    const int i = (int)(idx / latent);
    const int c = (int)(idx % latent);
    const int row = (base_ptr ? *base_ptr : 0) + row0 + i;
    float v = 0.0f;
    if (row >= 0 && row < cap) v = la_bf16_to_f32(cache[(size_t)row * (latent + tail) + c]);
    out[idx] = v;
}

extern "C" __global__ void latent_attention_scatter_expanded_bf16(
    const float* __restrict__ proj,      // [n_rows, H*(nope+v)]
    uint16_t* __restrict__ knope,        // [cap, H, nope]
    uint16_t* __restrict__ vmir,         // [cap, H, v]
    const int* __restrict__ base_ptr,    // nullable device scalar
    int row0, int n_rows, int n_heads, int nope, int v, int cap
) {
    const int per = nope + v;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)n_rows * n_heads * per;
    if (idx >= total) return;
    const int i = (int)(idx / ((size_t)n_heads * per));
    const int o = (int)(idx % ((size_t)n_heads * per));
    const int h = o / per;
    const int e = o % per;
    const int row = (base_ptr ? *base_ptr : 0) + row0 + i;
    if (row < 0 || row >= cap) return;
    const uint16_t b = la_f32_to_bf16(proj[idx]);
    if (e < nope) knope[((size_t)row * n_heads + h) * nope + e] = b;
    else vmir[((size_t)row * n_heads + h) * v + (e - nope)] = b;
}

#define LA_TM 16
#define LA_TN 16
#define LA_TK 32

extern "C" __global__ __launch_bounds__(256) void latent_attention_decompress_dense(
    const uint16_t* __restrict__ cache,  // [cap, latent+tail] BF16
    const int* __restrict__ base_ptr,    // nullable device scalar
    const void* __restrict__ w,          // [H*(nope+v), latent] BF16 or F32
    uint16_t* __restrict__ knope,        // [cap, H, nope]
    uint16_t* __restrict__ vmir,         // [cap, H, v]
    int row0, int n_rows, int n_heads, int nope, int v, int latent, int tail,
    int cap, int w_bf16, int mcap
) {
    __shared__ float cs[LA_TM][LA_TK + 1];
    __shared__ float ws[LA_TN][LA_TK + 1];
    const int tx = threadIdx.x;  // output column within tile
    const int ty = threadIdx.y;  // row within tile
    const int per = nope + v;
    const int n_out = n_heads * per;
    const int col_base = blockIdx.x * LA_TN;
    const int row_base = blockIdx.y * LA_TM;
    const int base = base_ptr ? *base_ptr : 0;
    const int lin = ty * LA_TN + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < latent; k0 += LA_TK) {
        for (int e = lin; e < LA_TM * LA_TK; e += 256) {
            const int r = e / LA_TK;
            const int k = e % LA_TK;
            float cv = 0.0f, wv = 0.0f;
            if (k0 + k < latent) {
                const int ri = row_base + r;
                const int row = base + row0 + ri;
                if (ri < n_rows && row >= 0 && row < cap)
                    cv = la_bf16_to_f32(cache[(size_t)row * (latent + tail) + k0 + k]);
                const int ci = col_base + r;
                if (ci < n_out) {
                    const size_t wi = (size_t)ci * latent + k0 + k;
                    wv = w_bf16 ? la_bf16_to_f32(((const uint16_t*)w)[wi]) : ((const float*)w)[wi];
                }
            }
            cs[r][k] = cv;
            ws[r][k] = wv;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < LA_TK; ++k) acc = fmaf(cs[ty][k], ws[tx][k], acc);
        __syncthreads();
    }
    const int ri = row_base + ty;
    const int o = col_base + tx;
    const int row = base + row0 + ri;
    if (ri >= n_rows || o >= n_out || row < 0 || row >= cap || row >= mcap) return;
    const int h = o / per;
    const int e = o % per;
    const uint16_t b = la_f32_to_bf16(acc);
    if (e < nope) knope[((size_t)row * n_heads + h) * nope + e] = b;
    else vmir[((size_t)row * n_heads + h) * v + (e - nope)] = b;
}
