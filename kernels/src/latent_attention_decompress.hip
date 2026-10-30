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

// ---------------------------------------------------------------------------
// Q8 expanded mirror (HIPFIRE_RAON_MIRROR_Q8): the K_nope and V planes hold INT8
// codes with one F16 scale per (row, head) and plane. A plane row is
//     [ n_heads * width int8 codes | n_heads F16 scales ]      (bytes)
// i.e. `n_heads * (width + 2)` bytes: head h's codes start at byte `h * width`,
// its scale at byte `n_heads * width + 2 * h`. Quantization of one (row, head,
// plane) vector x[0..width):
//     s16   = F16_rne(max|x| / 127)            (clamped to the F16 range)
//     q[e]  = clamp(rint(x[e] / float(s16)), -127, 127)       (s16 == 0 -> 0)
// Dequantized value q * float(s16). The projector output is quantized directly
// from F32 (no intermediate BF16 rounding).
// ---------------------------------------------------------------------------

#define LA_Q8_THREADS 128

// Quantize one `n`-element vector of F32 (strided source `src`) to int8 codes at
// `dst` and an F16 scale at `sdst`. Block-wide: every thread of the 128-thread
// block takes part; `red` is a 128-float LDS scratch. Ends with a barrier, so
// `red` may be reused immediately.
static __device__ __forceinline__ void la_q8_quant(
    const float* __restrict__ src, int n, int8_t* __restrict__ dst,
    _Float16* __restrict__ sdst, float* red
) {
    const int tid = threadIdx.x;
    float m = 0.0f;
    for (int e = tid; e < n; e += LA_Q8_THREADS) m = fmaxf(m, fabsf(src[e]));
    red[tid] = m;
    __syncthreads();
    for (int s = LA_Q8_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    float sc = red[0] / 127.0f;
    sc = fminf(sc, 65504.0f);
    const _Float16 s16 = (_Float16)sc;
    const float sf = (float)s16;
    for (int e = tid; e < n; e += LA_Q8_THREADS) {
        float q = 0.0f;
        if (sf > 0.0f) q = fminf(fmaxf(rintf(src[e] / sf), -127.0f), 127.0f);
        dst[e] = (int8_t)(int)q;
    }
    if (tid == 0) *sdst = s16;
    __syncthreads();
}

extern "C" __global__ __launch_bounds__(LA_Q8_THREADS) void latent_attention_scatter_expanded_q8(
    const float* __restrict__ proj,      // [n_rows, H*(nope+v)]
    uint8_t* __restrict__ knope,         // [cap][H*(nope+2)] bytes
    uint8_t* __restrict__ vmir,          // [cap][H*(v+2)] bytes
    const int* __restrict__ base_ptr,    // nullable device scalar
    int row0, int n_rows, int n_heads, int nope, int v, int cap
) {
    __shared__ float red[LA_Q8_THREADS];
    const int per = nope + v;
    const int blk = blockIdx.x;                 // (row i, head h)
    const int i = blk / n_heads;
    const int h = blk - i * n_heads;
    if (i >= n_rows) return;
    const int row = (base_ptr ? *base_ptr : 0) + row0 + i;
    if (row < 0 || row >= cap) return;          // block-uniform
    const float* src = proj + ((size_t)i * n_heads + h) * per;
    uint8_t* krow = knope + (size_t)row * ((size_t)n_heads * (nope + 2));
    uint8_t* vrow = vmir + (size_t)row * ((size_t)n_heads * (v + 2));
    la_q8_quant(src, nope, (int8_t*)(krow + (size_t)h * nope),
                (_Float16*)(krow + (size_t)n_heads * nope + 2 * (size_t)h), red);
    la_q8_quant(src + nope, v, (int8_t*)(vrow + (size_t)h * v),
                (_Float16*)(vrow + (size_t)n_heads * v + 2 * (size_t)h), red);
}
