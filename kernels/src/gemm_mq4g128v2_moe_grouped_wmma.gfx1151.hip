// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__gfx1151__)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;
typedef _Float16 __attribute__((ext_vector_type(2))) half2_p_t;
typedef unsigned int __attribute__((ext_vector_type(8))) uint8_p_t;
// Weights 2j, 2j+1 of nibble words (lo, hi) as sc*q + zp in F16: 0x6400 | q
// is exactly 1024 + q, minus 1024 is exactly q, and one fused multiply-add
// per pair is the per-element fma the scalar decode compiles to.
__device__ __forceinline__ half16_t dq_nibbles(unsigned int lo, unsigned int hi,
                                              _Float16 sc, _Float16 zp) {
    const half2_p_t sc2 = {sc, sc};
    const half2_p_t zp2 = {zp, zp};
    const half2_p_t bias = {(_Float16)1024.0f, (_Float16)1024.0f};
    uint8_p_t out;
#pragma unroll
    for (int p = 0; p < 8; p++) {
        const unsigned int t = (p < 4 ? lo : hi) >> (8 * (p & 3));
        const unsigned int bits = (t & 0xFu) | ((t << 12) & 0xF0000u) | 0x64006400u;
        const half2_p_t q2 = __builtin_bit_cast(half2_p_t, bits) - bias;
        out[p] = __builtin_bit_cast(unsigned int, __builtin_elementwise_fma(sc2, q2, zp2));
    }
    return __builtin_bit_cast(half16_t, out);
}

// `x_row_div` controls how sorted_slot_index maps to X_src rows:
//   gate_up: X_src = x_rot_batch[N × K], x_row_div = K_TOP → x_row = flat / K_TOP
//   down:    X_src = rot_batch[total_slots × K], x_row_div = 1 → x_row = flat

// Grouped WMMA GEMM for MoE down over MQ4G128V2 (qt=53) routed experts:
// gemm_mq4g256v2_moe_grouped_wmma_k2 (PR #775 gate/up arm) with the 68 B
// single-header group ([0..2) f16 scale, [2..4) f16 zero, [4..68) 64 B
// nibbles: byte i holds weight 2i low / 2i+1 high).  Eight K-tiles per group.
// F16 dequant and F16 inputs: not bit-exact with the F32 SIMT kernels.
// K % 128 == 0.  Same expert-tile pairing and unrolled K-tile loop as the
// gate/up kernel.  Grid [ceil(M/16), m_total/16], block 32, dynamic LDS
// 16 * (K / 128) * 68 bytes.
// F32 -> BF16 bits, round to nearest even; non-finite values truncate (the
// BF16 round trip leaves them unchanged).  Widening the bits back gives the
// value moe_bf16_round_trip / hip_bfloat16 rounding yields for finite input.
__device__ __forceinline__ unsigned short moe_bf16_bits(float v) {
    const unsigned int u = __float_as_uint(v);
    if (!isfinite(v)) return (unsigned short)(u >> 16);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// One 128-weight group of the lane's row against the tiles' X.  `gw` holds
// the group's 17 words (header, 64 nibble bytes), in LDS or global memory.
template <bool PAIR>
__device__ __forceinline__ void moe_down_wmma_group(
    const unsigned int* gw, const _Float16* const* x_base, int g, float8_t* acc) {
    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const unsigned int hdr = gw[0];
    const _Float16 sc_h = __builtin_bit_cast(_Float16, (unsigned short)(hdr & 0xFFFFu));
    const _Float16 zp_h = __builtin_bit_cast(_Float16, (unsigned short)(hdr >> 16));
#pragma unroll
    for (int kt = 0; kt < 8; kt += 2) {
        const int k_off_a = kt * 16;
        const int k_off_b = (kt + 1) * 16;
        const half16_t a_a = dq_nibbles(gw[1 + 2 * kt], gw[2 + 2 * kt], sc_h, zp_h);
        const half16_t a_b = dq_nibbles(gw[3 + 2 * kt], gw[4 + 2 * kt], sc_h, zp_h);
#pragma unroll
        for (int t = 0; t < (PAIR ? 2 : 1); t++) {
            half16_t b_a = zero_b, b_b = zero_b;
            if (x_base[t] != nullptr) {
                b_a = *(const half16_t*)(x_base[t] + g * 128 + k_off_a);
                b_b = *(const half16_t*)(x_base[t] + g * 128 + k_off_b);
            }
            acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc[t]);
            acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_b, b_b, acc[t]);
        }
    }
}

template <bool PAIR, bool OUT_BF16>
__device__ __forceinline__ void moe_down_wmma_tiles(
    const char* __restrict__ A, const int* __restrict__ sorted_slot_index,
    const _Float16* __restrict__ X_src, void* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total, int tid, int row_start, int slot_start) {
    const int lane = tid & 15;
    const _Float16* x_base[2] = {nullptr, nullptr};
#pragma unroll
    for (int t = 0; t < (PAIR ? 2 : 1); t++) {
        const int slot_idx = slot_start + 16 * t + lane;
        if (slot_idx < m_total) {
            const int flat = sorted_slot_index[slot_idx];
            if (flat >= 0)
                x_base[t] = X_src + (long long)((x_row_div > 1) ? (flat / x_row_div) : flat) * K;
        }
    }
    const int groups_per_row = K / 128;
    const int row_words = groups_per_row * 17;
    float8_t acc[2];
    acc[0] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    acc[1] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    if (row_start + 16 <= M) {
        // The 16 rows are one contiguous 16 * row_bytes block (16-byte
        // aligned): whole-line b128 loads stage it in LDS (dynamic, exactly
        // that size) and the lanes read their row from there.  Loading 8-16
        // bytes of 16 rows per instruction ran DRAM well under streaming rate.
        extern __shared__ __align__(16) unsigned int moe_down_a_lds[];
        const uint4* tile = (const uint4*)(A + (long long)row_start * row_words * 4);
        const int pieces = 4 * row_words;  // 16 rows * row_words words, 4 per piece
#pragma unroll 4
        for (int p = tid; p < pieces; p += 32)
            ((uint4*)moe_down_a_lds)[p] = tile[p];
        __syncthreads();
        for (int g = 0; g < groups_per_row; g++)
            moe_down_wmma_group<PAIR>(moe_down_a_lds + lane * row_words + g * 17, x_base, g, acc);
    } else {
        const char* row_base = A + (long long)min(row_start + lane, M - 1) * row_words * 4;
        for (int g = 0; g < groups_per_row; g++)
            moe_down_wmma_group<PAIR>((const unsigned int*)(row_base + g * 68), x_base, g, acc);
    }
#pragma unroll
    for (int t = 0; t < (PAIR ? 2 : 1); t++) {
        const int out_col = slot_start + 16 * t + lane;
        if (out_col < m_total) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int out_row = row_start + 2 * j + (tid >> 4);
                if (out_row >= M) continue;
                const long long o = (long long)out_col * M + out_row;
                if (OUT_BF16)
                    ((unsigned short*)Y_grouped)[o] = moe_bf16_bits(acc[t][j]);
                else
                    ((float*)Y_grouped)[o] = acc[t][j];
            }
        }
    }
}

template <bool OUT_BF16>
__device__ __forceinline__ void gemm_mq4g128v2_moe_grouped_wmma_gfx1151_entry(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int*                __restrict__ expert_tile_ids,
    const int*                __restrict__ sorted_slot_index,
    const _Float16*           __restrict__ X_src,
    void* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;
    if (row_start >= M || slot_start >= m_total) return;
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;
    int local = 0;
    while (tile_y - local - 1 >= 0 && expert_tile_ids[tile_y - local - 1] == expert_id) ++local;
    if (local & 1) return;
    const char* A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);
    const bool pair = (tile_y + 1) * 16 < m_total && expert_tile_ids[tile_y + 1] == expert_id;
    if (pair)
        moe_down_wmma_tiles<true, OUT_BF16>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, tid, row_start, slot_start);
    else
        moe_down_wmma_tiles<false, OUT_BF16>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, tid, row_start, slot_start);
}

#define MOE_WMMA_ARGS \
    const unsigned long long* __restrict__ expert_weight_ptrs, \
    const int* __restrict__ expert_tile_ids, const int* __restrict__ sorted_slot_index, \
    const _Float16* __restrict__ X_src, void* __restrict__ Y_grouped, \
    int M, int K, int x_row_div, int m_total
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g128v2_moe_grouped_wmma_gfx1151(MOE_WMMA_ARGS) {
    gemm_mq4g128v2_moe_grouped_wmma_gfx1151_entry<false>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src,
                         Y_grouped, M, K, x_row_div, m_total);
}
// Y_grouped as BF16 (RNE) for consumers that round each value to BF16 first.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g128v2_moe_grouped_wmma_gfx1151_bf16out(MOE_WMMA_ARGS) {
    gemm_mq4g128v2_moe_grouped_wmma_gfx1151_entry<true>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src,
                        Y_grouped, M, K, x_row_div, m_total);
}
#undef MOE_WMMA_ARGS

#endif
