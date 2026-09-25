// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__gfx1151__)

typedef _Float16 __attribute__((ext_vector_type(16))) half16_t;
typedef float __attribute__((ext_vector_type(8))) float8_t;

// Grouped WMMA GEMM for MoE down over MQ4G128V2 (qt=53) routed experts:
// gemm_mq4g256v2_moe_grouped_wmma_k2 (PR #775 gate/up arm) with the 68 B single-header group
// ([0..2) f16 scale, [2..4) f16 zero, [4..68) 64 B nibbles: byte i holds
// weight 2i low / 2i+1 high).  Eight K-tiles per group.  F16 dequant and
// F16 inputs: not bit-exact with the F32 SIMT kernels.  K % 128 == 0.
// Grid [ceil(M/16), m_total/16], block 32.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g128v2_moe_grouped_wmma_gfx1151(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int*                __restrict__ expert_tile_ids,
    const int*                __restrict__ sorted_slot_index,
    const _Float16*           __restrict__ X_src,
    float*                    __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total
) {
    const int tid = threadIdx.x;
    const int row_start = blockIdx.x * 16;
    const int tile_y = blockIdx.y;
    const int slot_start = tile_y * 16;
    if (row_start >= M || slot_start >= m_total) return;
    const int expert_id = expert_tile_ids[tile_y];
    if (expert_id < 0) return;
    const char* __restrict__ A = reinterpret_cast<const char*>(expert_weight_ptrs[expert_id]);

    const int lane = tid & 15;
    const int slot_idx = slot_start + lane;
    int x_row = -1;
    if (slot_idx < m_total) {
        const int flat = sorted_slot_index[slot_idx];
        if (flat >= 0) x_row = (x_row_div > 1) ? (flat / x_row_div) : flat;
    }
    const _Float16* x_base = (x_row >= 0) ? (X_src + (long long)x_row * K) : (const _Float16*)nullptr;
    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 128;
    const char* row_base = A + (long long)safe_row * groups_per_row * 68;

    float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 68;
        const unsigned int hdr = *(const unsigned int*)(gp);
        const _Float16 sc_h = __builtin_bit_cast(_Float16, (unsigned short)(hdr & 0xFFFFu));
        const _Float16 zp_h = __builtin_bit_cast(_Float16, (unsigned short)(hdr >> 16));
        const _Float16* xg = (x_base != nullptr) ? (x_base + g * 128) : nullptr;
        for (int kt = 0; kt < 8; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;
            unsigned int pk0a = *(const unsigned int*)(gp + 4 + k_off_a / 2);
            unsigned int pk1a = *(const unsigned int*)(gp + 4 + k_off_a / 2 + 4);
            unsigned int pk0b = *(const unsigned int*)(gp + 4 + k_off_b / 2);
            unsigned int pk1b = *(const unsigned int*)(gp + 4 + k_off_b / 2 + 4);
            half16_t b_a, b_b;
            if (xg != nullptr) {
                b_a = *(const half16_t*)(xg + k_off_a);
                b_b = *(const half16_t*)(xg + k_off_b);
            } else {
                b_a = zero_b;
                b_b = zero_b;
            }
            half16_t a_a;
            #define DQ(reg, i, pk, sh) reg[i] = sc_h * (_Float16)(float)(((pk) >> (sh)) & 0xFu) + zp_h
            DQ(a_a, 0,  pk0a,  0); DQ(a_a, 1,  pk0a,  4); DQ(a_a, 2,  pk0a,  8); DQ(a_a, 3,  pk0a, 12);
            DQ(a_a, 4,  pk0a, 16); DQ(a_a, 5,  pk0a, 20); DQ(a_a, 6,  pk0a, 24); DQ(a_a, 7,  pk0a, 28);
            DQ(a_a, 8,  pk1a,  0); DQ(a_a, 9,  pk1a,  4); DQ(a_a, 10, pk1a,  8); DQ(a_a, 11, pk1a, 12);
            DQ(a_a, 12, pk1a, 16); DQ(a_a, 13, pk1a, 20); DQ(a_a, 14, pk1a, 24); DQ(a_a, 15, pk1a, 28);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc);
            DQ(a_a, 0,  pk0b,  0); DQ(a_a, 1,  pk0b,  4); DQ(a_a, 2,  pk0b,  8); DQ(a_a, 3,  pk0b, 12);
            DQ(a_a, 4,  pk0b, 16); DQ(a_a, 5,  pk0b, 20); DQ(a_a, 6,  pk0b, 24); DQ(a_a, 7,  pk0b, 28);
            DQ(a_a, 8,  pk1b,  0); DQ(a_a, 9,  pk1b,  4); DQ(a_a, 10, pk1b,  8); DQ(a_a, 11, pk1b, 12);
            DQ(a_a, 12, pk1b, 16); DQ(a_a, 13, pk1b, 20); DQ(a_a, 14, pk1b, 24); DQ(a_a, 15, pk1b, 28);
            #undef DQ
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_b, acc);
        }
    }
    const int out_col = slot_start + lane;
    if (out_col < m_total) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int out_row = row_start + 2 * j + (tid >> 4);
            if (out_row < M) Y_grouped[(long long)out_col * M + out_row] = acc[j];
        }
    }
}

#endif
