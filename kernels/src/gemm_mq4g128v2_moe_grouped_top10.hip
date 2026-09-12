// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026
// hipfire — Qwen4 MQ4G128V2 grouped top-10 down projection.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Correctness-first grouped consumer for qt=53.  Sixteen route slots share a
// block; sixteen lanes per slot reduce the full K dimension.  The row mapping
// and padded grouped capacity are the same as gemm_q8_0_moe_grouped_top10,
// while weight decoding is the exact 68-byte MQ4G128V2 wire format.
//
// Y_grouped remains unweighted.  The sealed top-10 combine applies route
// weights exactly once after this launch.
__launch_bounds__(256, 2)
extern "C" __global__ void gemm_mq4g128v2_moe_grouped_top10(
    const unsigned long long* __restrict__ expert_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const float* __restrict__ X_src,
    float* __restrict__ Y_grouped,
    int M,
    int K,
    int x_row_div,
    int grouped_rows,
    int x_src_rows,
    int n_exp
) {
    const int row = blockIdx.x;
    const int tile = blockIdx.y;
    const int lane = threadIdx.x;
    const int slot = lane >> 4;
    const int sublane = lane & 15;
    if (row >= M || tile * 16 >= grouped_rows || lane >= 256) return;

    const int expert = expert_tile_ids[tile];
    const int grouped_slot = tile * 16 + slot;
    int x_row = -1;
    if (grouped_slot < grouped_rows) {
        const int flat = sorted_slot_index[grouped_slot];
        if (flat >= 0 && x_row_div > 0) {
            x_row = flat / x_row_div;
            if (x_row < 0 || x_row >= x_src_rows) x_row = -1;
        }
    }

    float partial = 0.0f;
    if (expert >= 0 && expert < n_exp && x_row >= 0 && K > 0) {
        const char* A = reinterpret_cast<const char*>(expert_ptrs[expert]);
        const int groups = (K + 127) / 128;
        const long long row_stride = (long long)groups * 68;
        const char* row_ptr = A + (long long)row * row_stride;
        const float* x = X_src + (long long)x_row * K;
        // Every sublane visits the full logical K with a 16-way stride.  The
        // final group's padded encoded values are multiplied by zero below.
        for (int k = sublane; k < K + 127; k += 16) {
            if (k >= K) continue;
            const int group = k >> 7;
            const int in_group = k & 127;
            const char* gp = row_ptr + (long long)group * 68;
            const unsigned short scale_bits = *(const unsigned short*)(gp + 0);
            const unsigned short zero_bits = *(const unsigned short*)(gp + 2);
            const float scale = __half2float(__ushort_as_half(scale_bits));
            const float zero = __half2float(__ushort_as_half(zero_bits));
            const unsigned char qbyte = *(const unsigned char*)(gp + 4 + (in_group >> 1));
            const unsigned char q = (in_group & 1) ? (qbyte >> 4) : (qbyte & 0xF);
            partial += (scale * (float)q + zero) * x[k];
        }
    }

    __shared__ float partials[16][16];
    partials[slot][sublane] = partial;
    __syncthreads();
    if (sublane == 0) {
        float sum = 0.0f;
        for (int i = 0; i < 16; ++i) sum += partials[slot][i];
        if (grouped_slot < grouped_rows)
            Y_grouped[(long long)grouped_slot * M + row] = sum;
    }
}
