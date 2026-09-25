#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Grouped-WMMA-GEMM for MoE prefill over MQ4G256V2 (qt=44) routed experts.
//
// Transcribed from `gemm_hfq4g256_moe_grouped_wmma_k2` (qt=13). The gather /
// tiling / WMMA pipeline is identical, line for line. The ONLY difference is
// the group header and which affine grid each K-tile dequantizes against.
//
// Why this kernel has to exist: qt44 had GEMV (decode) and dense GEMM paths
// but no MoE grouped-expert path at all, so a qt44 A3B MoE model failed prefill
// with `unsupported moe.prefill-grouped-gemm-dtype`. Routed experts are ~99% of
// such a model's tensors, so without this the format cannot be used for MoE.
//
// Group layout, 136 B stride for BOTH formats:
//
//   qt13 MQ4G256      [0..4)  f32 scale (all 256 weights)
//                     [4..8)  f32 min   (all 256 weights)
//                     [8..136) 128 nibble bytes
//
//   qt44 MQ4G256V2    [0..2)  f16 scale, weights   0..127
//                     [2..4)  f16 zero,  weights   0..127
//                     [4..6)  f16 scale, weights 128..255
//                     [6..8)  f16 zero,  weights 128..255
//                     [8..136) 128 nibble bytes   (identical packing)
//
// Nibble packing is unchanged: byte i holds weight 2i in the low nibble and
// weight 2i+1 in the high nibble. The affine-grid boundary at weight 128
// therefore falls exactly on byte 64, so no byte straddles the two grids.
//
// K-tile → grid mapping: tile kt covers weights [kt*16, kt*16+16). Tiles 0-7
// are weights 0..127 (grid 0); tiles 8-15 are weights 128..255 (grid 1). The
// loop consumes tiles in pairs (kt, kt+1) with kt even, and every such pair
// lies wholly inside one grid — (6,7) → weights 96..127, (8,9) → 128..159.
// So the grid is selected once per pair, not per tile.
//
// Degenerate groups: the encoder writes scale bits 0 and all nibbles 0 when
// hi == lo, giving 0 * q + zero == zero. That is the intended constant-group
// value and needs no special case here.

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
//
// Tiles of one expert are consecutive in expert_tile_ids.  The wave at an even
// expert-local tile index also computes the next tile of the same expert,
// sharing each A load and decode (most experts have a second, mostly padded
// tile); the odd tile's wave exits.  Every tile keeps the single-tile WMMA
// sequence, so outputs are unchanged.  The K-tile loop is unrolled so later
// tile pairs' loads issue ahead of the current decode/WMMA.
// F32 -> BF16 bits, round to nearest even; non-finite values truncate (the
// BF16 round trip leaves them unchanged).  Widening the bits back gives the
// value moe_bf16_round_trip / hip_bfloat16 rounding yields for finite input.
__device__ __forceinline__ unsigned short moe_bf16_bits(float v) {
    const unsigned int u = __float_as_uint(v);
    if (!isfinite(v)) return (unsigned short)(u >> 16);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

template <bool PAIR, bool OUT_BF16>
__device__ __forceinline__ void moe_grouped_wmma_k2_tiles(
    const char* __restrict__ A, const int* __restrict__ sorted_slot_index,
    const _Float16* __restrict__ X_src, void* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total, int tid, int row_start, int slot_start) {
    const int lane = tid & 15;
    const half16_t zero_b = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
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
    const int safe_row = (row_start + lane < M) ? (row_start + lane) : (M - 1);
    const int groups_per_row = K / 256;
    const char* row_base = A + (long long)safe_row * groups_per_row * 136;
    float8_t acc[2];
    acc[0] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    acc[1] = float8_t{0, 0, 0, 0, 0, 0, 0, 0};
    for (int g = 0; g < groups_per_row; g++) {
        const char* gp = row_base + g * 136;
        const unsigned int hdr_lo = *(const unsigned int*)(gp);
        const unsigned int hdr_hi = *(const unsigned int*)(gp + 4);
        const _Float16 sc_0 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_lo & 0xFFFFu));
        const _Float16 zp_0 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_lo >> 16));
        const _Float16 sc_1 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_hi & 0xFFFFu));
        const _Float16 zp_1 = __builtin_bit_cast(_Float16, (unsigned short)(hdr_hi >> 16));
    constexpr int KT_UNROLL = PAIR ? 4 : 8;  // PAIR at 8 spills
#pragma unroll KT_UNROLL
        for (int kt = 0; kt < 16; kt += 2) {
            const int k_off_a = kt * 16;
            const int k_off_b = (kt + 1) * 16;
            const bool upper = (k_off_a >= 128);
            const _Float16 sc_h = upper ? sc_1 : sc_0;
            const _Float16 zp_h = upper ? zp_1 : zp_0;
            const unsigned int pk0a = *(const unsigned int*)(gp + 8 + k_off_a / 2);
            const unsigned int pk1a = *(const unsigned int*)(gp + 8 + k_off_a / 2 + 4);
            const unsigned int pk0b = *(const unsigned int*)(gp + 8 + k_off_b / 2);
            const unsigned int pk1b = *(const unsigned int*)(gp + 8 + k_off_b / 2 + 4);
            const half16_t a_a = dq_nibbles(pk0a, pk1a, sc_h, zp_h);
            const half16_t a_b = dq_nibbles(pk0b, pk1b, sc_h, zp_h);
#pragma unroll
            for (int t = 0; t < (PAIR ? 2 : 1); t++) {
                half16_t b_a = zero_b, b_b = zero_b;
                if (x_base[t] != nullptr) {
                    b_a = *(const half16_t*)(x_base[t] + g * 256 + k_off_a);
                    b_b = *(const half16_t*)(x_base[t] + g * 256 + k_off_b);
                }
                acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_a, b_a, acc[t]);
                acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_b, b_b, acc[t]);
            }
        }
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
__device__ __forceinline__ void gemm_mq4g256v2_moe_grouped_wmma_k2_entry(
    const unsigned long long* __restrict__ expert_weight_ptrs,
    const int* __restrict__ expert_tile_ids,
    const int* __restrict__ sorted_slot_index,
    const _Float16* __restrict__ X_src,
    void* __restrict__ Y_grouped,
    int M, int K, int x_row_div, int m_total) {
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
        moe_grouped_wmma_k2_tiles<true, OUT_BF16>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, tid, row_start, slot_start);
    else
        moe_grouped_wmma_k2_tiles<false, OUT_BF16>(A, sorted_slot_index, X_src, Y_grouped, M, K, x_row_div, m_total, tid, row_start, slot_start);
}

#define MOE_WMMA_ARGS \
    const unsigned long long* __restrict__ expert_weight_ptrs, \
    const int* __restrict__ expert_tile_ids, const int* __restrict__ sorted_slot_index, \
    const _Float16* __restrict__ X_src, void* __restrict__ Y_grouped, \
    int M, int K, int x_row_div, int m_total
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_wmma_k2(MOE_WMMA_ARGS) {
    gemm_mq4g256v2_moe_grouped_wmma_k2_entry<false>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src,
                         Y_grouped, M, K, x_row_div, m_total);
}
// Y_grouped as BF16 (RNE) for consumers that round each value to BF16 first.
__launch_bounds__(32, 2)
extern "C" __global__ void gemm_mq4g256v2_moe_grouped_wmma_k2_bf16out(MOE_WMMA_ARGS) {
    gemm_mq4g256v2_moe_grouped_wmma_k2_entry<true>(expert_weight_ptrs, expert_tile_ids, sorted_slot_index, X_src,
                        Y_grouped, M, K, x_row_div, m_total);
}
#undef MOE_WMMA_ARGS
