// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — Qwen4 fixed top-10 grouped gate/up unscatter fused with SwiGLU.
//
// One pass replaces moe_gate_up_unscatter_top10 -> bf16_round_trip_f32 (gate,
// up) -> silu_mul_f32 -> bf16_round_trip_f32 (activation): for each live
// grouped slot it reads the gate/up logits, applies the same BF16 round trips
// (when `bf16_rt` is set), the same SiLU·up expression, and writes the
// activation to the slot's flat (token, k_rank) row.  Every live flat row is
// produced exactly once (each token owns exactly top-10 routed slots), so the
// output buffer is fully written, bit for bit as the unfused sequence.
//
// Grid: (m_total, ceil(mi / 256)). Block: [256].

#include <hip/hip_runtime.h>

// kernels/src/bf16_round_trip.hip: non-finite values pass through unchanged.
static __device__ __forceinline__ float moe_bf16_round_trip(float v) {
    if (!isfinite(v)) return v;
    union { float f; unsigned int u; } box;
    box.f = v;
    const unsigned int lsb = (box.u >> 16) & 1u;
    box.u = (box.u + 0x7FFFu + lsb) & 0xFFFF0000u;
    return box.f;
}

static __device__ __forceinline__ float moe_widen(float v) { return v; }
static __device__ __forceinline__ float moe_widen(unsigned short b) {
    return __uint_as_float((unsigned int)b << 16);
}

template <typename T>
__device__ __forceinline__ void moe_unscatter_silu(
    const T* __restrict__ Y_grouped,           // [m_total × 2*mi]
    const int*   __restrict__ sorted_slot_index,
    float*       __restrict__ activation,      // [N × 10 × mi] (written)
    int mi, int m_total, int bf16_rt
) {
    const int slot = blockIdx.x;
    if (slot >= m_total) return;
    const int flat = sorted_slot_index[slot];
    if (flat < 0) return;
    const int m_col = blockIdx.y * blockDim.x + threadIdx.x;
    if (m_col >= mi) return;

    const long long y_base = (long long)slot * 2 * mi;
    float v = moe_widen(Y_grouped[y_base + m_col]);
    float u = moe_widen(Y_grouped[y_base + mi + m_col]);
    if (bf16_rt) {
        v = moe_bf16_round_trip(v);
        u = moe_bf16_round_trip(u);
    }
    // kernels/src/silu_mul.hip expression.
    float out = (v / (1.0f + expf(-v))) * u;
    if (bf16_rt) out = moe_bf16_round_trip(out);
    activation[(long long)flat * mi + m_col] = out;
}

extern "C" __global__ void moe_gate_up_unscatter_silu_top10(
    const float* __restrict__ Y_grouped, const int* __restrict__ sorted_slot_index,
    float* __restrict__ activation, int mi, int m_total, int bf16_rt) {
    moe_unscatter_silu(Y_grouped, sorted_slot_index, activation, mi, m_total, bf16_rt);
}

// Grouped gate/up stored as BF16 bits (already the round trip's values).
extern "C" __global__ void moe_gate_up_unscatter_silu_top10_bf16in(
    const unsigned short* __restrict__ Y_grouped, const int* __restrict__ sorted_slot_index,
    float* __restrict__ activation, int mi, int m_total, int bf16_rt) {
    moe_unscatter_silu(Y_grouped, sorted_slot_index, activation, mi, m_total, bf16_rt);
}
