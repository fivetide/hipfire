// SPDX-License-Identifier: MIT
// hipfire — see LICENSE and NOTICE in the project root.

// MQ4G256V2 A8 MMQ for gfx11 (RDNA3 — gfx1100 / gfx1151): int8 per-128-K
// activations x MQ4V2 weight nibbles on `v_wmma_i32_16x16x16_iu8` (wave32).
//
// The gfx11 sibling of `gemm_mq4g256v2_residual_mmq_i8.gfx12.hip`: same
// argument lists (7-arg SET/ADD, 7-arg gate/up SiLU), same `block_i8_128`
// activations (nibble-pair qs order, f32 d, exact int s), banded raster and
// the same pinned fold, so the output is bit-identical to the gfx12 route
// and to the CPU reference in
// `crates/hipfire-runtime/examples/a8_gfx11_oracle.rs`:
//   C   = sum_k (q - 8) * x8       per (row, token, K128), integer-exact
//   acc = fma(RN(sc * d), (float)C, acc)   per K128 block, in K order
//
// CTA tile I8_TILE x I8_TILE (weight rows x tokens), I8_TILE = 128 (16
// waves, entries named as on gfx12) or 256 (32 waves, entries suffixed
// `_t256`). On gfx1151 the A8 route is bound by memory-system power, not by
// WMMA issue: the socket sits at its 85 W limit and the GFX clock drops.
// The 256 tile halves the L2/fabric bytes per MAC (int8 activations are
// twice the IU4 bytes, so 128-tiles move ~2.9x IU4's bytes per MAC).
//
// RDNA3 WMMA operand form: A and B are 16 int8 per lane (one K16 step),
// lanes r and r+16 carry identical operands, and C element l of lane L is
// C[2l + (L >> 4)][L & 15]. The kernel puts the ACTIVATIONS on A (C rows =
// tokens) and the expanded WEIGHTS on B (C columns = weight rows), so one
// epilogue store instruction writes two 64-byte runs of Y = [N][M].
//
// Wave tile 16*I8_NB tokens (nb, A) x 32 weight rows (rg = 0..1, B):
// 2*I8_NB int32x8 chains; per K16 step I8_NB + 2 b128 LDS reads feed
// 2*I8_NB WMMAs. I8_NB = 2 (tile 128) or 4 (tile 256).
//
// Weights stay packed in LDS (8 bytes = one K16 step of one row) and each
// step expands them to the B operand {lo(pk0), hi(pk0), lo(pk1), hi(pk1)},
// lo = pk & 0x0F0F0F0F, hi = (pk >> 4) & 0x0F0F0F0F: exactly the nibble-pair
// byte order of the activations, so no byte shuffle exists in the K loop.
// (Packed, the 256 tile fits the 64 KB workgroup LDS limit and the weight
// plane costs half the LDS bytes.) Codes enter the WMMA raw and UNSIGNED
// (0..15) against signed int8 activations; the -8 * s zero-point term rides
// in the chain seed (I8_MAGIC - 8 * s per token), and the fold reads
// fc = as_float(C) - 12582912.0f, exact for
// |sum| <= 15 * 127 * 128 + 8 * 127 * 128 < 2^22.
//
// LDS (I8_LDS_BYTES): two slots of one K64 slab each,
//   A_s [I8_TILE tokens][64 B]: 16-byte chunk c of token e at chunk
//       c ^ ((e >> 1) & 3) (b128 reads and writes conflict-free per 8 lanes),
//   W_s [I8_TILE rows][32 B]: 8-byte step c of row e at step c ^ ((e >> 2) & 3)
//       (b64 reads conflict-free per 16 lanes),
// then per-K128 parity planes
//   DS_p: seed[I8_TILE] i32 + d[I8_TILE] f32 in the C-row order
//         p(t) = (t & ~15) | ((t & 1) << 3) | ((t >> 1) & 7),
//   SZ_p: sc[I8_TILE] f32 (weight row scale of the K128 half).
// Every staging map has one thread per (token, 16-byte chunk) and per (row,
// 8-byte step) because I8_TILE == threads / 4. Two K64 slabs per K128 run
// as a register-prefetched ring with one barrier per slab.
// Grid: [ceil(M/I8_TILE), ceil(N/I8_TILE)] (gate/up: [ceil(2M/I8_TILE), ...]).
// Block [threads,1,1]. Compile gate: gfx1100 / gfx1151 only.
//
// I8_A8_EPI build (one entry, `gemm_mq4g256v2_gate_up_silu_a8_mmq_i8`): the
// gate/up SiLU GEMM on a 512-virtual-row x 128-token CTA (32 waves, wave tile
// as in the 256 build), so one CTA owns 256 h rows = one whole FWHT-256 group
// of w_down's input. Its epilogue runs the A8 hin producer
// (`fused_silu_mul_mq_rotate_awq.hip` under HIPFIRE_SILU_HIN) on its own h
// and stores w_down's `block_i8_128` sidecar; h never reaches memory. Same
// L2 bytes per MAC as the 256 tile (512 x 0.5 B weights + 128 x 1 B
// activations per K). Grid [2M/512, ceil(N/128)], requires M % 256 == 0.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifdef I8_A8_EPI
#define I8_TM 512
#define I8_TN 128
#define I8_NB 4
#define I8_NTHREADS 1024
#else
#ifndef I8_TILE
#define I8_TILE 128
#endif
#if I8_TILE == 256
#define I8_ENTRY(name) name##_t256
#elif I8_TILE == 128
#define I8_ENTRY(name) name
#else
#error "I8_TILE must be 128 or 256"
#endif
// CTA tile I8_TM weight rows x I8_TN tokens.
#define I8_TM I8_TILE
#define I8_TN I8_TILE
#define I8_NB (I8_TILE / 64)
#define I8_NTHREADS (I8_TILE * 4)
#endif

#if defined(__gfx1100__) || defined(__gfx1151__)

// block_i8_128 comes from the host JIT prelude (block_i8_128_quant.hip).

// Token waves per CTA: wave_tok0 = (wave & (I8_TW - 1)) * 16 * I8_NB, row
// wave wave >> I8_TW_LOG2 (a shift, as before the A8 build: `/` changes the
// SILU entries' register allocation).
#define I8_TW (I8_TN / (16 * I8_NB))
#define I8_TW_LOG2 (I8_TW == 4 ? 2 : 1)
_Static_assert(I8_TW == (1 << I8_TW_LOG2), "token waves: 2 or 4");
#ifndef I8_RASTER_BAND
#define I8_RASTER_BAND 8
#endif
// VGPR budgets at granule 24 of 1536: tile 128 -> 120 = 12 waves/SIMD, three
// 16-wave CTAs per WGP (unconstrained the allocator takes 124: two CTAs);
// tile 256 -> 192 = 8 waves/SIMD, one 32-wave CTA per WGP.
#ifndef I8_NUM_VGPR
#define I8_NUM_VGPR (I8_NTHREADS == 1024 ? 192 : 120)
#endif

#define I8_A_SLAB_BYTES (I8_TN * 64)
#define I8_W_SLAB_BYTES (I8_TM * 32)
#define I8_SLOT_BYTES (I8_A_SLAB_BYTES + I8_W_SLAB_BYTES)
#define I8_DS_OFF (2 * I8_SLOT_BYTES)
#define I8_DS_BYTES (I8_TN * 8)
#define I8_SZ_OFF (I8_DS_OFF + 2 * I8_DS_BYTES)
#define I8_SZ_BYTES (I8_TM * 4)
#define I8_LDS_BYTES (I8_SZ_OFF + 2 * I8_SZ_BYTES)
_Static_assert(I8_LDS_BYTES == (I8_NTHREADS == 1024 ? 55296 : 27648),
               "i8 gfx11 planes must match the host LDS contract");
_Static_assert(I8_LDS_BYTES <= 65536, "RDNA3 workgroup LDS limit");
#ifdef I8_A8_EPI
// A8 epilogue h staging (reuses the dead K-loop planes): 32 tokens x 256 h
// rows f32, row stride 272 floats (the two tokens of one store sit 16 banks
// apart).
#define I8_H_STRIDE 272
_Static_assert(32 * I8_H_STRIDE * 4 <= I8_LDS_BYTES, "h staging fits the planes");
#endif
#define I8_MAGIC 0x4B400000

typedef int32_t __attribute__((ext_vector_type(4))) int32x4_t;
typedef int32_t __attribute__((ext_vector_type(8))) int32x8_t;
typedef unsigned __attribute__((ext_vector_type(2))) u32x2_t;
typedef unsigned __attribute__((ext_vector_type(4))) u32x4_t;

// Byte offset of 16-byte chunk c of entity e inside a [I8_TILE][64 B] plane.
static __device__ __forceinline__ unsigned i8_chunk(unsigned e, unsigned c) {
    return e * 64u + ((c ^ ((e >> 1) & 3u)) << 4);
}

// C-row order of the DS planes: tokens 2l + hi of a 16-token group sit at
// hi * 8 + l, so one lane's 8 C rows are 32 contiguous bytes.
static __device__ __forceinline__ unsigned i8_ds_slot(unsigned t) {
    return (t & ~15u) | ((t & 1u) << 3) | ((t >> 1) & 7u);
}

enum { I8_EPI_SET = 0, I8_EPI_ADD = 1, I8_EPI_SILU = 2, I8_EPI_SILU_A8 = 3 };
#define I8_SILU_MUL(g, u) ((g) / (1.0f + expf(-(g))) * (u))

template <int EPI>
static __device__ __forceinline__ void gemm_mq4g256v2_residual_mmq_i8_gfx11_body(
    const char*           __restrict__ A,   // [M x groups_per_row x 136]
    const char*           __restrict__ U,   // SILU: W_up, same layout as A
    const block_i8_128*   __restrict__ Xq,  // [K/128 x N]
    float*                __restrict__ Y,   // [N x M] column-major
    // SILU_A8 only: w_down's AWQ scale [M], FWHT signs [256] and its
    // block_i8_128 sidecar Yd [M/128 x N].
    const float*          __restrict__ awq,
    const float*          __restrict__ signs1,
    const float*          __restrict__ signs2,
    block_i8_128*         __restrict__ Yd,
    int M, int K, int N
) {
    constexpr bool ADD = EPI == I8_EPI_ADD;
    constexpr bool SILU = EPI == I8_EPI_SILU || EPI == I8_EPI_SILU_A8;
    const int tid = threadIdx.x;
    const int wave = tid >> 5;
    const int lane = tid & 31;
    int rs;
    int bs;
    {
        // Banded raster (gfx12 A8 / K1): bijective for any grid.
        const int n_row_tiles = gridDim.x;
        const int lin = blockIdx.y * n_row_tiles + blockIdx.x;
        const int per_band = I8_RASTER_BAND * gridDim.y;
        const int band = lin / per_band;
        const int in_band = lin - band * per_band;
        const int band_row0 = band * I8_RASTER_BAND;
        const int band_rows = min(n_row_tiles - band_row0, I8_RASTER_BAND);
        rs = (band_row0 + in_band % band_rows) * I8_TM;
        bs = (in_band / band_rows) * I8_TN;
    }
    if (rs >= (SILU ? 2 * M : M) || bs >= N) return;
    const int hs = rs >> 1;  // SILU: first h row of the tile

    const int groups_per_row = K / 256;
    const int kblocks = K / 128;
    extern __shared__ char LDS[];

    // Wave tile: tokens wave_tok0 + 16 nb + .., weight rows wave_row0 + 16 rg + ..
    const unsigned l16 = (unsigned)lane & 15u;
    const unsigned hi = (unsigned)lane >> 4;
    const int wave_tok0 = (wave & (I8_TW - 1)) * (16 * I8_NB);
    const int wave_row0 = (wave >> I8_TW_LOG2) * 32;

    // Tile row R -> source row and matrix. SILU: 16-row groups alternate
    // gate (even) and up (odd) rows of the same h rows, so a wave's rg 0 /
    // rg 1 chains are gate / up of identical h rows.
    auto src_row = [&](int R) -> int {
        if constexpr (SILU) return hs + (R >> 5) * 16 + (R & 15);
        else return rs + R;
    };
    auto src_base = [&](int R) -> const char* {
        if constexpr (SILU) return ((R >> 4) & 1) ? U : A;
        else return A;
    };

    // --- Staging maps (loop-invariant). ---
#if I8_TM == I8_TN
    // A slab: thread stages token T = tid >> 2, chunk c = tid & 3.
    const unsigned st_T = (unsigned)(tid >> 2);
    const unsigned st_c = (unsigned)(tid & 3);
    const unsigned st_aoff = (unsigned)(bs + (int)st_T) * (unsigned)sizeof(block_i8_128) + 8u + st_c * 16u;
    const unsigned st_alds = i8_chunk(st_T, st_c);
    // W slab: row R = tid >> 2, K16 step c = tid & 3 (8 packed bytes). A
    // wave stages rows 8w..8w+7, one 16-row group: the gate/up selector is
    // wave-uniform, and readfirstlane keeps the descriptor in SGPRs (a VGPR
    // descriptor turns every load into a waterfall loop).
    const int st_wsrc = src_row((int)st_T);
    const unsigned st_woff =
        (unsigned)(st_wsrc < M ? st_wsrc : (M - 1)) * (unsigned)groups_per_row * 136u + 8u + st_c * 8u;
    // Packed W plane: row R's 8 bytes of K16 step c at R*32 + 8*(c ^ ((R >> 2) & 3)).
    const unsigned st_wlds = I8_A_SLAB_BYTES + st_T * 32u + ((st_c ^ ((st_T >> 2) & 3u)) << 3);
    const char* st_wbase = src_base(__builtin_amdgcn_readfirstlane((int)st_T));
#else
    // 512 x 128 tile, 1024 threads. A slab: token T = tid >> 3, 8-byte half
    // h = tid & 7 of chunk h >> 1. W slab: row R = tid >> 1, K16 steps
    // 2j, 2j + 1 (j = tid & 1) from one 16-byte load, stored as two swizzled
    // 8-byte steps. A wave stages rows 16w..16w+15: one gate/up group.
    const unsigned st_T = (unsigned)(tid >> 3);
    const unsigned st_h = (unsigned)(tid & 7);
    const unsigned st_aoff = (unsigned)(bs + (int)st_T) * (unsigned)sizeof(block_i8_128) + 8u + st_h * 8u;
    const unsigned st_alds = i8_chunk(st_T, st_h >> 1) + ((st_h & 1u) << 3);
    const unsigned st_R = (unsigned)(tid >> 1);
    const unsigned st_j = (unsigned)(tid & 1);
    const int st_wsrc = src_row((int)st_R);
    const unsigned st_woff =
        (unsigned)(st_wsrc < M ? st_wsrc : (M - 1)) * (unsigned)groups_per_row * 136u + 8u + st_j * 16u;
    const unsigned st_wsw = (st_R >> 2) & 3u;
    const unsigned st_wlds0 = I8_A_SLAB_BYTES + st_R * 32u + (((2u * st_j) ^ st_wsw) << 3);
    const unsigned st_wlds1 = I8_A_SLAB_BYTES + st_R * 32u + (((2u * st_j + 1u) ^ st_wsw) << 3);
    const char* st_wbase = src_base(__builtin_amdgcn_readfirstlane((int)st_R));
#endif
    const auto w_rsrc = __builtin_amdgcn_make_buffer_rsrc((void*)st_wbase, 0, 0xffffffffu, 0x31004000u);
    // Headers: every thread fetches and publishes token (tid % I8_TN)'s
    // (d, s) and row (tid % I8_TM)'s scale (identical writers per slot). A
    // wave's 32 header rows span a gate and an up group: plain per-lane
    // loads.
    const int hdr_t = tid & (I8_TN - 1);
    const int hdr_r = tid & (I8_TM - 1);
    const unsigned hdr_ds = i8_ds_slot((unsigned)hdr_t);
    const int hdr_src = src_row(hdr_r);
    const char* hdr_sp = src_base(hdr_r)
        + (long long)(hdr_src < M ? hdr_src : (M - 1)) * groups_per_row * 136;
    const unsigned a_bytes = (unsigned)N * (unsigned)sizeof(block_i8_128);

    // Consumer lane bases. The swizzle terms (A: (e >> 1) & 3, W: (e >> 2) &
    // 3) depend on l16 only, so each operand is one lane base XOR the step's
    // chunk bits plus an immediate (16 nb tokens = 1024 B, 16 rg rows = 512 B).
    const unsigned rd_sw = ((l16 >> 1) & 3u) << 4;
    const unsigned rd_a = ((unsigned)wave_tok0 + l16) * 64u + rd_sw;
    const unsigned rd_w = I8_A_SLAB_BYTES + ((unsigned)wave_row0 + l16) * 32u + (((l16 >> 2) & 3u) << 3);

    float acc[I8_NB][2][8];
    #pragma unroll
    for (int nb = 0; nb < I8_NB; nb++)
        #pragma unroll
        for (int rg = 0; rg < 2; rg++)
            #pragma unroll
            for (int l = 0; l < 8; l++) acc[nb][rg][l] = 0.0f;
    int32x8_t cacc[I8_NB][2];

#if I8_TM == I8_TN
    u32x4_t a_pf;
    u32x2_t w_pf;
#else
    u32x2_t a_pf;
    u32x4_t w_pf;
#endif
    u32x2_t hd_pf;
    unsigned hs_pf;

    // Global fetch of K64 slab (kb, s) into registers.
    auto fetch_slab = [&](int kb, int s) {
        const auto a_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(Xq + (long long)kb * N), 0, a_bytes, 0x31004000u);
#if I8_TM == I8_TN
        a_pf = __builtin_amdgcn_raw_buffer_load_b128(a_rsrc, st_aoff, (unsigned)(s * 64), 0);
        w_pf = __builtin_amdgcn_raw_buffer_load_b64(
            w_rsrc, st_woff, (unsigned)((kb >> 1) * 136 + (kb & 1) * 64 + s * 32), 0);
#else
        a_pf = __builtin_amdgcn_raw_buffer_load_b64(a_rsrc, st_aoff, (unsigned)(s * 64), 0);
        w_pf = __builtin_amdgcn_raw_buffer_load_b128(
            w_rsrc, st_woff, (unsigned)((kb >> 1) * 136 + (kb & 1) * 64 + s * 32), 0);
#endif
    };
    auto publish_slab = [&](int slot) {
        char* base = LDS + slot * I8_SLOT_BYTES;
#if I8_TM == I8_TN
        *(u32x4_t*)(base + st_alds) = a_pf;
        *(u32x2_t*)(base + st_wlds) = w_pf;
#else
        *(u32x2_t*)(base + st_alds) = a_pf;
        *(u32x2_t*)(base + st_wlds0) = (u32x2_t){w_pf.x, w_pf.y};
        *(u32x2_t*)(base + st_wlds1) = (u32x2_t){w_pf.z, w_pf.w};
#endif
    };
    // Header fetch / publish of K128 block kb, both in every thread with no
    // branch: a descriptor merged across a divergent branch lands in VGPRs
    // (waterfall loops), and loads consumed only inside one get sunk next to
    // their use with a full vmcnt(0) wait in front of the barrier.
    auto fetch_hdr = [&](int kb) {
        const auto d_rsrc = __builtin_amdgcn_make_buffer_rsrc(
            (void*)(Xq + (long long)kb * N), 0, a_bytes, 0x31004000u);
        hd_pf = __builtin_amdgcn_raw_buffer_load_b64(
            d_rsrc, (unsigned)(bs + hdr_t) * (unsigned)sizeof(block_i8_128), 0u, 0);
        hs_pf = *(const unsigned*)(hdr_sp + (kb >> 1) * 136 + (kb & 1) * 4);
    };
    auto publish_hdr = [&](int par) {
        // Tokens >= N read d = 0, s = 0 (descriptor bound).
        int* seed = (int*)(LDS + I8_DS_OFF + par * I8_DS_BYTES);
        float* dd = (float*)(LDS + I8_DS_OFF + par * I8_DS_BYTES + I8_TN * 4);
        float* sz = (float*)(LDS + I8_SZ_OFF + par * I8_SZ_BYTES);
        seed[hdr_ds] = I8_MAGIC - 8 * (int)hd_pf.y;
        dd[hdr_ds] = __uint_as_float(hd_pf.x);
        sz[hdr_r] = __half2float(__ushort_as_half((unsigned short)(hs_pf & 0xFFFFu)));
    };

    // One packed K16 weight step -> the B operand {lo(pk0), hi(pk0), lo(pk1),
    // hi(pk1)}: the nibble-pair byte order of the activations.
    auto expand_w = [](u32x2_t pk) -> int32x4_t {
        int32x4_t w;
        w.x = (int)(pk.x & 0x0F0F0F0Fu);
        w.y = (int)((pk.x >> 4) & 0x0F0F0F0Fu);
        w.z = (int)(pk.y & 0x0F0F0F0Fu);
        w.w = (int)((pk.y >> 4) & 0x0F0F0F0Fu);
        return w;
    };
    // One K64 slab of compute from `slot`: 4 K16 steps, each fenced so only
    // one step's operands are live. Unfenced, the scheduler hoists every
    // step's LDS reads to the slab start and the kernel spills. (Loading the
    // next step's operands ahead measured no faster at 12 waves/SIMD.)
    auto compute_slab = [&](int slot) {
        const char* base = LDS + slot * I8_SLOT_BYTES;
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            int32x4_t af[I8_NB];
            #pragma unroll
            for (int nb = 0; nb < I8_NB; nb++)
                af[nb] = *(const int32x4_t*)(base + (rd_a ^ ((unsigned)t << 4)) + nb * 1024);
            const int32x4_t w0 = expand_w(*(const u32x2_t*)(base + (rd_w ^ ((unsigned)t << 3))));
            const int32x4_t w1 = expand_w(*(const u32x2_t*)(base + (rd_w ^ ((unsigned)t << 3)) + 512));
            #pragma unroll
            for (int nb = 0; nb < I8_NB; nb++) {
                cacc[nb][0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[nb], false, w0, cacc[nb][0], false);
                cacc[nb][1] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af[nb], false, w1, cacc[nb][1], false);
            }
            __builtin_amdgcn_sched_barrier(0);
        }
    };
    // WMMAs touch no memory: without this pin they sink past the next
    // barrier and every slab operand stays live across it.
    auto pin_chains = [&]() {
        #pragma unroll
        for (int nb = 0; nb < I8_NB; nb++)
            asm volatile("" : "+v"(cacc[nb][0]), "+v"(cacc[nb][1]));
    };

    // --- Prologue: headers of block 0 and slab (0, 0) into slot 0. ---
    fetch_hdr(0);
    fetch_slab(0, 0);
    publish_hdr(0);
    publish_slab(0);
    __syncthreads();

    #pragma clang loop unroll(disable)
    for (int kb = 0; kb < kblocks; kb++) {
        const int par = kb & 1;
        // The last block refetches itself into dead planes (no reader after
        // the final barrier): the body stays one basic block.
        const int kn = (kb + 1 == kblocks) ? kb : kb + 1;
        // Chain seeds: I8_MAGIC - 8 * s per C row (token).
        {
            const int* seed = (const int*)(LDS + I8_DS_OFF + par * I8_DS_BYTES)
                + wave_tok0 + hi * 8;
            #pragma unroll
            for (int nb = 0; nb < I8_NB; nb++) {
                const int32x4_t s0 = *(const int32x4_t*)(seed + nb * 16);
                const int32x4_t s1 = *(const int32x4_t*)(seed + nb * 16 + 4);
                const int32x8_t sv = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
                cacc[nb][0] = sv;
                cacc[nb][1] = sv;
            }
        }
        // Slab 0 (slot 0); prefetch slab 1 -> slot 1 and the next headers.
        fetch_slab(kb, 1);
        fetch_hdr(kn);
        compute_slab(0);
        pin_chains();
        publish_slab(1);
        publish_hdr(par ^ 1);
        __syncthreads();
        // Slab 1 (slot 1); prefetch the next block's slab 0 -> slot 0.
        fetch_slab(kn, 0);
        compute_slab(1);
        pin_chains();
        // Fold this K128 block: acc = fma(RN(sc * d), as_float(C) - magic, acc).
        {
            const float* dd = (const float*)(LDS + I8_DS_OFF + par * I8_DS_BYTES + I8_TN * 4)
                + wave_tok0 + hi * 8;
            const float* sz = (const float*)(LDS + I8_SZ_OFF + par * I8_SZ_BYTES)
                + wave_row0 + l16;
            const float sc[2] = {sz[0], sz[16]};
            #pragma unroll
            for (int nb = 0; nb < I8_NB; nb++) {
                float d8[8];
                *(float4*)(d8) = *(const float4*)(dd + nb * 16);
                *(float4*)(d8 + 4) = *(const float4*)(dd + nb * 16 + 4);
                #pragma unroll
                for (int rg = 0; rg < 2; rg++) {
                    #pragma unroll
                    for (int l = 0; l < 8; l++) {
                        const float t1 = __fmul_rn(sc[rg], d8[l]);
                        // Scalar copy first: clang lowers __builtin_bit_cast
                        // of an ext-vector element lvalue to element 0.
                        const int c = cacc[nb][rg][l];
                        const float fc = __int_as_float(c) - 12582912.0f;
                        acc[nb][rg][l] = __builtin_fmaf(t1, fc, acc[nb][rg][l]);
                    }
                }
            }
        }
        publish_slab(0);
        __syncthreads();
    }

    // --- Epilogue: lane holds token wave_tok0 + 16 nb + 2l + hi and row
    // wave_row0 + 16 rg + l16 (SILU: h row hs + 16 (wave >> I8_TW_LOG2) + l16). ---
#ifdef I8_A8_EPI
    if constexpr (EPI == I8_EPI_SILU_A8) {
        // h rows hs..hs+255 are FWHT group hs / 256 of w_down's input. Pass p
        // stages h of tokens 64 tw + 16 p + (0..15) of both token waves tw
        // (32 tokens) into the dead K-loop planes; wave w then runs the A8
        // hin producer (fused_silu_mul_mq_rotate_awq.hip, HIPFIRE_SILU_HIN)
        // on pass token w with that kernel's expressions in its order, so the
        // blocks are byte-identical to fused_silu_hin_rotate_mq_i8_batched
        // on the f32 h this GEMM's SILU entry stores.
        float* Hs = (float*)LDS;
        const int group = hs >> 8;
        const int d0 = lane * 8;
        const int base = hs + d0;
        const int feat = (wave >> I8_TW_LOG2) * 16 + (int)l16;
        const int ptw = (wave & (I8_TW - 1)) * 16 + (int)hi;
        const float* h_b = Hs + wave * I8_H_STRIDE;
        const int token0 = bs + (wave >> 4) * (16 * I8_NB) + (wave & 15);
        #pragma unroll
        for (int p = 0; p < I8_NB; p++) {
            #pragma unroll
            for (int l = 0; l < 8; l++)
                Hs[(ptw + 2 * l) * I8_H_STRIDE + feat] = I8_SILU_MUL(acc[p][0][l], acc[p][1][l]);
            __syncthreads();
            const int token = token0 + p * 16;
            if (token < N) {
                const float4 ha = *(const float4*)(h_b + d0);
                const float4 hb = *(const float4*)(h_b + d0 + 4);
                float v0 = (ha.x / awq[base    ]) * signs1[d0    ];
                float v1 = (ha.y / awq[base + 1]) * signs1[d0 + 1];
                float v2 = (ha.z / awq[base + 2]) * signs1[d0 + 2];
                float v3 = (ha.w / awq[base + 3]) * signs1[d0 + 3];
                float v4 = (hb.x / awq[base + 4]) * signs1[d0 + 4];
                float v5 = (hb.y / awq[base + 5]) * signs1[d0 + 5];
                float v6 = (hb.z / awq[base + 6]) * signs1[d0 + 6];
                float v7 = (hb.w / awq[base + 7]) * signs1[d0 + 7];

                float t;
                t = v0; v0 = v0 + v1; v1 = t - v1;
                t = v2; v2 = v2 + v3; v3 = t - v3;
                t = v4; v4 = v4 + v5; v5 = t - v5;
                t = v6; v6 = v6 + v7; v7 = t - v7;

                t = v0; v0 = v0 + v2; v2 = t - v2;
                t = v1; v1 = v1 + v3; v3 = t - v3;
                t = v4; v4 = v4 + v6; v6 = t - v6;
                t = v5; v5 = v5 + v7; v7 = t - v7;

                t = v0; v0 = v0 + v4; v4 = t - v4;
                t = v1; v1 = v1 + v5; v5 = t - v5;
                t = v2; v2 = v2 + v6; v6 = t - v6;
                t = v3; v3 = v3 + v7; v7 = t - v7;

                #define I8_HBFLY(v, pat, str) do { \
                    float _p = __int_as_float(__builtin_amdgcn_ds_swizzle(__float_as_int(v), (pat))); \
                    if (lane & (str)) { (v) = _p - (v); } else { (v) = (v) + _p; } \
                } while(0)
                #define I8_HBFLY8(pat, str) \
                    I8_HBFLY(v0,pat,str); I8_HBFLY(v1,pat,str); I8_HBFLY(v2,pat,str); I8_HBFLY(v3,pat,str); \
                    I8_HBFLY(v4,pat,str); I8_HBFLY(v5,pat,str); I8_HBFLY(v6,pat,str); I8_HBFLY(v7,pat,str)
                I8_HBFLY8(0x041F, 1);
                I8_HBFLY8(0x081F, 2);
                I8_HBFLY8(0x101F, 4);
                I8_HBFLY8(0x201F, 8);
                I8_HBFLY8(0x401F, 16);
                #undef I8_HBFLY8
                #undef I8_HBFLY

                const float s = 0.0625f;
                const float f0 = v0 * s * signs2[d0    ];
                const float f1 = v1 * s * signs2[d0 + 1];
                const float f2 = v2 * s * signs2[d0 + 2];
                const float f3 = v3 * s * signs2[d0 + 3];
                const float f4 = v4 * s * signs2[d0 + 4];
                const float f5 = v5 * s * signs2[d0 + 5];
                const float f6 = v6 * s * signs2[d0 + 6];
                const float f7 = v7 * s * signs2[d0 + 7];
                emit_i8_sidecar_from_producer8(f0, f1, f2, f3, f4, f5, f6, f7, Yd, group, token, N, lane);
            }
            if (p + 1 < I8_NB) __syncthreads();
        }
        return;
    }
#endif
    if constexpr (SILU) {
        const int hrow = hs + (wave >> I8_TW_LOG2) * 16 + (int)l16;
        if (hrow >= M) return;
        #pragma unroll
        for (int nb = 0; nb < I8_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < 8; l++) {
                const int tok = bs + wave_tok0 + nb * 16 + 2 * l + (int)hi;
                if (tok < N) Y[(long long)tok * M + hrow] = I8_SILU_MUL(acc[nb][0][l], acc[nb][1][l]);
            }
        }
        return;
    }
    #pragma unroll
    for (int rg = 0; rg < 2; rg++) {
        const int row = rs + wave_row0 + rg * 16 + (int)l16;
        if (row >= M) continue;
        #pragma unroll
        for (int nb = 0; nb < I8_NB; nb++) {
            #pragma unroll
            for (int l = 0; l < 8; l++) {
                const int tok = bs + wave_tok0 + nb * 16 + 2 * l + (int)hi;
                if (tok < N) {
                    float* yp = Y + (long long)tok * M + row;
                    if constexpr (ADD) *yp += acc[nb][rg][l];
                    else *yp = acc[nb][rg][l];
                }
            }
        }
    }
}

#ifdef I8_A8_EPI
// Gate/up SwiGLU + w_down A8 sidecar: G/U = gate/up MQ4V2 rows (M each,
// M % 256 == 0), awq = w_down AWQ scale [M], Yd = block_i8_128 [M/128 x N].
extern "C" __global__ __launch_bounds__(I8_NTHREADS) __attribute__((amdgpu_num_vgpr(I8_NUM_VGPR)))
void gemm_mq4g256v2_gate_up_silu_a8_mmq_i8(
    const char*           __restrict__ G,
    const char*           __restrict__ U,
    const block_i8_128*   __restrict__ Xq,
    const float*          __restrict__ awq,
    const float*          __restrict__ signs1,
    const float*          __restrict__ signs2,
    block_i8_128*         __restrict__ Yd,
    int M, int K, int N
) {
    gemm_mq4g256v2_residual_mmq_i8_gfx11_body<I8_EPI_SILU_A8>(
        G, U, Xq, nullptr, awq, signs1, signs2, Yd, M, K, N);
}
#else
extern "C" __global__ __launch_bounds__(I8_NTHREADS) __attribute__((amdgpu_num_vgpr(I8_NUM_VGPR)))
void I8_ENTRY(gemm_mq4g256v2_residual_mmq_i8_full_add)(
    const char*           __restrict__ A,
    const block_i8_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int
) {
    gemm_mq4g256v2_residual_mmq_i8_gfx11_body<I8_EPI_ADD>(A, nullptr, Xq, Y, nullptr, nullptr, nullptr, nullptr, M, K, N);
}

extern "C" __global__ __launch_bounds__(I8_NTHREADS) __attribute__((amdgpu_num_vgpr(I8_NUM_VGPR)))
void I8_ENTRY(gemm_mq4g256v2_residual_mmq_i8_full_set)(
    const char*           __restrict__ A,
    const block_i8_128*   __restrict__ Xq,
    float*                __restrict__ Y,
    int M, int K, int N, int
) {
    gemm_mq4g256v2_residual_mmq_i8_gfx11_body<I8_EPI_SET>(A, nullptr, Xq, Y, nullptr, nullptr, nullptr, nullptr, M, K, N);
}

// Gate/up SwiGLU: G/U = gate/up MQ4V2 rows (M each), H = h [N][M].
extern "C" __global__ __launch_bounds__(I8_NTHREADS) __attribute__((amdgpu_num_vgpr(I8_NUM_VGPR)))
void I8_ENTRY(gemm_mq4g256v2_gate_up_silu_mmq_i8)(
    const char*           __restrict__ G,
    const char*           __restrict__ U,
    const block_i8_128*   __restrict__ Xq,
    float*                __restrict__ H,
    int M, int K, int N
) {
    gemm_mq4g256v2_residual_mmq_i8_gfx11_body<I8_EPI_SILU>(G, U, Xq, H, nullptr, nullptr, nullptr, nullptr, M, K, N);
}
#endif

#undef I8_SILU_MUL

#else

#ifdef I8_A8_EPI
extern "C" __global__ void gemm_mq4g256v2_gate_up_silu_a8_mmq_i8(
    const char*, const char*, const void*, const float*, const float*, const float*, void*, int, int, int
) {}
#else
extern "C" __global__ void I8_ENTRY(gemm_mq4g256v2_residual_mmq_i8_full_add)(
    const char*, const void*, float*, int, int, int, int
) {}

extern "C" __global__ void I8_ENTRY(gemm_mq4g256v2_residual_mmq_i8_full_set)(
    const char*, const void*, float*, int, int, int, int
) {}

extern "C" __global__ void I8_ENTRY(gemm_mq4g256v2_gate_up_silu_mmq_i8)(
    const char*, const char*, const void*, float*, int, int, int
) {}
#endif

#endif

#undef I8_ENTRY
