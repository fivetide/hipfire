# Phase 2: Kernel Optimization Research

## Key Finding: llama.cpp Q4_K GEMV Architecture

llama.cpp achieves 65-70% of peak bandwidth through:
1. Pre-quantize x to Q8_1 format (4x less activation bandwidth)
2. dp4a packed int8 dot product (4 MADs per instruction on RDNA2+)
3. Single warp (32 threads) per row on RDNA — no shared memory needed
4. __launch_bounds__(..., 1) for max register use
5. Scales decoded to int, combined before float conversion
6. On RDNA1: inline ASM fallback (v_mul_i32_i24 + v_add3_u32)

## RDNA1 (gfx1010) Hardware Details

- Cache line: 128 bytes (GCN was 64 — key difference)
- Wave32 × 4 bytes = 128 bytes = one cache line (ideal coalescing)
- GL0 (L0): 16 KB per CU
- GL1 (new in RDNA): 128 KB per shader array
- L2: 4 MB shared
- LDS: 128 KB per WGP (2 CUs)
- VGPR: 128 KB per SIMD, 20 wavefront slots
- Optimal GEMV: 64-128 threads, uint4 loads, x in LDS

## Our Current Performance

- Q4K GEMV v5: 52 GB/s peak (11.7% of 448 GB/s)
- Gap to llama.cpp: ~6-7x (dp4a + Q8_1 pre-quant is the main missing piece)
- Deferred: dp4a implementation requires careful per-block scale handling
