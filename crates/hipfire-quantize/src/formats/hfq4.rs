use crate::strategy::ScaleStrategy;

// ─── HFQ4-G256 Quantization ────────────────────────────────────────────────

/// Quantize F32 weights to HFQ4-G256: flat 4-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][128B nibbles] = 136 bytes per 256 weights.
pub fn quantize_hfq4g256(f32_data: &[f32], strategy: &dyn ScaleStrategy) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let (scale, zero) = strategy.solve_affine(group, 16);
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&zero.to_le_bytes());

        let actual_len = end - start;
        // Pack 256 weights into 128 bytes of nibbles
        // byte[i] = weight[2*i] (lo nibble) | weight[2*i+1] (hi nibble)
        for i in 0..128 {
            let idx_lo = 2 * i;
            let idx_hi = 2 * i + 1;
            let lo_val = if idx_lo < actual_len { group[idx_lo] } else { zero };
            let hi_val = if idx_hi < actual_len { group[idx_hi] } else { zero };

            let lo_q = ((lo_val - zero) * inv_scale + 0.5).max(0.0).min(15.0) as u8;
            let hi_q = ((hi_val - zero) * inv_scale + 0.5).max(0.0).min(15.0) as u8;

            output[out_off + 8 + i] = lo_q | (hi_q << 4);
        }
    }

    output
}

// ─── HFQ4-G128 Quantization ────────────────────────────────────────────────

/// Quantize F32 weights to HFQ4-G128: flat 4-bit with 128-weight groups.
/// Block: [f32 scale][f32 zero][64B nibbles] = 72 bytes per 128 weights (0.5625 B/w).
/// 14 VGPRs, 100% occupancy. Better quality for small K dimensions.
pub fn quantize_hfq4g128(f32_data: &[f32], strategy: &dyn ScaleStrategy) -> Vec<u8> {
    let group_size = 128;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let (scale, zero) = strategy.solve_affine(group, 16);
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&zero.to_le_bytes());

        let actual_len = end - start;
        for i in 0..64 {
            let idx_lo = 2 * i;
            let idx_hi = 2 * i + 1;
            let lo_val = if idx_lo < actual_len { group[idx_lo] } else { zero };
            let hi_val = if idx_hi < actual_len { group[idx_hi] } else { zero };

            let lo_q = ((lo_val - zero) * inv_scale + 0.5).max(0.0).min(15.0) as u8;
            let hi_q = ((hi_val - zero) * inv_scale + 0.5).max(0.0).min(15.0) as u8;

            output[out_off + 8 + i] = lo_q | (hi_q << 4);
        }
    }

    output
}
