use crate::strategy::ScaleStrategy;

// ─── HFQ6-G256 Quantization ────────────────────────────────────────────────

/// Quantize F32 weights to HFQ6-G256: 6-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][192B packed 6-bit] = 200 bytes per 256 weights (0.78125 B/w).
pub fn quantize_hfq6g256(f32_data: &[f32], strategy: &dyn ScaleStrategy) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 200; // 8 (scale+zero) + 192 (packed 6-bit)
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let (scale, zero) = strategy.solve_affine(group, 64);
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&zero.to_le_bytes());

        let actual_len = end - start;
        // Pack 4 values per 3 bytes: v0[5:0]|v1[1:0], v1[5:2]|v2[3:0], v2[5:4]|v3[5:0]
        for i in (0..256).step_by(4) {
            let q0 = if i < actual_len { ((group[i] - zero) * inv_scale + 0.5).max(0.0).min(63.0) as u8 } else { 0 };
            let q1 = if i + 1 < actual_len { ((group[i+1] - zero) * inv_scale + 0.5).max(0.0).min(63.0) as u8 } else { 0 };
            let q2 = if i + 2 < actual_len { ((group[i+2] - zero) * inv_scale + 0.5).max(0.0).min(63.0) as u8 } else { 0 };
            let q3 = if i + 3 < actual_len { ((group[i+3] - zero) * inv_scale + 0.5).max(0.0).min(63.0) as u8 } else { 0 };

            let byte_off = 8 + (i / 4) * 3;
            output[out_off + byte_off]     = q0 | (q1 << 6);
            output[out_off + byte_off + 1] = (q1 >> 2) | (q2 << 4);
            output[out_off + byte_off + 2] = (q2 >> 4) | (q3 << 2);
        }
    }
    output
}
