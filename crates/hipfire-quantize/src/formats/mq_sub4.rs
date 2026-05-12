use crate::formats::fwht::cpu_fwht_256;
use crate::hfq_writer::f32_to_fp16_bits;
use crate::strategy::ScaleStrategy;

/// MagnumQuant MQ3-G256: FWHT-rotated 3-bit quantization.
/// Block: [f32 scale][f32 zero][96B packed 3-bit] = 104 bytes per 256 weights.
pub fn quantize_mq3g256(
    f32_data: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    strategy: &dyn ScaleStrategy,
) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 104;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        // FWHT rotation — equalizes outliers across the group (QuIP#-style RHT)
        cpu_fwht_256(&mut group, signs1, signs2);

        let (scale, zero) = strategy.solve_affine(&group, 8);
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&zero.to_le_bytes());

        // Pack 256 weights as 32 chunks of 8 weights x 3 bits = 3 bytes each.
        // Bit layout matches the HFQ3-G256 GEMV kernel unpack (cross-byte).
        for chunk in 0..32 {
            let ci = chunk * 8;
            let mut q = [0u8; 8];
            for j in 0..8 {
                q[j] = ((group[ci + j] - zero) * inv_scale + 0.5).clamp(0.0, 7.0) as u8;
            }
            let b0 = (q[0] & 7) | ((q[1] & 7) << 3) | ((q[2] & 3) << 6);
            let b1 = ((q[2] >> 2) & 1) | ((q[3] & 7) << 1) | ((q[4] & 7) << 4) | ((q[5] & 1) << 7);
            let b2 = ((q[5] >> 1) & 3) | ((q[6] & 7) << 2) | ((q[7] & 7) << 5);

            let bo = out_off + 8 + chunk * 3;
            output[bo] = b0;
            output[bo + 1] = b1;
            output[bo + 2] = b2;
        }
    }

    output
}

/// MagnumQuant MQ2-G256: FWHT-rotated 2-bit quantization.
/// Same binary format as HFQ2-G256 (72 bytes/group). Rotation is baked into
/// the weights via cpu_fwht_256; the GEMV kernel rotates x instead.
pub fn quantize_mq2g256(
    f32_data: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    strategy: &dyn ScaleStrategy,
) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        cpu_fwht_256(&mut group, signs1, signs2);

        let (scale, zero) = strategy.solve_affine(&group, 4);
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&zero.to_le_bytes());

        // Pack 256 weights into 64 bytes (4 per byte at 2-bit).
        for i in 0..64 {
            let mut byte_val = 0u8;
            for j in 0..4 {
                let q = ((group[4 * i + j] - zero) * inv_scale + 0.5) as u8;
                byte_val |= q.min(3) << (j * 2);
            }
            output[out_off + 8 + i] = byte_val;
        }
    }

    output
}

/// MagnumQuant HFQ3-G256-Lloyd: per-block 8-entry fp16 codebook fitted via
/// Lloyd's algorithm. 16 B header (8 fp16) + 96 B packed 3-bit indices = 112 B/group
/// (vs uniform MQ3's 104 B — only +7.7% bandwidth). Direct extension of MQ2-Lloyd
/// with K=8; targets sub-9B MQ3 collapse rescue (#114) and 9B MQ3 -> MQ4 ppl gap.
pub fn quantize_mq3g256_lloyd(
    f32_data: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    _strategy: &dyn ScaleStrategy,
) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 112;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;

            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            // Lloyd-Max has its own scale logic — strategy is unused.
            let elem_weights = [1.0f32; 256];

            // Initial centroid placement: 8 evenly-spaced percentiles
            // (1/16, 3/16, ..., 15/16) of the rotated block.
            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut cb: [f32; 8] = [0.0; 8];
            for k in 0..8 {
                let frac = (2 * k + 1) as f32 / 16.0;
                let idx = ((frac * 255.0).round() as usize).min(255);
                cb[k] = sorted[idx];
            }

            let range = sorted[255] - sorted[0];
            let mut indices = [0u8; 256];
            if range > 0.0 {
                let max_iter = 8;
                let mut prev_assignments = [0u8; 256];
                for it in 0..max_iter {
                    let mut sums = [0.0f64; 8];
                    let mut wt_sums = [0.0f64; 8];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let wi = elem_weights[i] as f64;
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..8 {
                            let d = (w - cb[k]).abs();
                            if d < best_d { best_d = d; best = k; }
                        }
                        if it == 0 || prev_assignments[i] != best as u8 { changed += 1; }
                        prev_assignments[i] = best as u8;
                        indices[i] = best as u8;
                        sums[best] += wi * w as f64;
                        wt_sums[best] += wi;
                    }
                    if it > 0 && changed == 0 { break; }
                    for k in 0..8 {
                        if wt_sums[k] > 0.0 {
                            cb[k] = (sums[k] / wt_sums[k]) as f32;
                        }
                    }
                }
            }

            // Sort centroids ascending; remap indices.
            let mut order: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            order.sort_by(|&a, &b| cb[a].partial_cmp(&cb[b]).unwrap_or(std::cmp::Ordering::Equal));
            let mut sorted_cb = [0.0f32; 8];
            let mut inv: [u8; 8] = [0; 8];
            for new_idx in 0..8 {
                sorted_cb[new_idx] = cb[order[new_idx]];
                inv[order[new_idx]] = new_idx as u8;
            }
            for i in 0..256 { indices[i] = inv[indices[i] as usize]; }

            // Header: 8 fp16 centroids = 16 bytes.
            for k in 0..8 {
                let bits = f32_to_fp16_bits(sorted_cb[k]);
                out_chunk[2 * k]     = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }

            // Data: 96 bytes — same cross-byte 3-bit packing as uniform MQ3, so
            // the kernel unpack code is identical (only the recon changes from
            // `scale*q + zero` to `cb[q]`).
            for chunk in 0..32 {
                let ci = chunk * 8;
                let q = [
                    indices[ci]     & 7, indices[ci + 1] & 7, indices[ci + 2] & 7, indices[ci + 3] & 7,
                    indices[ci + 4] & 7, indices[ci + 5] & 7, indices[ci + 6] & 7, indices[ci + 7] & 7,
                ];
                let b0 = q[0] | (q[1] << 3) | ((q[2] & 3) << 6);
                let b1 = (q[2] >> 2) | (q[3] << 1) | (q[4] << 4) | ((q[5] & 1) << 7);
                let b2 = (q[5] >> 1) | (q[6] << 2) | (q[7] << 5);
                let bo = 16 + chunk * 3;
                out_chunk[bo] = b0;
                out_chunk[bo + 1] = b1;
                out_chunk[bo + 2] = b2;
            }
        });

    output
}

/// MagnumQuant HFQ2-G256-Lloyd: per-block 4-entry fp16 codebook fitted via
/// Lloyd's algorithm to minimize squared reconstruction error on FWHT-rotated
/// weights. 8 B header (4 fp16) + 64 B packed 2-bit indices = 72 B/group —
/// bandwidth-identical to uniform MQ2. The "true non-uniform 4-entry codebook"
/// described in `docs/plans/mq-sub4bit-research-queue.md` Q1.
pub fn quantize_mq2g256_lloyd(
    f32_data: &[f32],
    signs1: &[f32],
    signs2: &[f32],
    _strategy: &dyn ScaleStrategy,
) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    // Parallelize across blocks: each block is independent (own FWHT, own
    // Lloyd's iterations, own centroids). On 24-core boxes this is ~10-15x over
    // the serial path on 9B (single tensor can have >20M blocks).
    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;

            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            // Lloyd-Max has its own scale logic — strategy is unused.
            let elem_weights = [1.0f32; 256];

            // Initial centroid placement: percentiles of the rotated block.
            // 12.5/37.5/62.5/87.5 gives a good starting partition — heavy-tail
            // blocks adapt across iterations.
            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile = |frac: f32| -> f32 {
                let idx = ((frac * 255.0).round() as usize).min(255);
                sorted[idx]
            };
            let mut cb: [f32; 4] = [
                percentile(0.125),
                percentile(0.375),
                percentile(0.625),
                percentile(0.875),
            ];

            let range = sorted[255] - sorted[0];
            let mut indices = [0u8; 256];
            if range > 0.0 {
                // Lloyd's iterations — cap at 8, early-exit on stable assignments.
                // Empirically Lloyd's converges in 4-6 iter for FWHT-rotated weight
                // distributions; the 12-iter cap was wasteful.
                let max_iter = 8;
                let mut prev_assignments = [0u8; 256];
                for it in 0..max_iter {
                    let mut sums = [0.0f64; 4];
                    let mut wt_sums = [0.0f64; 4];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let wi = elem_weights[i] as f64;
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..4 {
                            let d = (w - cb[k]).abs();
                            if d < best_d { best_d = d; best = k; }
                        }
                        if it == 0 || prev_assignments[i] != best as u8 { changed += 1; }
                        prev_assignments[i] = best as u8;
                        indices[i] = best as u8;
                        sums[best] += wi * w as f64;
                        wt_sums[best] += wi;
                    }
                    if it > 0 && changed == 0 { break; }
                    for k in 0..4 {
                        if wt_sums[k] > 0.0 {
                            cb[k] = (sums[k] / wt_sums[k]) as f32;
                        }
                    }
                }
            }

            // Sort centroids ascending; remap indices to keep header canonical
            // and the permutation deterministic across re-runs.
            let mut order: [usize; 4] = [0, 1, 2, 3];
            order.sort_by(|&a, &b| cb[a].partial_cmp(&cb[b]).unwrap_or(std::cmp::Ordering::Equal));
            let mut sorted_cb = [0.0f32; 4];
            let mut inv: [u8; 4] = [0; 4];
            for new_idx in 0..4 {
                sorted_cb[new_idx] = cb[order[new_idx]];
                inv[order[new_idx]] = new_idx as u8;
            }
            for i in 0..256 { indices[i] = inv[indices[i] as usize]; }

            for k in 0..4 {
                let bits = f32_to_fp16_bits(sorted_cb[k]);
                out_chunk[2 * k]     = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }
            // 256 indices x 2 bits = 64 bytes. Same packing as uniform MQ2.
            for i in 0..64 {
                let mut byte_val = 0u8;
                for j in 0..4 { byte_val |= (indices[4 * i + j] & 0x3) << (j * 2); }
                out_chunk[8 + i] = byte_val;
            }
        });

    output
}
