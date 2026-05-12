use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

// ─── FP16/BF16 Conversion ───────────────────────────────────────────────────

pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign << 31); }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        let frac32 = if frac == 0 { 0 } else { frac << 13 | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0xFF {
        let f16_frac = if frac == 0 { 0 } else { (frac >> 13) | 1 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 { return ((sign << 15) | (0x1F << 10)) as u16; }
    if new_exp <= 0 {
        if new_exp < -10 { return (sign << 15) as u16; }
        let f = frac | 0x800000;
        let shift = (1 - new_exp + 13) as u32;
        return ((sign << 15) | (f >> shift)) as u16;
    }
    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

/// Convert raw tensor bytes to F32 based on dtype string
pub fn to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F16" => {
            data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        "BF16" => {
            data.chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        "F32" => {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        other => panic!("unsupported dtype: {other}"),
    }
}

/// Encode an f32 to IEEE-754 fp16 bits (round-to-nearest-even, no NaN/Inf preservation
/// beyond the trivial case — block centroids are bounded means of fp32 weights so
/// the simple path is safe).
pub fn f32_to_fp16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xFF) as i32;
    let mant = (bits & 0x7FFFFF) as u32;
    if exp == 0xFF {
        // Inf or NaN
        let m16 = if mant != 0 { 0x200 } else { 0 };
        return sign | 0x7C00 | m16;
    }
    exp -= 127 - 15;
    if exp >= 0x1F {
        return sign | 0x7C00; // overflow -> +-Inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // underflow -> +-0
        }
        // Subnormal: shift mantissa
        let m = mant | 0x800000;
        let shift = (1 - exp) as u32 + 13;
        let mut m16 = (m >> shift) as u16;
        // Round-half-to-even via remainder
        let lost = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if lost > half || (lost == half && (m16 & 1) == 1) {
            m16 = m16.wrapping_add(1);
        }
        return sign | m16;
    }
    let mut m16 = (mant >> 13) as u16;
    let lost = mant & 0x1FFF;
    if lost > 0x1000 || (lost == 0x1000 && (m16 & 1) == 1) {
        m16 = m16.wrapping_add(1);
        if m16 == 0x400 {
            // Mantissa overflow -> carry into exponent
            m16 = 0;
            exp += 1;
            if exp >= 0x1F { return sign | 0x7C00; }
        }
    }
    sign | ((exp as u16) << 10) | m16
}

// ─── HFQ File Format ────────────────────────────────────────────────────────

pub const HFQ_MAGIC: &[u8; 4] = b"HFQM";
pub const HFQ_VERSION: u32 = 1;

#[repr(u8)]
#[derive(Clone, Copy)]
pub enum QuantType {
    Q4F16G64 = 0,
    F16 = 1,
    F32 = 2,
    Q8F16 = 3,
    Q4K = 4,
    Q8HFQ = 5,
    HFQ4G256 = 6,
    HFQ4G128 = 7,
    HFQ6G256 = 8,
    HFQ2G256 = 9,
    HFQ2G128 = 10,
    HFQ3G256 = 11,
    HFQ3G128 = 12,
    MQ4G256 = 13,  // MagnumQuant: FWHT-rotated HFQ4-G256
    MQ8G256 = 14,  // MagnumQuant: FWHT-rotated symmetric INT8, dp4a target
    MQ6G256 = 15,  // MagnumQuant: FWHT-rotated HFQ6-G256 (6-bit, 200 B/group)
    BF16 = 16,     // Original BF16 weights (zero precision loss for vision)
    MQ3G256 = 17,  // MagnumQuant: FWHT-rotated HFQ3-G256 (3-bit, 104 B/group)
    MQ2G256 = 18,  // MagnumQuant: FWHT-rotated HFQ2-G256 (2-bit, 72 B/group)
    MQ2G256Lloyd = 19, // MagnumQuant 2-bit + per-block Lloyd-Max 4-entry fp16 codebook (72 B/group)
    MQ3G256Lloyd = 20, // MagnumQuant 3-bit + per-block Lloyd-Max 8-entry fp16 codebook (112 B/group)
    // HFP4 family — RDNA-optimal FP4 (E2M1 elements + UE8M0 block scale + FP16 row scale).
    // See docs/quant-formats/hfp4.md for byte layout, dequant, rotation modes.
    // Per-row header is 16 B; per-block payload is (1 + g/2) bytes (UE8M0 + nibbles).
    HFP4G32 = 21,      // E2M1 + UE8M0 g32 + FP16 row scale — canonical (FP8-WMMA-K aligned)
    // MFP4G32 = HFP4G32 + offline FWHT rotation (256-element FWHT applied to weights at quant time;
    // runtime applies the same FWHT to x via mq_rotate_x). format_flags bit 0 + bits 2-3 = 0b0101
    // signals "rotation present, offline FWHT" for future interop/detection.
    MFP4G32 = 24,      // v1.5 — HFP4G32 + offline FWHT (drop-in MQ4 replacement)
    // Reserved IDs — DO NOT REUSE for unrelated formats. Documented in docs/quant-formats/hfp4.md.
    // HFP4G16     = 22, // v1.5 — NV-aligned FP16-WMMA-K alignment ablation
    // HFP4G64     = 23, // v1.5 — RDNA1/2 sweet-spot ablation
    // HFP4G32MX   = 25, // v2  — strict OCP MXFP4 interop alias (no row scale, UE8M0 only)
    // HFP4G16NV   = 26, // v2  — strict NVFP4 interop alias (E4M3 scale + FP32 tensor)
    // HFP8E4M3G32 = 27, // v2  — HFP8 E4M3 family
    // HFP8E5M2G32 = 28, // v2  — HFP8 E5M2 family
    // MFP4G32R    = 29, // v3  — HFP4G32 + online block-diag-128 rotation (AMD recipe)
}

pub struct HfqTensor {
    pub name: String,
    pub quant_type: QuantType,
    pub shape: Vec<u32>,
    pub group_size: u32,
    pub data: Vec<u8>,
    /// When data is spilled to disk, this holds the byte count.
    /// `data` is empty and the bytes live in the spill file.
    pub spilled_len: u64,
}

/// Streaming tensor spill file. When the quantizer accumulates more than
/// `SPILL_THRESHOLD` bytes of tensor data in memory, it flushes completed
/// tensors to this file. At write_hfq time, spilled data is copied from
/// the spill file instead of from memory, keeping peak RSS bounded.
pub struct TensorSpill {
    file: std::io::BufWriter<File>,
    pub path: PathBuf,
    offset: u64,
}

impl TensorSpill {
    pub fn new(dir: &Path) -> std::io::Result<Self> {
        let path = dir.join(".hipfire_quant_spill.tmp");
        let file = std::io::BufWriter::with_capacity(
            4 * 1024 * 1024,
            File::create(&path)?,
        );
        Ok(Self { file, path, offset: 0 })
    }

    /// Write tensor data to the spill file. Returns the byte count written.
    pub fn spill(&mut self, data: &[u8]) -> std::io::Result<u64> {
        self.file.write_all(data)?;
        self.offset += data.len() as u64;
        Ok(data.len() as u64)
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }

    pub fn cleanup(self) {
        // Explicit cleanup — Drop impl handles the actual removal.
        drop(self);
    }
}

impl Drop for TensorSpill {
    fn drop(&mut self) {
        // Ensure the temp file is removed even on panic.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Spill tensors whose data is in memory to the spill file, freeing RAM.
/// Called after each layer's expert batch to keep peak RSS bounded.
pub fn maybe_spill(tensors: &mut [HfqTensor], spill: &mut TensorSpill, threshold: usize) {
    let in_mem: usize = tensors.iter().filter(|t| t.spilled_len == 0).map(|t| t.data.len()).sum();
    if in_mem < threshold { return; }
    for t in tensors.iter_mut() {
        if t.spilled_len == 0 && !t.data.is_empty() {
            let len = spill.spill(&t.data).unwrap_or(0);
            t.spilled_len = len;
            t.data = Vec::new(); // free the memory
        }
    }
    let _ = spill.flush();
}

pub fn write_hfq(
    path: &Path,
    arch: u32,
    metadata_json: &str,
    tensors: &[HfqTensor],
    spill: Option<&mut TensorSpill>,
) -> std::io::Result<()> {
    let mut f = File::create(path)?;

    let metadata_bytes = metadata_json.as_bytes();

    // Calculate offsets
    let header_size = 32u64;
    let metadata_offset = header_size;
    let metadata_size = metadata_bytes.len() as u64;

    // Tensor index follows metadata
    let index_offset = metadata_offset + metadata_size;
    let mut index_bytes = Vec::new();
    // Write tensor count
    index_bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        // name length + name
        let name_bytes = t.name.as_bytes();
        index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(name_bytes);
        // quant type
        index_bytes.push(t.quant_type as u8);
        // n_dims + shape
        index_bytes.push(t.shape.len() as u8);
        for &d in &t.shape {
            index_bytes.extend_from_slice(&d.to_le_bytes());
        }
        // group size
        index_bytes.extend_from_slice(&t.group_size.to_le_bytes());
        // data size (offset computed at read time from cumulative sizes)
        let data_len = if t.spilled_len > 0 { t.spilled_len } else { t.data.len() as u64 };
        index_bytes.extend_from_slice(&data_len.to_le_bytes());
    }

    // Data starts after index, aligned to 4096
    let data_start_unaligned = index_offset + index_bytes.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;

    // Write header (32 bytes)
    f.write_all(HFQ_MAGIC)?;
    f.write_all(&HFQ_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;

    // Write metadata
    f.write_all(metadata_bytes)?;

    // Write tensor index
    f.write_all(&index_bytes)?;

    // Pad to data alignment
    let pad_size = (data_offset - data_start_unaligned) as usize;
    f.write_all(&vec![0u8; pad_size])?;

    // Write tensor data — from spill file or from memory
    if let Some(spill) = spill {
        let _ = spill.flush();
        let mut spill_reader = std::io::BufReader::new(
            File::open(&spill.path)?
        );
        let mut buf = vec![0u8; 4 * 1024 * 1024]; // 4 MB copy buffer
        for t in tensors {
            if t.spilled_len > 0 {
                // Copy from spill file
                let mut remaining = t.spilled_len as usize;
                while remaining > 0 {
                    let chunk = remaining.min(buf.len());
                    use std::io::Read;
                    spill_reader.read_exact(&mut buf[..chunk])?;
                    f.write_all(&buf[..chunk])?;
                    remaining -= chunk;
                }
            } else {
                f.write_all(&t.data)?;
            }
        }
    } else {
        for t in tensors {
            f.write_all(&t.data)?;
        }
    }

    Ok(())
}
