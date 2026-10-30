// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.
//! raon_oracle `mla` module: runs the PRODUCTION latent-attention stages
//! (`execute_latent_attention`) on GPU against the reference fixtures
//! `<fixtures>/mla/<case>.safetensors` (+ optional shared `weights` file).
//!
//! Fixture tensors (F32 holding BF16-rounded values; mode "bf16" only):
//!   inputs : q_a_out[N,qlora] kv_a_out[N,latent+tail] q_a_norm[qlora]
//!            kv_a_norm[latent] q_proj[N,H*(nope+tail)] kv_b[H*(nope+v),latent]
//!            g_out[N,H*v] cache_prefix[P,latent+tail]
//!   outputs: qlat[N,qlora] cache_out[P+N,latent+tail]
//!            kv_expanded[P+N,H,nope+v] attn_y[N,H*v] gated[N,H*v]
//! dims: {N,P,H,nope,tail,v,latent,qlora}.
//!
//! Checks per case: qlat / cache rows / expanded K,V / attention / gate against
//! the reference (PLAN §6.2 BF16 band: rel_l2<=1e-2, cosine>=.9999, no
//! localized outlier); prefix rows bit-preserved; BITWISE equality of the
//! mirror built by two different decompression schedules and by the
//! gather -> external projector -> scatter route; repeated attention
//! byte-identical; cache poison beyond the live length never read.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use hipfire_dispatch::pipeline::latent_attention::{
    execute_latent_attention, ExpandedMirror, KvExpansion, LatentAttentionOp, LatentCache,
    LatentGeometry, RowRange,
};
use rdna_compute::{DType, Gpu, GpuTensor};
use serde_json::{json, Value};

const BF16_REL_L2: f64 = 1e-2;
const BF16_COSINE: f64 = 0.9999;
/// Max-abs gate as a fraction of max|reference| (catches localized corruption).
const BF16_MAX_ABS_FRAC: f64 = 2e-2;
/// BF16 bits of 1e30: finite, huge — any read of a poisoned row wrecks softmax.
const POISON_BF16: u16 = 0x7149;
const EXTERNAL_ROW_LIMIT: usize = 130;

// ── safetensors (F32 / BF16 / F16 → f32) ────────────────────────────────────

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn f32_to_bf16(f: f32) -> u16 {
    let u = f.to_bits();
    if (u & 0x7fff_ffff) > 0x7f80_0000 {
        return ((u >> 16) | 0x40) as u16;
    }
    (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s
        } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            s | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 {
        s | 0x7f80_0000 | (m << 13)
    } else {
        s | (((e - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

struct Tensors {
    map: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Tensors {
    fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        if bytes.len() < 8 {
            return Err(format!("{}: truncated safetensors", path.display()));
        }
        let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let hdr_end = 8usize
            .checked_add(hlen)
            .filter(|e| *e <= bytes.len())
            .ok_or_else(|| format!("{}: bad header length", path.display()))?;
        let header: Value = serde_json::from_slice(&bytes[8..hdr_end])
            .map_err(|e| format!("{}: header json: {e}", path.display()))?;
        let data = &bytes[hdr_end..];
        let mut map = HashMap::new();
        for (name, meta) in header.as_object().ok_or("header not an object")? {
            if name == "__metadata__" {
                continue;
            }
            let dtype = meta["dtype"].as_str().ok_or("tensor dtype missing")?;
            let shape: Vec<usize> = meta["shape"]
                .as_array()
                .ok_or("tensor shape missing")?
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = meta["data_offsets"].as_array().ok_or("data_offsets missing")?;
            let (a, b) = (
                off[0].as_u64().ok_or("offset")? as usize,
                off[1].as_u64().ok_or("offset")? as usize,
            );
            let raw = data
                .get(a..b)
                .ok_or_else(|| format!("{name}: offsets outside file"))?;
            let vals: Vec<f32> = match dtype {
                "F32" => raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
                "BF16" => raw
                    .chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect(),
                "F16" => raw
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect(),
                other => return Err(format!("{name}: unsupported dtype {other}")),
            };
            if vals.len() != shape.iter().product::<usize>() {
                return Err(format!("{name}: {} values for shape {shape:?}", vals.len()));
            }
            map.insert(name.clone(), (shape, vals));
        }
        Ok(Self { map })
    }

    fn merge(&mut self, other: Tensors) {
        for (k, v) in other.map {
            self.map.entry(k).or_insert(v);
        }
    }

    fn get(&self, name: &str, expect: usize) -> Result<&[f32], String> {
        let (_, v) = self
            .map
            .get(name)
            .ok_or_else(|| format!("fixture tensor `{name}` missing"))?;
        if v.len() != expect {
            return Err(format!("fixture tensor `{name}` has {} values, expected {expect}", v.len()));
        }
        Ok(v)
    }
}

// ── metrics ─────────────────────────────────────────────────────────────────

struct Cmp {
    ok: bool,
    json: Value,
}

fn compare(name: &str, got: &[f32], want: &[f32]) -> Cmp {
    let (mut num, mut den, mut dot, mut gg, mut max_abs, mut max_ref) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    let mut nonfinite = 0usize;
    for (g, w) in got.iter().zip(want) {
        if !g.is_finite() {
            nonfinite += 1;
        }
        let (g, w) = (*g as f64, *w as f64);
        let d = g - w;
        num += d * d;
        den += w * w;
        dot += g * w;
        gg += g * g;
        max_abs = max_abs.max(d.abs());
        max_ref = max_ref.max(w.abs());
    }
    let rel = if den > 0.0 { (num / den).sqrt() } else { num.sqrt() };
    let cos = if den > 0.0 && gg > 0.0 { dot / (den.sqrt() * gg.sqrt()) } else { 1.0 };
    let tol = 2e-5 + BF16_MAX_ABS_FRAC * max_ref;
    let first = got
        .iter()
        .zip(want)
        .position(|(g, w)| !g.is_finite() || ((*g - *w).abs() as f64) > tol);
    let ok = got.len() == want.len()
        && nonfinite == 0
        && rel <= BF16_REL_L2
        && cos >= BF16_COSINE
        && first.is_none();
    let mut j = json!({
        "tensor": name, "pass": ok, "rel_l2": rel, "cosine": cos, "max_abs": max_abs,
        "max_abs_ref": max_ref, "band": {"rel_l2": BF16_REL_L2, "cosine": BF16_COSINE,
        "max_abs_tol": tol}, "len": got.len(), "nonfinite": nonfinite,
    });
    if let Some(i) = first {
        j["first_mismatch"] = json!({"index": i, "got": got[i], "want": want[i]});
    }
    Cmp { ok, json: j }
}

fn exact(name: &str, a: &[u8], b: &[u8]) -> Cmp {
    let first = a.iter().zip(b).position(|(x, y)| x != y);
    let ok = a.len() == b.len() && first.is_none();
    let mut j = json!({"tensor": name, "pass": ok, "bitwise": true, "bytes": a.len()});
    if let Some(i) = first {
        j["first_mismatch"] = json!({"byte": i, "a": a[i], "b": b[i]});
    } else if a.len() != b.len() {
        j["first_mismatch"] = json!({"len_a": a.len(), "len_b": b.len()});
    }
    Cmp { ok, json: j }
}

// ── device helpers ──────────────────────────────────────────────────────────

fn he<T: std::fmt::Display>(e: T) -> String {
    e.to_string()
}

fn up_f32(gpu: &mut Gpu, v: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    gpu.upload_f32(v, shape).map_err(he)
}

fn up_bf16(gpu: &mut Gpu, v: &[f32], shape: &[usize]) -> Result<GpuTensor, String> {
    let t = gpu.alloc_tensor(shape, DType::BF16).map_err(he)?;
    let bytes: Vec<u8> = v.iter().flat_map(|f| f32_to_bf16(*f).to_le_bytes()).collect();
    gpu.memcpy_htod_auto(&t.buf, &bytes).map_err(he)?;
    Ok(t)
}

fn poison_bf16(gpu: &mut Gpu, shape: &[usize]) -> Result<GpuTensor, String> {
    let t = gpu.alloc_tensor(shape, DType::BF16).map_err(he)?;
    let n: usize = shape.iter().product();
    let bytes: Vec<u8> = (0..n).flat_map(|_| POISON_BF16.to_le_bytes()).collect();
    gpu.memcpy_htod_auto(&t.buf, &bytes).map_err(he)?;
    Ok(t)
}

fn bytes_bf16(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

fn set_pos(gpu: &mut Gpu, pos: &GpuTensor, p: usize) -> Result<(), String> {
    gpu.memcpy_htod_auto(&pos.buf, &(p as i32).to_le_bytes()).map_err(he)
}

/// Head-major mirrors → `[rows, H, nope+v]` F32 in the reference layout.
fn expand_mirror(k: &[u8], v: &[u8], g: &LatentGeometry, cap: usize, rows: usize) -> Vec<f32> {
    let kf = bytes_bf16(k);
    let vf = bytes_bf16(v);
    let per = g.nope + g.v_dim;
    let mut out = vec![0f32; rows * g.n_heads * per];
    for t in 0..rows {
        for h in 0..g.n_heads {
            let dst = (t * g.n_heads + h) * per;
            let ks = (h * cap + t) * g.nope;
            out[dst..dst + g.nope].copy_from_slice(&kf[ks..ks + g.nope]);
            let vs = (h * cap + t) * g.v_dim;
            out[dst + g.nope..dst + per].copy_from_slice(&vf[vs..vs + g.v_dim]);
        }
    }
    out
}

/// Bytes of the valid rows of one mirror ([H][cap][w], rows `0..rows`).
fn valid_rows(bytes: &[u8], heads: usize, cap: usize, w: usize, rows: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(heads * rows * w * 2);
    for h in 0..heads {
        let s = h * cap * w * 2;
        out.extend_from_slice(&bytes[s..s + rows * w * 2]);
    }
    out
}

fn dim(dims: &Value, k: &str) -> Result<usize, String> {
    dims.get(k)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| format!("manifest dims missing `{k}`"))
}

// ── one case ────────────────────────────────────────────────────────────────

fn run_case(gpu: &mut Gpu, dir: &Path, case: &Value) -> Result<Value, String> {
    let name = case["name"].as_str().ok_or("case name missing")?.to_string();
    if case.get("mode").and_then(Value::as_str).unwrap_or("bf16") != "bf16" {
        return Err(format!("case {name}: only mode \"bf16\" is implemented for mla"));
    }
    let dims = case.get("dims").ok_or("case dims missing")?;
    let (n, p) = (dim(dims, "N")?, dim(dims, "P")?);
    let g = LatentGeometry {
        n_heads: dim(dims, "H")?,
        nope: dim(dims, "nope")?,
        tail: dim(dims, "tail")?,
        v_dim: dim(dims, "v")?,
        latent: dim(dims, "latent")?,
    };
    let qlora = dim(dims, "qlora")?;
    let file = case["file"].as_str().ok_or("case file missing")?;
    let mut t = Tensors::load(&dir.join(file))?;
    if let Some(w) = case.get("weights").and_then(Value::as_str) {
        t.merge(Tensors::load(&dir.join(w))?);
    }
    let (width, dqk) = (g.row_width(), g.qk_dim());
    let (h, v, per) = (g.n_heads, g.v_dim, g.nope + g.v_dim);
    let total = p + n;
    let cap = total + 37; // poison rows beyond the live length
    let scale = g.default_scale();

    let q_a_out = t.get("q_a_out", n * qlora)?.to_vec();
    let kv_a_out = t.get("kv_a_out", n * width)?.to_vec();
    let q_a_norm = t.get("q_a_norm", qlora)?.to_vec();
    let kv_a_norm = t.get("kv_a_norm", g.latent)?.to_vec();
    let q_proj = t.get("q_proj", n * h * dqk)?.to_vec();
    let kv_b = t.get("kv_b", h * per * g.latent)?.to_vec();
    let g_out = t.get("g_out", n * h * v)?.to_vec();
    let cache_prefix = if p == 0 {
        Vec::new()
    } else {
        t.get("cache_prefix", p * width)?.to_vec()
    };
    let r_qlat = t.get("qlat", n * qlora)?;
    let r_cache = t.get("cache_out", total * width)?;
    let r_exp = t.get("kv_expanded", total * h * per)?;
    let r_y = t.get("attn_y", n * h * v)?;
    let r_gated = t.get("gated", n * h * v)?;

    // Device state: poisoned cache/mirrors; prefix rows uploaded as BF16.
    let cache_t = poison_bf16(gpu, &[cap, width])?;
    if p > 0 {
        let bytes: Vec<u8> = cache_prefix
            .iter()
            .flat_map(|f| f32_to_bf16(*f).to_le_bytes())
            .collect();
        gpu.memcpy_htod_auto(&cache_t.buf, &bytes).map_err(he)?;
    }
    let knope = poison_bf16(gpu, &[h, cap, g.nope])?;
    let vmir = poison_bf16(gpu, &[h, cap, v])?;
    let pos = gpu.alloc_tensor(&[1], DType::F32).map_err(he)?;
    set_pos(gpu, &pos, p)?;

    let d_qa = up_f32(gpu, &q_a_out, &[n, qlora])?;
    let d_kva = up_f32(gpu, &kv_a_out, &[n, width])?;
    let d_qg = up_f32(gpu, &q_a_norm, &[qlora])?;
    let d_kg = up_bf16(gpu, &kv_a_norm, &[g.latent])?; // BF16 gamma path
    let d_q = up_f32(gpu, &q_proj, &[n, h * dqk])?;
    let d_w = up_bf16(gpu, &kv_b, &[h * per, g.latent])?;
    let d_g = up_f32(gpu, &g_out, &[n, h * v])?;
    let d_qlat = gpu.zeros(&[n, qlora], DType::F32).map_err(he)?;
    let d_y = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let d_y2 = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;
    let d_gated = gpu.zeros(&[n, h * v], DType::F32).map_err(he)?;

    let cache = LatentCache { rows: &cache_t, capacity_rows: cap, pos: &pos, host_pos: Some(p) };
    let mirror = ExpandedMirror { knope: &knope, vmir: &vmir, capacity_rows: cap };
    let dense = KvExpansion::Dense { weight: &d_w };
    let run = |gpu: &mut Gpu, op: LatentAttentionOp<'_>| -> Result<(), String> {
        execute_latent_attention(gpu, &op).map_err(he)
    };

    // 1. q latent norm; 2. split into cache at device position P.
    run(gpu, LatentAttentionOp::NormRows {
        x: &d_qa, gamma: &d_qg, out: &d_qlat, rows: n, width: qlora, eps: 1e-6,
    })?;
    run(gpu, LatentAttentionOp::SplitKv {
        geometry: g, cache, kv_a: &d_kva, gamma: &d_kg, rows: n, eps: 1e-6,
    })?;
    // 3. schedule A: prefix rows in 37-row absolute tiles, new rows via device base.
    let mut s = 0;
    while s < p {
        let c = 37.min(p - s);
        run(gpu, LatentAttentionOp::Decompress {
            geometry: g, cache, mirror, expansion: dense, range: RowRange::Absolute { start: s, count: c },
        })?;
        s += c;
    }
    run(gpu, LatentAttentionOp::Decompress {
        geometry: g, cache, mirror, expansion: dense, range: RowRange::NewRows { count: n },
    })?;
    // 4. attention (twice, for byte determinism) and gate.
    for y in [&d_y, &d_y2] {
        run(gpu, LatentAttentionOp::Attention {
            geometry: g, cache, mirror, q: &d_q, y, rows: n, scale,
        })?;
    }
    run(gpu, LatentAttentionOp::Gate { y: &d_y, g: &d_g, out: &d_gated, elements: n * h * v })?;

    gpu.hip.device_synchronize().map_err(he)?;
    let qlat = gpu.download_f32(&d_qlat).map_err(he)?;
    let cache_bytes = gpu.download_raw_bytes(&cache_t).map_err(he)?;
    let k_a = gpu.download_raw_bytes(&knope).map_err(he)?;
    let v_a = gpu.download_raw_bytes(&vmir).map_err(he)?;
    let y1 = gpu.download_f32(&d_y).map_err(he)?;
    let y2 = gpu.download_f32(&d_y2).map_err(he)?;
    let gated = gpu.download_f32(&d_gated).map_err(he)?;

    let cache_f = bytes_bf16(&cache_bytes);
    let mut checks: Vec<Cmp> = Vec::new();
    checks.push(compare("qlat", &qlat, r_qlat));
    checks.push(compare("cache_out", &cache_f[..total * width], r_cache));
    // Prefix rows must be untouched (bit-preserved) by the split.
    let prefix_expect: Vec<u8> = cache_prefix
        .iter()
        .flat_map(|f| f32_to_bf16(*f).to_le_bytes())
        .collect();
    checks.push(exact("cache_prefix_preserved", &cache_bytes[..p * width * 2], &prefix_expect));
    // Tail channels of the new rows are the raw kv_a tail (never normalized).
    let new_tail: Vec<u8> = (0..n)
        .flat_map(|i| {
            kv_a_out[i * width + g.latent..(i + 1) * width]
                .iter()
                .flat_map(|f| f32_to_bf16(*f).to_le_bytes())
                .collect::<Vec<u8>>()
        })
        .collect();
    let got_tail: Vec<u8> = (0..n)
        .flat_map(|i| {
            let r = p + i;
            cache_bytes[(r * width + g.latent) * 2..((r + 1) * width) * 2].to_vec()
        })
        .collect();
    checks.push(exact("cache_tail_raw", &got_tail, &new_tail));
    let exp_a = expand_mirror(&k_a, &v_a, &g, cap, total);
    checks.push(compare("kv_expanded", &exp_a, r_exp));
    checks.push(compare("attn_y", &y1, r_y));
    checks.push(compare("gated", &gated, r_gated));
    checks.push(exact(
        "attn_y_repeat_bytewise",
        &y1.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>(),
        &y2.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>(),
    ));

    // 5. schedule B: every live row in one absolute call, fresh mirrors.
    let kn_b = poison_bf16(gpu, &[h, cap, g.nope])?;
    let vm_b = poison_bf16(gpu, &[h, cap, v])?;
    let mirror_b = ExpandedMirror { knope: &kn_b, vmir: &vm_b, capacity_rows: cap };
    run(gpu, LatentAttentionOp::Decompress {
        geometry: g, cache, mirror: mirror_b, expansion: dense,
        range: RowRange::Absolute { start: 0, count: total },
    })?;
    gpu.hip.device_synchronize().map_err(he)?;
    let k_b = gpu.download_raw_bytes(&kn_b).map_err(he)?;
    let v_b = gpu.download_raw_bytes(&vm_b).map_err(he)?;
    checks.push(exact(
        "mirror_k_schedules_bitwise",
        &valid_rows(&k_a, h, cap, g.nope, total),
        &valid_rows(&k_b, h, cap, g.nope, total),
    ));
    checks.push(exact(
        "mirror_v_schedules_bitwise",
        &valid_rows(&v_a, h, cap, v, total),
        &valid_rows(&v_b, h, cap, v, total),
    ));

    // 6. gather -> external projector (host, ascending-k fma) -> scatter.
    if total <= EXTERNAL_ROW_LIMIT {
        let plane = gpu.zeros(&[total, g.latent], DType::F32).map_err(he)?;
        run(gpu, LatentAttentionOp::GatherLatent {
            geometry: g, cache, range: RowRange::Absolute { start: 0, count: total }, out: &plane,
        })?;
        gpu.hip.device_synchronize().map_err(he)?;
        let lat = gpu.download_f32(&plane).map_err(he)?;
        let wq: Vec<f32> = kv_b.iter().map(|f| bf16_to_f32(f32_to_bf16(*f))).collect();
        let mut proj = vec![0f32; total * h * per];
        for r in 0..total {
            for o in 0..h * per {
                let mut acc = 0f32;
                for k in 0..g.latent {
                    acc = lat[r * g.latent + k].mul_add(wq[o * g.latent + k], acc);
                }
                proj[r * h * per + o] = acc;
            }
        }
        let d_proj = up_f32(gpu, &proj, &[total, h * per])?;
        let kn_c = poison_bf16(gpu, &[h, cap, g.nope])?;
        let vm_c = poison_bf16(gpu, &[h, cap, v])?;
        let mirror_c = ExpandedMirror { knope: &kn_c, vmir: &vm_c, capacity_rows: cap };
        run(gpu, LatentAttentionOp::Decompress {
            geometry: g, cache, mirror: mirror_c,
            expansion: KvExpansion::External { projected: &d_proj },
            range: RowRange::Absolute { start: 0, count: total },
        })?;
        gpu.hip.device_synchronize().map_err(he)?;
        let k_c = gpu.download_raw_bytes(&kn_c).map_err(he)?;
        let v_c = gpu.download_raw_bytes(&vm_c).map_err(he)?;
        checks.push(exact(
            "mirror_k_external_bitwise",
            &valid_rows(&k_a, h, cap, g.nope, total),
            &valid_rows(&k_c, h, cap, g.nope, total),
        ));
        checks.push(exact(
            "mirror_v_external_bitwise",
            &valid_rows(&v_a, h, cap, v, total),
            &valid_rows(&v_c, h, cap, v, total),
        ));
    }

    let pass = checks.iter().all(|c| c.ok);
    Ok(json!({
        "name": name, "pass": pass,
        "dims": {"N": n, "P": p, "H": h, "nope": g.nope, "tail": g.tail, "v": v,
                 "latent": g.latent, "qlora": qlora},
        "checks": checks.into_iter().map(|c| c.json).collect::<Vec<_>>(),
    }))
}

/// Entry point used by `raon_oracle module --module mla`.
pub fn run(fixtures: &Path, opts: &BTreeMap<String, String>) -> Result<Value, String> {
    let dir = fixtures.join("mla");
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(dir.join("manifest.json"))
            .map_err(|e| format!("{}: {e}", dir.join("manifest.json").display()))?,
    )
    .map_err(|e| format!("mla manifest.json: {e}"))?;
    let cases = manifest["cases"].as_array().ok_or("manifest.cases missing")?;
    if cases.is_empty() {
        return Err("mla manifest has no cases".into());
    }
    let filter = opts.get("case");
    let mut gpu = Gpu::init().map_err(he)?;
    let mut out = Vec::new();
    let mut all = true;
    let mut ran = 0usize;
    for case in cases {
        let nm = case["name"].as_str().unwrap_or("");
        if filter.is_some_and(|f| !nm.contains(f.as_str())) {
            continue;
        }
        ran += 1;
        let r = match run_case(&mut gpu, &dir, case) {
            Ok(v) => v,
            Err(e) => json!({"name": nm, "pass": false, "error": e}),
        };
        all &= r["pass"].as_bool().unwrap_or(false);
        out.push(r);
    }
    if ran == 0 {
        return Err(format!("no mla case matched filter {filter:?}"));
    }
    Ok(json!({
        "module": "mla", "pass": all, "cases_run": ran,
        "ref_sha256": manifest.get("ref_sha256").cloned().unwrap_or(Value::Null),
        "cases": out,
    }))
}
