// SPDX-License-Identifier: Apache-2.0
//! On-host collection of the schema-4 calibration (`hipfire calibrate-npu-trunk --measure`).
//!
//! Every cost is timed on this host through the serving pair executor of an `on` load: a GPU-only
//! pair (every layer native, its input projection timed whole and per family group), `on` pairs
//! with the NPU chains harvested from the relay, and GPU-only small-bank pairs. The key is the
//! runtime key serving derives from the same bundle, so the record matches by construction.
use super::*;
use rdna_compute::profile::{begin_deferred, resolve_deferred, PendingTimer};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// What one probe span times inside a bank of the pair executor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Phase {
    /// Embedding and PLE staging of the bank.
    Init,
    /// The layer's steps before its mixer (HC read, PLE).
    Pre,
    /// The fused native input projection (GPU-only pass).
    InProj,
    /// One calibration family group alone (GPU-only pass): 0 = qkv | q, 1 = zab | kvidx.
    Family(usize),
    /// The mixer core.
    Core,
    /// Output projection, HC write and MoE.
    Post,
    /// Rotation plus NPU A-pack (`on` pass).
    Pack,
}
/// (bank, layer kind: 0 GDN / 1 QSA, phase).
pub(super) type SpanKey = (usize, usize, Phase);

/// Instrumentation of one measured pair. Spans are deferred HIP events, resolved once after the pair.
pub(super) struct Probe {
    pub(super) gpu_only: bool,
    spans: RefCell<Vec<(SpanKey, PendingTimer)>>,
    pub(super) records: Vec<TraceRecord>,
}
impl NpuForward {
    pub(super) fn span_begin(
        &self,
        gpu: &Gpu,
    ) -> Result<Option<PendingTimer>, Qwen4GpuForwardError> {
        if self.probe.is_none() {
            return Ok(None);
        }
        Ok(Some(begin_deferred(&gpu.hip, gpu.active_stream.as_ref())?))
    }
    pub(super) fn span_end(
        &self,
        gpu: &Gpu,
        key: SpanKey,
        timer: Option<PendingTimer>,
    ) -> Result<(), Qwen4GpuForwardError> {
        if let (Some(probe), Some(timer)) = (&self.probe, timer) {
            timer.mark_stop(&gpu.hip, gpu.active_stream.as_ref())?;
            probe.spans.borrow_mut().push((key, timer));
        }
        Ok(())
    }
    /// Run `f`, timing it as `key` when a probe is installed.
    pub(super) fn timed<T>(
        &self,
        gpu: &mut Gpu,
        key: SpanKey,
        f: impl FnOnce(&mut Gpu) -> Result<T, Qwen4GpuForwardError>,
    ) -> Result<T, Qwen4GpuForwardError> {
        let span = self.span_begin(gpu)?;
        let result = f(gpu)?;
        self.span_end(gpu, key, span)?;
        Ok(result)
    }
    /// A GPU-only measured layer from its mixer on: the fused native input projection, then each family group alone
    /// through the path an NPU pair uses for kept families (rewriting identical values), the Core, and the rest.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn measured_native(
        &self,
        gpu: &mut Gpu,
        ctx: &DispatchCtx,
        program: &mut layers::BankProgram<'_>,
        hc: &mut HcFusionCursor,
        layer: usize,
        bank: usize,
        window: std::ops::Range<usize>,
    ) -> Result<(), Qwen4GpuForwardError> {
        let index = window.start;
        let kind = self.layers[layer].shape.index();
        phase(&mut program.steps[index], MixerPhase::InProj);
        self.timed(gpu, (bank, kind, Phase::InProj), |gpu| {
            execute_validated_steps_window(gpu, ctx, &program.steps, index..index + 1, hc)
                .map_err(fail)
        })?;
        for group in 0..2 {
            self.timed(gpu, (bank, kind, Phase::Family(group)), |gpu| {
                match &program.steps[index] {
                    // Covered masks in [qkv, a, b, z] and [idx, q, k, v] order: only the group runs.
                    Step::GatedDeltaNet(op) => execute_gated_delta_net_uncovered(
                        gpu,
                        op,
                        [group == 0, group == 1, group == 1, group == 1].map(|run| !run),
                    ),
                    Step::IndexedAttention(op) => execute_indexed_attention_uncovered(
                        gpu,
                        op,
                        [group == 1, group == 0, group == 1, group == 1].map(|run| !run),
                    ),
                    _ => unreachable!("typed mixer"),
                }
                .map_err(fail)
            })?;
        }
        phase(&mut program.steps[index], MixerPhase::Core);
        self.timed(gpu, (bank, kind, Phase::Core), |gpu| {
            self.core_launch(
                gpu,
                layer,
                bank,
                &program.steps[index],
                false,
                CoreWindow {
                    lo: 0,
                    rows: program.rows,
                },
            )
        })?;
        phase(&mut program.steps[index], MixerPhase::OutProj);
        self.timed(gpu, (bank, kind, Phase::Post), |gpu| {
            execute_validated_steps_window(gpu, ctx, &program.steps, window, hc).map_err(fail)
        })?;
        phase(&mut program.steps[index], MixerPhase::Whole);
        Ok(())
    }
}

const ROWS: usize = trunk::CHUNK;
const REPS: usize = 3;
/// Bank B rows of the second `on` pair: a second work size for the NPU chain fit.
const SMALL_BANK: usize = 2048;
/// Candidate `min_bank_rows`, smallest first.
const LADDER: [usize; 4] = [512, 1024, 2048, 4096];
/// A bank size is admitted once its GPU cost per row is within this factor of a full chunk's:
/// the planner prices GPU work linearly in rows, so smaller banks would be priced too cheap.
const LINEAR_TOLERANCE: f64 = 1.10;

/// Summed span milliseconds (and span count) per key of one pair, plus its relay records.
struct Pass {
    spans: BTreeMap<SpanKey, (f64, usize)>,
    records: Vec<TraceRecord>,
    wall_ms: f64,
}
impl Pass {
    fn ms(&self, bank: usize, kind: usize, phase: Phase) -> f64 {
        self.spans.get(&(bank, kind, phase)).map_or(0.0, |s| s.0)
    }
    fn count(&self, bank: usize, kind: usize, phase: Phase) -> usize {
        self.spans.get(&(bank, kind, phase)).map_or(0, |s| s.1)
    }
    /// GPU layer work of one bank without the per-group re-runs: pre + fused in-projection + core + post.
    fn layers(&self, bank: usize) -> f64 {
        (0..2)
            .map(|kind| {
                [Phase::Pre, Phase::InProj, Phase::Core, Phase::Post]
                    .iter()
                    .map(|&p| self.ms(bank, kind, p))
                    .sum::<f64>()
            })
            .sum()
    }
    /// Pre + core + post of both banks: what an NPU pair keeps on the GPU apart from pack.
    fn kept(&self) -> f64 {
        (0..2)
            .flat_map(|bank| (0..2).map(move |kind| (bank, kind)))
            .map(|(bank, kind)| {
                [Phase::Pre, Phase::Core, Phase::Post]
                    .iter()
                    .map(|&p| self.ms(bank, kind, p))
                    .sum::<f64>()
            })
            .sum()
    }
}
fn mean(passes: &[Pass], f: impl Fn(&Pass) -> f64) -> f64 {
    passes.iter().map(f).sum::<f64>() / passes.len() as f64
}

/// Raw on-host measurements in planner units (milliseconds; per 8192 rows unless stated).
#[derive(Clone, Debug, serde::Serialize)]
pub(super) struct Measured {
    /// Layers per kind (GDN, QSA).
    layers: [usize; 2],
    /// Fused native in-projection summed over the layers of each kind.
    inproj: [f64; 2],
    /// Each family group alone summed over the layers: [[qkv, zab], [q, kvidx]].
    groups: [[f64; 2]; 2],
    /// Mean mixer core per layer at prefix 0 (GDN, QSA).
    core: [f64; 2],
    /// Mean QSA core per layer at prefix 8192.
    qsa_core_prefix: f64,
    /// Mean pre + post per layer.
    uniform: f64,
    /// Embedding and PLE staging of a chunk.
    fixed: f64,
    /// Rotation plus A-pack per layer.
    pack: f64,
    /// NPU chains: (layer kind, live rows, relay submit-to-done ms).
    chains: Vec<(usize, usize, f64)>,
    /// GPU pre + core + post of an 8192 + 8192 `on` pair minus the same of the GPU-only pair.
    contention_ms: f64,
    /// Declared NPU DRAM bytes (A + B panels + C) per kilo-op of the full-width chains.
    bytes_per_kop: f64,
    /// GPU layer milliseconds per row of a bank at prefix 0, by bank rows (ascending, ending at 8192).
    per_row: Vec<(usize, f64)>,
    /// GPU layer work of 4096 + 4096 rows minus that of one 8192-row chunk.
    split_tax: f64,
}

/// Least-squares `chain = sync + work / tops` over (work in ms at 1 TOPS, chain ms). A negative intercept is noise
/// around a zero fixed cost: refit through the origin.
fn fit_chain(points: &[(f64, f64)]) -> Result<(f64, f64), String> {
    let n = points.len() as f64;
    let (mx, my) = (
        points.iter().map(|p| p.0).sum::<f64>() / n,
        points.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let sxx: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    if points.len() < 2 || sxx <= 0.0 {
        return Err("NPU chain fit needs at least two distinct work sizes".into());
    }
    let slope = sxy / sxx;
    if !(slope > 0.0) {
        return Err(format!(
            "NPU chain time does not grow with work (slope {slope})"
        ));
    }
    let sync = my - slope * mx;
    if sync >= 0.0 {
        return Ok((sync, 1.0 / slope));
    }
    let through_origin = points.iter().map(|p| p.0 * p.1).sum::<f64>()
        / points.iter().map(|p| p.0 * p.0).sum::<f64>();
    Ok((0.0, 1.0 / through_origin))
}

/// Assemble the record the planner reads from on-host measurements.
pub(super) fn calibration(m: &Measured, key: trunk::CalKey) -> Result<trunk::Calibration, String> {
    use trunk::{
        Calibration, ContentionCalibration, FamilyCalibration, GlueCalibration, GpuCalibration,
        NpuCalibration,
    };
    if m.layers != [36, 12] {
        return Err(format!(
            "the planner models 36 GDN + 12 QSA layers, this model has {:?}",
            m.layers
        ));
    }
    let cols = |shape: Shape, families: &[Family]| -> usize {
        shape
            .column_map(FamilyMask::ALL)
            .parts
            .iter()
            .filter(|(f, _, _)| families.contains(f))
            .map(|&(_, _, w)| w as usize)
            .sum()
    };
    let names = [["qkv", "zab"], ["q", "kvidx"]];
    let members: [[&[Family]; 2]; 2] = [
        [
            &[Family::GdnQkv],
            &[Family::GdnZ, Family::GdnA, Family::GdnB],
        ],
        [
            &[Family::QsaQ],
            &[Family::QsaK, Family::QsaV, Family::QsaIdx],
        ],
    ];
    let mut trunk_in = BTreeMap::new();
    let mut families = BTreeMap::new();
    for (kind, shape) in [Shape::GdnIn, Shape::QsaIn].into_iter().enumerate() {
        // Each group alone repeats the input rotation: keep its share of the fused projection.
        let alone = m.groups[kind][0] + m.groups[kind][1];
        if !(alone > 0.0) {
            return Err(format!(
                "{} input projections measured no GPU time",
                ["GDN", "QSA"][kind]
            ));
        }
        for group in 0..2 {
            let name = names[kind][group];
            trunk_in.insert(
                name.to_owned(),
                m.inproj[kind] * m.groups[kind][group] / alone,
            );
            families.insert(
                name.to_owned(),
                FamilyCalibration {
                    layer: ["gdn", "qsa"][kind].into(),
                    cols: cols(shape, members[kind][group]),
                    out_dtype: if name == "qkv" { "bf16" } else { "f32" }.into(),
                    selectable: true,
                    out: None,
                },
            );
        }
    }
    let full = m
        .per_row
        .last()
        .filter(|r| r.0 == ROWS)
        .ok_or("per-row ladder must end at a full chunk")?
        .1;
    let min_bank_rows = m
        .per_row
        .iter()
        .find(|r| r.1 <= LINEAR_TOLERANCE * full)
        .map_or(ROWS, |r| r.0);
    let points: Vec<_> = m
        .chains
        .iter()
        .map(|&(kind, rows, ms)| {
            let padded = [Shape::GdnIn, Shape::QsaIn][kind]
                .column_map(FamilyMask::ALL)
                .padded_cols();
            (
                2.0 * rows as f64 * padded as f64 * trunk::TRUNK_K as f64 / 1e9,
                ms,
            )
        })
        .collect();
    let (sync_ms, tops_corun) = fit_chain(&points)?;
    let mut cal = Calibration {
        schema_version: trunk::CALIBRATION_SCHEMA_VERSION,
        key,
        gpu: GpuCalibration {
            trunk_in_ms_per_8192: trunk_in,
            uniform_per_layer_ms: m.uniform,
            gdn_core_per_layer_ms: m.core[0],
            qsa_per_layer_ms: m.core[1],
            qsa_growth_per_layer_ms_per_8192_prefix: (m.qsa_core_prefix - m.core[1]).max(0.0),
            fixed_per_chunk_ms: m.fixed,
            boundary_tax_ms: m.split_tax.max(0.0),
            min_bank_rows,
        },
        families,
        // The 0.4.2 route has no consume pass: Core reads the NPU arenas directly.
        glue: GlueCalibration {
            pack_ms_8192: m.pack,
            visit_ms_8192: 0.0,
            sync_ms,
            consume_mode: "direct".into(),
            consume_ms_per_col_8192: [("bf16".to_owned(), 0.0), ("f32".to_owned(), 0.0)].into(),
        },
        npu: NpuCalibration {
            tops_corun,
            tops_alone: None,
            bytes_per_kop: m.bytes_per_kop,
            epilogue: "bf16".into(),
            panel_cols: trunk::PANEL_COLS,
        },
        contention: ContentionCalibration {
            alpha_overlap: 1.0,
            ms_per_gb: 0.0,
            model: String::new(),
        },
        margin: 0.0,
        qsa_order: None,
    };
    // Invert the planner's own burst term: extra = (alpha - 1) * sum of min(G_other, N_this) over the pair's layers.
    let (a, b) = (
        trunk::bank_segments(&cal, FamilyMask::ALL, ROWS, 0),
        trunk::bank_segments(&cal, FamilyMask::ALL, ROWS, ROWS),
    );
    let overlap: f64 = (0..2)
        .map(|k| m.layers[k] as f64 * (a.gpu[k].min(b.npu[k]) + b.gpu[k].min(a.npu[k])))
        .sum();
    let alpha = if overlap > 0.0 {
        1.0 + (m.contention_ms / overlap).max(0.0)
    } else {
        1.0
    };
    cal.contention = ContentionCalibration {
        alpha_overlap: alpha, ms_per_gb: 0.0,
        model: format!("measured on this host by hipfire calibrate-npu-trunk --measure: GPU work of an 8192+8192 on pair vs the GPU-only pair, +{:.1} ms over {overlap:.1} ms of planned overlap (x{alpha:.3})", m.contention_ms),
    };
    cal.validate()?;
    Ok(cal)
}

/// The record and its raw measurements, collected on this host.
pub(crate) struct Collected {
    pub(crate) calibration: trunk::Calibration,
    pub(crate) report: serde_json::Value,
}

fn pass(
    owner: &mut Qwen4GpuForward,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
    banks: [usize; 2],
    gpu_only: bool,
) -> Result<Pass, Qwen4GpuForwardError> {
    bundle.reset(gpu).map_err(fail)?;
    owner.begin_npu_trunk_request();
    let tokens: Vec<u32> = (0..banks[0] + banks[1])
        .map(|i| 10 + (i % 1000) as u32)
        .collect();
    let npu = owner.npu_trunk.as_mut().expect("measured load");
    npu.probe = Some(Probe {
        gpu_only,
        spans: RefCell::new(Vec::new()),
        records: Vec::new(),
    });
    gpu.hip.device_synchronize()?;
    let started = Instant::now();
    let result = run_pair_request(
        owner,
        bundle,
        gpu,
        &tokens,
        0,
        banks,
        None,
        None,
        None,
        Qwen4OutputPolicy::None,
    );
    let synced = gpu.hip.device_synchronize();
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let probe = owner
        .npu_trunk
        .as_mut()
        .expect("measured load")
        .probe
        .take()
        .expect("installed probe");
    let resolved = resolve_deferred(
        &gpu.hip,
        gpu.active_stream.as_ref(),
        probe.spans.into_inner(),
    );
    result?;
    synced?;
    let mut spans = BTreeMap::<SpanKey, (f64, usize)>::new();
    for (key, us) in resolved {
        let entry = spans.entry(key).or_default();
        entry.0 += us / 1e3;
        entry.1 += 1;
    }
    Ok(Pass {
        spans,
        records: probe.records,
        wall_ms,
    })
}

/// Measure every mandatory schema-4 field on this host. Requires an `on` load with every family and an open runtime.
pub(crate) fn collect(
    owner: &mut Qwen4GpuForward,
    bundle: &mut Qwen4Bundle,
    gpu: &mut Gpu,
) -> Result<Collected, Qwen4GpuForwardError> {
    let npu = owner.npu_trunk.as_ref().ok_or_else(|| {
        fail("the measurement needs an npu_trunk=on load (the trunk is not attached)")
    })?;
    if npu.mode != NpuTrunkMode::On || npu.runtime.is_none() || npu.bank_b.is_none() {
        return Err(fail(format!(
            "the measurement needs npu_trunk=on with the NPU runtime open (mode={})",
            npu.mode
        )));
    }
    if npu.families != FamilyMask::ALL {
        return Err(fail(
            "the measurement needs every family on the NPU; move the matched calibration aside",
        ));
    }
    if bundle.state.max_seq_len < 2 * ROWS || bundle.session_next_boundary().is_some() {
        return Err(fail(format!(
            "the measurement needs max_seq >= {} and no pending session snapshot",
            2 * ROWS
        )));
    }
    // Exactly the key serving derives in `match_calibration`.
    let index = bundle
        .npu_model_index
        .as_ref()
        .ok_or_else(|| fail("model index identity unavailable"))?
        .as_ref()
        .map_err(fail)?;
    let pci = gpu.hip.device_pci_bus_id(gpu.device_id)?;
    let key =
        trunk::runtime_calibration_key(&gpu.arch, &pci, index, &npu.trunk_route).map_err(fail)?;
    let kinds: Vec<usize> = npu.layers.iter().map(|l| l.shape.index()).collect();
    let panel_bytes = npu.npu_panels_bytes as f64;
    let prior_scope = std::mem::replace(&mut owner.prefill_scope, true);
    let result = (|| {
        eprintln!("npu-trunk calibrate: warmup");
        pass(owner, bundle, gpu, [ROWS, ROWS], true)?;
        pass(owner, bundle, gpu, [ROWS, ROWS], false)?;
        let (mut alone, mut paired, mut small) = (Vec::new(), Vec::new(), Vec::new());
        for rep in 0..REPS {
            eprintln!("npu-trunk calibrate: pass {}/{REPS} (GPU-only pair, on pair, on pair 8192+{SMALL_BANK})", rep + 1);
            alone.push(pass(owner, bundle, gpu, [ROWS, ROWS], true)?);
            paired.push(pass(owner, bundle, gpu, [ROWS, ROWS], false)?);
            small.push(pass(owner, bundle, gpu, [ROWS, SMALL_BANK], false)?);
        }
        let mut ladder = Vec::new();
        for rows in LADDER {
            eprintln!("npu-trunk calibrate: GPU-only {rows}+{rows} pair");
            ladder.push((rows, pass(owner, bundle, gpu, [rows, rows], true)?));
        }
        Ok::<_, Qwen4GpuForwardError>((alone, paired, small, ladder))
    })();
    owner.prefill_scope = prior_scope;
    let reset = bundle.reset(gpu).map_err(fail);
    let (alone, paired, small, ladder) = result?;
    reset?;
    let a = &alone[0];
    let layers = [a.count(0, 0, Phase::Core), a.count(0, 1, Phase::Core)];
    let per_layer = |kind: usize, phase: Phase, bank: usize| {
        mean(&alone, |p| p.ms(bank, kind, phase)) / layers[kind] as f64
    };
    let mut chains = Vec::new();
    for p in paired.iter().chain(&small) {
        for r in p
            .records
            .iter()
            .filter(|r| r.seq != 0 && r.npu_done_ns > r.npu_submit_ns)
        {
            chains.push((
                kinds[r.layer],
                r.rows,
                (r.npu_done_ns - r.npu_submit_ns) as f64 / 1e6,
            ));
        }
    }
    let full_width = [Shape::GdnIn, Shape::QsaIn].map(|s| s.column_map(FamilyMask::ALL));
    let (bytes, kops) = (0..2).fold((panel_bytes, 0.0), |(bytes, kops), kind| {
        let n = layers[kind] as f64;
        let cols = full_width[kind].padded_cols() as f64;
        (
            bytes
                + n * (layout(&full_width[kind], ROWS).a_bytes() as f64 + ROWS as f64 * cols * 2.0),
            kops + n * 2.0 * ROWS as f64 * cols * trunk::TRUNK_K as f64 / 1e3,
        )
    });
    let mut per_row: Vec<_> = ladder
        .iter()
        .map(|(rows, p)| (*rows, p.layers(0) / *rows as f64))
        .collect();
    per_row.push((ROWS, mean(&alone, |p| p.layers(0)) / ROWS as f64));
    let split = &ladder
        .iter()
        .find(|(rows, _)| *rows == ROWS / 2)
        .expect("ladder holds half a chunk")
        .1;
    let measured = Measured {
        layers,
        inproj: [0, 1].map(|kind| mean(&alone, |p| p.ms(0, kind, Phase::InProj))),
        groups: [0, 1]
            .map(|kind| [0, 1].map(|g| mean(&alone, |p| p.ms(0, kind, Phase::Family(g))))),
        core: [per_layer(0, Phase::Core, 0), per_layer(1, Phase::Core, 0)],
        qsa_core_prefix: per_layer(1, Phase::Core, 1),
        uniform: (0..2)
            .map(|k| mean(&alone, |p| p.ms(0, k, Phase::Pre) + p.ms(0, k, Phase::Post)))
            .sum::<f64>()
            / (layers[0] + layers[1]) as f64,
        fixed: mean(&alone, |p| p.ms(0, 0, Phase::Init)),
        pack: mean(&paired, |p| {
            (0..2)
                .flat_map(|b| (0..2).map(move |k| (b, k)))
                .map(|(b, k)| p.ms(b, k, Phase::Pack))
                .sum::<f64>()
        }) / (2 * (layers[0] + layers[1])) as f64,
        chains,
        contention_ms: mean(&paired, Pass::kept) - mean(&alone, Pass::kept),
        bytes_per_kop: bytes / kops,
        per_row,
        split_tax: split.layers(0) + split.layers(1) - mean(&alone, |p| p.layers(0)),
    };
    let calibration = calibration(&measured, key).map_err(fail)?;
    let predicted = trunk::pair_time(&calibration, FamilyMask::ALL, ROWS, 0, ROWS, ROWS);
    let report = serde_json::json!({
        "measured": measured,
        "pair_8192_8192": {
            "measured_wall_ms": paired.iter().map(|p| p.wall_ms).collect::<Vec<_>>(),
            "predicted_ms": predicted.t,
            "predicted_gpu_only_ms": predicted.gpu_only,
        },
    });
    Ok(Collected {
        calibration,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_fit_recovers_rate_and_sync_and_clamps_negative_sync() {
        let chain = |x: f64| 4.5 + x / 12.5;
        let points: Vec<_> = [100.0, 400.0, 700.0].map(|x| (x, chain(x))).into();
        let (sync, tops) = fit_chain(&points).unwrap();
        assert!((sync - 4.5).abs() < 1e-9 && (tops - 12.5).abs() < 1e-9);
        let (sync, tops) = fit_chain(&[(100.0, 7.0), (200.0, 16.2)]).unwrap();
        assert_eq!(sync, 0.0);
        assert!(
            (tops - (100.0 * 100.0 + 200.0 * 200.0) / (100.0 * 7.0 + 200.0 * 16.2)).abs() < 1e-9
        );
        assert!(fit_chain(&[(100.0, 7.0), (100.0, 8.0)]).is_err());
        assert!(fit_chain(&[(100.0, 9.0), (200.0, 8.0)]).is_err());
    }

    /// Measurements synthesized from the reference overlay record reproduce its planner inputs.
    #[test]
    fn assembled_record_reproduces_the_reference_planner_inputs() {
        let reference: trunk::Calibration =
            serde_json::from_str(include_str!("calibrations/halo-xdna2-overlay-v4.json")).unwrap();
        let g = &reference.gpu;
        let t = |name: &str| g.trunk_in_ms_per_8192[name];
        let gdn_cols = Shape::GdnIn.column_map(FamilyMask::ALL).padded_cols() as f64;
        let qsa_cols = Shape::QsaIn.column_map(FamilyMask::ALL).padded_cols() as f64;
        let work = |rows: usize, cols: f64| 2.0 * rows as f64 * cols * 2560.0 / 1e9;
        let chains: Vec<_> = [
            (0, 8192, gdn_cols),
            (1, 8192, qsa_cols),
            (0, 2048, gdn_cols),
            (1, 2048, qsa_cols),
        ]
        .map(|(kind, rows, cols)| {
            (
                kind,
                rows,
                reference.glue.sync_ms + work(rows, cols) / reference.npu.tops_corun,
            )
        })
        .into();
        let mut m = Measured {
            layers: [36, 12],
            // Each group alone carries a rotation the fused projection shares.
            inproj: [t("qkv") + t("zab"), t("q") + t("kvidx")],
            groups: [
                [t("qkv") * 1.1, t("zab") * 1.1],
                [t("q") * 1.3, t("kvidx") * 1.3],
            ],
            core: [g.gdn_core_per_layer_ms, g.qsa_per_layer_ms],
            qsa_core_prefix: g.qsa_per_layer_ms + g.qsa_growth_per_layer_ms_per_8192_prefix,
            uniform: g.uniform_per_layer_ms,
            fixed: g.fixed_per_chunk_ms,
            pack: reference.glue.pack_ms_8192,
            chains,
            contention_ms: 0.0,
            bytes_per_kop: reference.npu.bytes_per_kop,
            per_row: vec![
                (512, 1.5),
                (1024, 1.2),
                (2048, 1.09),
                (4096, 1.0),
                (8192, 1.0),
            ],
            split_tax: g.boundary_tax_ms,
        };
        let key = reference.key.clone();
        let base = calibration(&m, key.clone()).unwrap();
        let (a, b) = (
            trunk::bank_segments(&base, FamilyMask::ALL, 8192, 0),
            trunk::bank_segments(&base, FamilyMask::ALL, 8192, 8192),
        );
        let overlap: f64 = (0..2)
            .map(|k| [36.0, 12.0][k] * (a.gpu[k].min(b.npu[k]) + b.gpu[k].min(a.npu[k])))
            .sum();
        m.contention_ms = (reference.contention.alpha_overlap - 1.0) * overlap;
        let cal = calibration(&m, key).unwrap();
        let close = |x: f64, y: f64| (x - y).abs() < 1e-9 * y.abs().max(1.0);
        for name in ["qkv", "zab", "q", "kvidx"] {
            assert!(close(cal.gpu.trunk_in_ms_per_8192[name], t(name)), "{name}");
        }
        assert!(close(
            cal.gpu.qsa_growth_per_layer_ms_per_8192_prefix,
            g.qsa_growth_per_layer_ms_per_8192_prefix
        ));
        assert!(
            close(cal.npu.tops_corun, reference.npu.tops_corun)
                && close(cal.glue.sync_ms, reference.glue.sync_ms)
        );
        assert!(close(
            cal.contention.alpha_overlap,
            reference.contention.alpha_overlap
        ));
        assert_eq!(
            (cal.gpu.min_bank_rows, cal.gpu.boundary_tax_ms),
            (2048, g.boundary_tax_ms)
        );
        assert_eq!(cal.npu.panel_cols, reference.npu.panel_cols);
        // Same planner inputs (current packing widths pad to the same panels): same selection and plans.
        assert_eq!(
            trunk::select_families(&cal).0,
            trunk::select_families(&reference).0
        );
        for rows in [9319, 16384, 57544, 65536] {
            let (p, r) = (
                trunk::plan(&cal, FamilyMask::ALL, rows),
                trunk::plan(&reference, FamilyMask::ALL, rows),
            );
            assert_eq!(p.steps, r.steps, "{rows}");
            assert!(
                close(p.t_pipe_ms, r.t_pipe_ms) && close(p.t_gpu_ms, r.t_gpu_ms),
                "{rows}"
            );
        }
    }
}
