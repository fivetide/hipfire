// SPDX-License-Identifier: Apache-2.0
//! NPU trunk calibration: on-host measurement (`--measure`) or offline assembly of an
//! already measured record (`--from`, which never runs a benchmark or opens a device).
use anyhow::{bail, Context, Result};
use clap::{Args, Parser};
use hipfire_arch_qwen4::npu_trunk::{
    plan, request_log_line, select_families, CalKey, Calibration, ContentionCalibration, FamilyCalibration,
    GlueCalibration, GpuCalibration, NpuCalibration, QsaOrder, CALIBRATION_SCHEMA_VERSION,
};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::PathBuf, process::Command};

#[derive(Args, Debug)]
pub(crate) struct CalibrateArgs {
    /// Measure this host: a fresh daemon loads the model with `npu_trunk=on` and times
    /// the trunk (a few minutes of exclusive GPU and NPU use).
    #[arg(long, conflicts_with = "from")]
    measure: bool,
    /// Model to measure with `--measure`.
    #[arg(long, value_name = "MODEL", conflicts_with = "from")]
    model: Option<String>,
    /// Assemble this already measured record instead.
    #[arg(long, value_name = "MEASUREMENTS.JSON", required_unless_present = "measure")]
    from: Option<PathBuf>,
    /// Output record. `--measure` defaults to the path serving reads
    /// (`fusions.npu_trunk_calibration`, default `<hipfire home>/npu-trunk/calibration.json`).
    #[arg(long, value_name = "CAL.JSON", required_unless_present = "measure")]
    out: Option<PathBuf>,
}
/// The one model family the trunk serves.
const DEFAULT_MODEL: &str = "qwen3.8:flash-next";
/// All measured costs are mandatory; only build defaults to the release contract.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Measurements {
    schema_version: u32,
    key: CalKey,
    gpu: GpuCalibration,
    families: BTreeMap<String, FamilyCalibration>,
    glue: GlueCalibration,
    npu: NpuCalibration,
    contention: ContentionCalibration,
    margin: f64,
    #[serde(default)]
    qsa_order: Option<QsaOrder>,
}
impl Measurements {
    fn assemble(self, uname_release: &str) -> Result<Calibration> {
        if self.schema_version != CALIBRATION_SCHEMA_VERSION { bail!("unsupported measurements schema_version"); }
        // Device/fw and index identity come from the measurement collector.
        // CalKey defaults an omitted build to this release's stable NPU contract.
        if self.key.kernel.uname_release != uname_release {
            bail!("measurement key mismatch: kernel.uname_release (measured {}, current {})", self.key.kernel.uname_release, uname_release);
        }
        let cal = Calibration {
            schema_version: CALIBRATION_SCHEMA_VERSION,
            key: self.key, gpu: self.gpu, families: self.families, glue: self.glue,
            npu: self.npu, contention: self.contention, margin: self.margin, qsa_order: self.qsa_order,
        };
        cal.validate().map_err(anyhow::Error::msg)?;
        Ok(cal)
    }
}
pub(crate) fn calibrate_command(paths: &crate::Paths, args: CalibrateArgs) -> Result<()> {
    if args.measure {
        return measure_command(paths, args.model.as_deref().unwrap_or(DEFAULT_MODEL), args.out);
    }
    let (Some(from), Some(out)) = (args.from, args.out) else { bail!("--from and --out are required without --measure") };
    let bytes = fs::read(&from).with_context(|| format!("read {}", from.display()))?;
    let input: Measurements = serde_json::from_slice(&bytes).context("invalid or incomplete measurement record")?;
    let output = Command::new("uname").arg("-r").output().context("read current kernel release")?;
    if !output.status.success() { bail!("uname -r failed"); }
    let release = std::str::from_utf8(&output.stdout).context("uname output is not UTF-8")?.trim();
    let cal = input.assemble(release)?;
    cal.store(&out).map_err(anyhow::Error::msg)?;
    println!("npu-trunk: calibration={} schema_version={}", out.display(), cal.schema_version);
    Ok(())
}

/// Load `model` in a fresh daemon with `npu_trunk=on` (the serving load path, overlay and memlock checks), measure
/// the trunk there and store the keyed record with its raw measurements beside it.
fn measure_command(paths: &crate::Paths, model: &str, out: Option<PathBuf>) -> Result<()> {
    let out = out.unwrap_or_else(hipfire_config::npu_trunk_calibration_path);
    // The bench load path: registry pull, PLE sidecar, configured load params; 32K covers the 16K measurement pairs.
    let Some(crate::Commands::Bench(bench)) = crate::Cli::try_parse_from(["hipfire", "bench", model, "--max-seq", "32768"])?.command
    else { unreachable!("bench arguments parse as bench") };
    eprintln!("npu-trunk: measuring {model} on this host with npu_trunk=on (exclusive GPU and NPU for a few minutes)");
    let (mut engine, ..) = crate::open_bench_engine(paths, &bench, None, &crate::BenchLoadOpts {
        config: &[("fusions.npu_trunk", "on")],
        ..Default::default()
    })?;
    let reply = crate::bench_probe(&mut engine, serde_json::json!({ "type": "calibrate_npu_trunk" }), "npu_trunk_calibration")?;
    let cal: Calibration = serde_json::from_value(reply["calibration"].clone()).context("daemon calibration record")?;
    if let Some(dir) = out.parent() {
        fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    cal.store(&out).map_err(anyhow::Error::msg)?;
    let report = out.with_extension("measurements.json");
    fs::write(&report, serde_json::to_vec_pretty(&reply["report"])?).with_context(|| format!("write {}", report.display()))?;
    println!("npu-trunk: calibration={} schema_version={} measurements={}", out.display(), cal.schema_version, report.display());
    let families = select_families(&cal).0;
    if families.is_empty() {
        println!("npu-trunk: the measured costs select no families; auto stays GPU-only on this host");
    } else {
        println!("{}", request_log_line(16384, &plan(&cal, families, 16384)));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn npu_trunk_cli_requires_both_paths_or_measure() {
        assert!(crate::Cli::try_parse_from(["hipfire", "calibrate-npu-trunk", "--from", "in.json", "--out", "out.json"]).is_ok());
        assert!(crate::Cli::try_parse_from(["hipfire", "calibrate-npu-trunk", "--from", "in.json"]).is_err());
        assert!(crate::Cli::try_parse_from(["hipfire", "calibrate-npu-trunk", "--measure"]).is_ok());
        assert!(crate::Cli::try_parse_from(["hipfire", "calibrate-npu-trunk", "--measure", "--model", "m", "--out", "o.json"]).is_ok());
        assert!(crate::Cli::try_parse_from(["hipfire", "calibrate-npu-trunk", "--measure", "--from", "in.json"]).is_err());
        assert!(crate::Cli::try_parse_from(["hipfire", "calibrate-npu-trunk", "--model", "m", "--from", "in.json", "--out", "o.json"]).is_err());
    }
    #[test]
    fn npu_trunk_measurements_reject_missing_fields() {
        assert!(serde_json::from_str::<Measurements>("{}").is_err());
        assert!(serde_json::from_str::<Measurements>(r#"{"schema_version":4,"margin":0}"#).is_err());
    }
    #[test]
    fn npu_trunk_measurements_accept_schema_v4_stock_and_overlay() {
        for source in [
            include_str!("../../hipfire-arch-qwen4/tests/fixtures/npu_trunk/g80bf16-stock-v2.json"),
            include_str!("../../hipfire-arch-qwen4/tests/fixtures/npu_trunk/g80bf16-overlay-v2.json"),
        ] {
            let input: Measurements = serde_json::from_str(source).unwrap();
            let release = input.key.kernel.uname_release.clone();
            let cal = input.assemble(&release).unwrap();
            assert_eq!(cal.schema_version, 4);
            assert_eq!(cal.families["qkv"].out_dtype, "bf16");
            assert_eq!(cal.families["zab"].out_dtype, "f32");
            assert_eq!(cal.glue.consume_mode, "retile");
            assert_eq!(cal.glue.consume_ms_per_col_8192["bf16"], 2.85 / 16480.0);
            let mut direct: serde_json::Value = serde_json::from_str(source).unwrap();
            direct["glue"]["consume_mode"] = serde_json::json!("direct");
            let input: Measurements = serde_json::from_value(direct).unwrap();
            assert!(input.assemble(&release).is_ok());
            let input: Measurements = serde_json::from_str(source).unwrap();
            assert!(input.assemble("wrong-kernel").is_err());
            let mut defaults: serde_json::Value = serde_json::from_str(source).unwrap();
            defaults["key"].as_object_mut().unwrap().remove("build");
            let input: Measurements = serde_json::from_value(defaults).unwrap();
            assert_eq!(input.assemble(&release).unwrap().key.build,
                hipfire_arch_qwen4::npu_trunk::calibration_build_identity());
            for (parent, field) in [("key", "trunk_route"), ("glue", "consume_mode"), ("glue", "consume_ms_per_col_8192")] {
                let mut incomplete: serde_json::Value = serde_json::from_str(source).unwrap();
                incomplete[parent].as_object_mut().unwrap().remove(field);
                assert!(serde_json::from_value::<Measurements>(incomplete).is_err());
            }
        }
    }
}
