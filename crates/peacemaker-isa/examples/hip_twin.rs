// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
// hipfire — see LICENSE and NOTICE in the project root.

//! `hip_twin --in X.co --arch gfx1100 --kernel SYM --out OUT.hxaco [--abi-ref INC.co --abi-ref-symbol S] [--report R.json]`
//! Seals a hipcc object as an `.hxaco` (+ `.hxaco.sha256`) and prints the report JSON.

use peacemaker_isa::{arch::Arch, hip_twin};

fn run() -> Result<(), String> {
    let (mut inp, mut arch, mut sym, mut out, mut aref, mut asym, mut rep) = (None, None, None, None, None, None, None);
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let slot = match a.as_str() {
            "--in" => &mut inp,
            "--arch" => &mut arch,
            "--kernel" => &mut sym,
            "--out" => &mut out,
            "--abi-ref" => &mut aref,
            "--abi-ref-symbol" => &mut asym,
            "--report" => &mut rep,
            _ => return Err(format!("unknown argument {a}")),
        };
        *slot = Some(it.next().ok_or(format!("{a} needs a value"))?);
    }
    let usage = "usage: hip_twin --in X.co --arch gfx1100 --kernel SYM --out OUT.hxaco [--abi-ref INC.co --abi-ref-symbol S] [--report R.json]";
    let (inp, arch, sym, out) = (
        inp.ok_or(usage)?,
        arch.ok_or(usage)?.parse::<Arch>().map_err(|e| e)?,
        sym.ok_or(usage)?,
        out.ok_or(usage)?,
    );
    let bytes = std::fs::read(&inp).map_err(|e| format!("{inp}: {e}"))?;
    let inc = match (&aref, &asym) {
        (Some(p), Some(s)) => Some((std::fs::read(p).map_err(|e| format!("{p}: {e}"))?, s.clone())),
        (None, None) => None,
        _ => return Err("--abi-ref and --abi-ref-symbol go together".into()),
    };
    let s = hip_twin::seal(&bytes, arch, &sym, inc.as_ref().map(|(b, s)| (b.as_slice(), s.as_str())))?;
    hip_twin::verify_hxaco(&s.hxaco, arch, &sym)?;
    std::fs::write(&out, &s.hxaco).map_err(|e| format!("{out}: {e}"))?;
    std::fs::write(format!("{out}.sha256"), &s.hxaco_sha256).map_err(|e| e.to_string())?;
    let json = serde_json::to_string_pretty(&s.report).unwrap();
    if let Some(r) = rep {
        std::fs::write(&r, &json).map_err(|e| format!("{r}: {e}"))?;
    }
    println!("{json}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("hip_twin: {e}");
        std::process::exit(1);
    }
}
