// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! raon_oracle `mhc` subcommand.
//!
//! STUB owned by slice O so the dispatcher compiles against the frozen
//! `run(fixtures, opts)` signature; the module owner REPLACES this body.
//! It refuses (Err) rather than reporting a pass.

pub fn run(
    _fixtures: &std::path::Path,
    _opts: &std::collections::BTreeMap<String, String>,
) -> Result<serde_json::Value, String> {
    Err("mhc oracle not built in this branch".to_string())
}

pub fn run_norm(
    _fixtures: &std::path::Path,
    _opts: &std::collections::BTreeMap<String, String>,
) -> Result<serde_json::Value, String> {
    Err("norm oracle not built in this branch".to_string())
}

pub fn run_head(
    _fixtures: &std::path::Path,
    _opts: &std::collections::BTreeMap<String, String>,
) -> Result<serde_json::Value, String> {
    Err("head oracle not built in this branch".to_string())
}
