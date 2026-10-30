// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! raon_oracle `rda` subcommand.
//!
//! STUB owned by slice O so the dispatcher compiles against the frozen
//! `run(fixtures, opts)` signature; the module owner REPLACES this body.
//! It refuses (Err) rather than reporting a pass.

pub fn run(
    _fixtures: &std::path::Path,
    _opts: &std::collections::BTreeMap<String, String>,
) -> Result<serde_json::Value, String> {
    Err("rda oracle not built in this branch".to_string())
}
