// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! raon_oracle `generate` subcommand.
//!
//! STUB created by slice O so the dispatcher compiles against the frozen
//! `run(fixtures, opts)` signature; slice I REPLACES this body. It refuses
//! (Err) rather than reporting a pass. Result: `{"text": "..", "pass": bool}`
//! (text is printed to stdout by the dispatcher).

pub fn run(
    _fixtures: &std::path::Path,
    _opts: &std::collections::BTreeMap<String, String>,
) -> Result<serde_json::Value, String> {
    Err("generate oracle not built in this branch".to_string())
}
