// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Kaden Schutt
//! Raon oracle CLI (PLAN §6.2): module | state | logits | bench. Owner: slice O.
//! Each subcommand dispatches into examples/raon_oracle/<module>.rs, which
//! drive production launchers. Unbuilt modules refuse; never placeholder pass.
mod raon_oracle {
    pub mod mhc;
    pub mod rda;
    pub mod mla;
    pub mod moe;
    pub mod state;
    pub mod logits;
    pub mod bench;
}

fn main() {
    eprintln!("raon_oracle: no subcommand implemented yet");
    std::process::exit(2);
}
