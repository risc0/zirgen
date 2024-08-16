// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

mod cpp;
mod ffi;
mod info;
mod poly_ext;
mod taps;

use risc0_zkp::{adapter::TapsProvider, taps::TapSet};

pub struct CircuitImpl;

impl CircuitImpl {
    pub const fn new() -> Self {
        CircuitImpl
    }
}

impl TapsProvider for CircuitImpl {
    fn get_taps(&self) -> &'static TapSet<'static> {
        taps::TAPSET
    }
}
