// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

mod hal;
#[cfg(test)]
mod tests;
mod witgen;

use anyhow::Result;

use crate::execute::segment::Segment;

pub type Seal = Vec<u32>;

pub trait SegmentProver {
    fn prove_segment(&self, segment: &Segment) -> Result<Seal>;
}

pub fn segment_prover() -> Result<Box<dyn SegmentProver>> {
    self::hal::cpu::segment_prover()
}

const REG_COUNT_DATA: usize = 192;
const REG_COUNT_GLOBAL: usize = 73;
