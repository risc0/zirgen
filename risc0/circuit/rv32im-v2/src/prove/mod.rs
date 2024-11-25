// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

mod hal;
#[cfg(test)]
mod tests;
mod witgen;

use anyhow::Result;
use cfg_if::cfg_if;
use risc0_zkp::core::{digest::Digest, hash::poseidon2::Poseidon2HashSuite};

use crate::{execute::segment::Segment, zirgen::CircuitImpl};

const GLOBAL_MIX: usize = 0;
const GLOBAL_OUT: usize = 1;

pub type Seal = Vec<u32>;

pub trait SegmentProver {
    fn prove(&self, segment: &Segment) -> Result<Seal>;

    fn verify(&self, seal: &Seal) -> Result<()> {
        let hash_suite = Poseidon2HashSuite::new_suite();

        // We don't have a `code' buffer to verify.
        let check_code_fn = |_: u32, _: &Digest| Ok(());

        Ok(risc0_zkp::verify::verify(
            &CircuitImpl,
            &hash_suite,
            seal,
            check_code_fn,
        )?)
    }
}

pub fn segment_prover() -> Result<Box<dyn SegmentProver>> {
    // cfg_if! {
    // if #[cfg(feature = "cuda")] {
    // self::hal::cuda::segment_prover(hashfn)
    // } else if #[cfg(any(all(target_os = "macos", target_arch = "aarch64"), target_os = "ios"))] {
    // self::hal::metal::segment_prover(hashfn)
    // } else {
    self::hal::cpu::segment_prover()
    // }
    // }
}
