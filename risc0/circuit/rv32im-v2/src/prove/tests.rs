// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

use std::rc::Rc;

use rand::thread_rng;
use risc0_binfmt::Program;
use risc0_zkp::{
    core::hash::poseidon2::Poseidon2HashSuite,
    field::{baby_bear::BabyBearExtElem, Elem as _},
    hal::cpu::CpuHal,
};
use test_log::test;

use super::{hal::StepMode, segment_prover};
use crate::{
    execute::{image::MemoryImage2, testutil, DEFAULT_SEGMENT_LIMIT_PO2},
    prove::{witgen::WitnessGenerator, REG_COUNT_DATA},
};

fn fwd_rev_ab_test(program: Program) {
    let image = MemoryImage2::new(program);

    let result = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        testutil::DEFAULT_SESSION_LIMIT,
        &testutil::NullSyscall,
        None,
    )
    .unwrap();

    // cfg_if! {
    //     if #[cfg(feature = "cuda")] {
    //         use risc0_zkp::hal::cuda::CudaHalSha256;
    //         use crate::prove::hal::cuda::CudaCircuitHalSha256;
    //         let hal = Rc::new(CudaHalSha256::new());
    //         let circuit_hal = CudaCircuitHalSha256::new(hal.clone());
    //     } else if #[cfg(any(all(target_os = "macos", target_arch = "aarch64"), target_os = "ios"))] {
    //         use risc0_zkp::hal::metal::MetalHalSha256;
    //         use crate::prove::hal::metal::MetalCircuitHal;
    //         let hal = Rc::new(MetalHalSha256::new());
    //         let circuit_hal = MetalCircuitHal::new(hal.clone());
    //     } else {
    use crate::prove::hal::cpu::CpuCircuitHal;
    let suite = Poseidon2HashSuite::new_suite();
    let hal = Rc::new(CpuHal::new(suite));
    let circuit_hal = CpuCircuitHal::new();
    //     }
    // }

    let mut rng = thread_rng();
    let nonce = BabyBearExtElem::random(&mut rng);

    let segments = result.segments;
    for segment in segments {
        let fwd_witgen = WitnessGenerator::new(
            hal.as_ref(),
            &circuit_hal,
            &segment,
            StepMode::SeqForward,
            nonce,
        )
        .unwrap();
        let rev_witgen = WitnessGenerator::new(
            hal.as_ref(),
            &circuit_hal,
            &segment,
            StepMode::SeqReverse,
            nonce,
        )
        .unwrap();
        let cycles = 1 << segment.po2;
        let fwd_vec = fwd_witgen.data.to_vec();
        let rev_vec = rev_witgen.data.to_vec();
        for row in 0..cycles {
            let fwd_row = &fwd_vec[row * REG_COUNT_DATA..row * REG_COUNT_DATA + REG_COUNT_DATA];
            let rev_row = &rev_vec[row * REG_COUNT_DATA..row * REG_COUNT_DATA + REG_COUNT_DATA];
            assert_eq!(fwd_row, rev_row, "cycle: {row}");
        }
    }
}

#[test]
fn basic() {
    let program = testutil::basic();
    let image = MemoryImage2::new(program);

    let result = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        testutil::DEFAULT_SESSION_LIMIT,
        &testutil::NullSyscall,
        None,
    )
    .unwrap();
    let segments = result.segments;
    let segment = segments.first().unwrap();

    let prover = segment_prover().unwrap();
    // let seal = prover.prove_segment(segment).unwrap();

    // let suite = Sha256HashSuite::new_suite();
    // let hal = CpuHal::new(suite.clone());
    // let checker = ControlCheck::new(&hal, segment.po2);
    // risc0_zkp::verify::verify(&CIRCUIT, &suite, &seal, |x, y| checker.check_ctrl(x, y)).unwrap();
}

#[test]
fn fwd_rev_ab_basic() {
    fwd_rev_ab_test(testutil::basic());
}

#[test]
fn fwd_rev_ab_split() {
    fwd_rev_ab_test(testutil::simple_loop());
}

// #[test]
// fn fwd_rev_ab_large_text() {
//     fwd_rev_ab_test(testutil::large_text());
// }
