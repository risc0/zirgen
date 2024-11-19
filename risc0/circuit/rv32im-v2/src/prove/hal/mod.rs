// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

pub mod cpu;

use std::rc::Rc;

use anyhow::Result;
use risc0_core::scope;
use risc0_zkp::{
    field::baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
    hal::{Buffer, CircuitHal, Hal},
};

use super::{
    witgen::{preflight::PreflightTrace, WitnessGenerator},
    Seal, SegmentProver,
};
use crate::execute::segment::Segment;

pub(crate) struct WitnessBuffer<H: Hal> {
    pub buf: H::Buffer<H::Elem>,
    pub rows: usize,
    pub cols: usize,
    pub checked_reads: bool,
}

impl<H> WitnessBuffer<H>
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
{
    pub fn to_vec(&self) -> Vec<BabyBearElem> {
        self.buf.to_vec()
    }
}

#[derive(PartialEq)]
pub(crate) enum StepMode {
    Parallel,
    SeqForward,
    #[allow(dead_code)]
    SeqReverse,
}

pub(crate) trait CircuitWitnessGenerator<H: Hal> {
    #[allow(clippy::too_many_arguments)]
    fn generate_witness(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        steps: usize,
        global: &WitnessBuffer<H>,
        data: &WitnessBuffer<H>,
    ) -> Result<()>;
}

pub(crate) struct SegmentProverImpl<H, C>
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
    C: CircuitHal<H> + CircuitWitnessGenerator<H>,
{
    hal: Rc<H>,
    circuit_hal: Rc<C>,
}

impl<H, C> SegmentProverImpl<H, C>
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
    C: CircuitHal<H> + CircuitWitnessGenerator<H>,
{
    pub fn new(hal: Rc<H>, circuit_hal: Rc<C>) -> Self {
        Self { hal, circuit_hal }
    }
}

impl<H, C> SegmentProver for SegmentProverImpl<H, C>
where
    H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
    C: CircuitHal<H> + CircuitWitnessGenerator<H>,
{
    fn prove_segment(&self, segment: &Segment) -> Result<Seal> {
        scope!("prove_segment");

        let witgen = WitnessGenerator::new(
            self.hal.as_ref(),
            self.circuit_hal.as_ref(),
            segment,
            StepMode::Parallel,
        )?;
        // let steps = witgen.steps;

        // Ok(scope!("prove", {
        //     let mut prover = Prover::new(self.hal.as_ref(), CIRCUIT.get_taps());
        //     let hashfn = &self.hal.get_hash_suite().hashfn;

        //     let mix = scope!("main", {
        //         // At the start of the protocol, seed the Fiat-Shamir transcript with context information
        //         // about the proof system and circuit.
        //         prover
        //             .iop()
        //             .commit(&hashfn.hash_elem_slice(&PROOF_SYSTEM_INFO.encode()));
        //         prover
        //             .iop()
        //             .commit(&hashfn.hash_elem_slice(&CircuitImpl::CIRCUIT_INFO.encode()));

        //         // Concat io (i.e. globals) and po2 into a vector.
        //         let mut io_po2 = vec![BabyBearElem::ZERO; io.len() + 1];
        //         witgen.io.view_mut(|view| {
        //             for (i, elem) in view.iter_mut().enumerate() {
        //                 *elem = elem.valid_or_zero();
        //                 io_po2[i] = *elem;
        //             }
        //             io_po2[io.len()] = BabyBearElem::new_raw(segment.po2 as u32);
        //         });

        //         let io_po2_digest = hashfn.hash_elem_slice(&io_po2);
        //         prover.iop().commit(&io_po2_digest);
        //         prover.iop().write_field_elem_slice(io_po2.as_slice());
        //         prover.set_po2(segment.po2);

        //         prover.commit_group(REGISTER_GROUP_CTRL, &witgen.ctrl);
        //         prover.commit_group(REGISTER_GROUP_DATA, &witgen.data);

        //         // Make the mixing values
        //         let mix: Vec<_> = scope!(
        //             "mix",
        //             (0..CircuitImpl::MIX_SIZE)
        //                 .map(|_| prover.iop().random_elem())
        //                 .collect()
        //         );

        //         let mix = scope!("copy(mix)", self.hal.copy_from_elem("mix", mix.as_slice()));

        //         let accum = scope!(
        //             "alloc(accum)",
        //             self.hal.alloc_elem_init(
        //                 "accum",
        //                 steps * CIRCUIT.accum_size(),
        //                 BabyBearElem::INVALID,
        //             )
        //         );

        //         // Add random noise to end of accum
        //         scope!("noise(accum)", {
        //             let mut rng = thread_rng();
        //             let noise =
        //                 vec![BabyBearElem::random(&mut rng); ZK_CYCLES * CIRCUIT.accum_size()];
        //             self.hal.eltwise_copy_elem_slice(
        //                 &accum,
        //                 &noise,
        //                 CIRCUIT.accum_size(), // from_rows
        //                 ZK_CYCLES,            // from_cols
        //                 0,                    // from_offset
        //                 ZK_CYCLES,            // from_stride
        //                 steps - ZK_CYCLES,    // into_offset
        //                 steps,                // into_stride
        //             );
        //         });

        //         self.circuit_hal.accumulate(
        //             &witgen.ctrl,
        //             &witgen.io,
        //             &witgen.data,
        //             &mix,
        //             &accum,
        //             steps,
        //         );

        //         prover.commit_group(REGISTER_GROUP_ACCUM, &accum);

        //         mix
        //     });

        //     prover.finalize(&[&mix, &witgen.io], self.circuit_hal.as_ref())
        // }))

        todo!()
    }
}
