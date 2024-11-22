// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

use std::rc::Rc;

use anyhow::Result;
use risc0_circuit_rv32im_v2_sys::{
    risc0_circuit_rv32im_v2_cpu_witgen, RawBuffer, RawExecutionTrace, RawPreflightTrace,
};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    core::hash::poseidon2::Poseidon2HashSuite,
    field::baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
    hal::{cpu::CpuBuffer, AccumPreflight, CircuitHal},
};

use super::{CircuitWitnessGenerator, SegmentProver, SegmentProverImpl, StepMode, WitnessBuffer};
use crate::prove::witgen::preflight::PreflightTrace;

type CpuHal = risc0_zkp::hal::cpu::CpuHal<BabyBear>;

#[derive(Default)]
pub struct CpuCircuitHal;

impl CpuCircuitHal {
    pub fn new() -> Self {
        Self
    }
}

impl CircuitWitnessGenerator<CpuHal> for CpuCircuitHal {
    fn generate_witness(
        &self,
        mode: StepMode,
        preflight: &PreflightTrace,
        cycles: usize,
        global: &WitnessBuffer<CpuHal>,
        data: &WitnessBuffer<CpuHal>,
    ) -> Result<()> {
        scope!("cpu_witgen");
        tracing::debug!("witgen: {cycles}");
        let global_buf = global.buf.as_slice();
        let data_buf = data.buf.as_slice();
        let exec_trace = RawExecutionTrace {
            global: RawBuffer {
                buf: global_buf.as_ptr(),
                rows: global.rows,
                cols: global.cols,
                checked_reads: global.checked_reads,
            },
            data: RawBuffer {
                buf: data_buf.as_ptr(),
                rows: data.rows,
                cols: data.cols,
                checked_reads: data.checked_reads,
            },
        };
        let preflight = RawPreflightTrace {
            cycles: preflight.cycles.as_ptr(),
            txns: preflight.txns.as_ptr(),
            table_split_cycle: preflight.table_split_cycle,
        };
        ffi_wrap(|| unsafe {
            risc0_circuit_rv32im_v2_cpu_witgen(mode as u32, &exec_trace, &preflight, cycles as u32)
        })
    }
}

impl CircuitHal<CpuHal> for CpuCircuitHal {
    fn eval_check(
        &self,
        check: &CpuBuffer<BabyBearElem>,
        groups: &[&CpuBuffer<BabyBearElem>],
        globals: &[&CpuBuffer<BabyBearElem>],
        poly_mix: BabyBearExtElem,
        po2: usize,
        steps: usize,
    ) {
        scope!("eval_check");

        // const EXP_PO2: usize = log2_ceil(INV_RATE);
        // let domain = steps * INV_RATE;

        // let poly_mix_pows = map_pow(poly_mix, crate::info::POLY_MIX_POWERS);

        // // SAFETY: Convert a borrow of a cell into a raw const slice so that we can pass
        // // it over the thread boundary. This should be safe because the scope of the
        // // usage is within this function and each thread access will not overlap with
        // // each other.

        // let ctrl = groups[REGISTER_GROUP_CTRL].as_slice();
        // let ctrl = unsafe { std::slice::from_raw_parts(ctrl.as_ptr(), ctrl.len()) };
        // let data = groups[REGISTER_GROUP_DATA].as_slice();
        // let data = unsafe { std::slice::from_raw_parts(data.as_ptr(), data.len()) };
        // let accum = groups[REGISTER_GROUP_ACCUM].as_slice();
        // let accum = unsafe { std::slice::from_raw_parts(accum.as_ptr(), accum.len()) };
        // let mix = globals[GLOBAL_MIX].as_slice();
        // let mix = unsafe { std::slice::from_raw_parts(mix.as_ptr(), mix.len()) };
        // let out = globals[GLOBAL_OUT].as_slice();
        // let out = unsafe { std::slice::from_raw_parts(out.as_ptr(), out.len()) };
        // let check = check.as_slice();
        // let check = unsafe { std::slice::from_raw_parts(check.as_ptr(), check.len()) };
        // let poly_mix_pows = poly_mix_pows.as_slice();

        // let args: &[&[BabyBearElem]] = &[ctrl, out, data, mix, accum];

        // (0..domain).into_par_iter().for_each(|cycle| {
        //     let tot = CIRCUIT.poly_fp(cycle, domain, poly_mix_pows, args);
        //     let x = BabyBearElem::ROU_FWD[po2 + EXP_PO2].pow(cycle);
        //     // TODO: what is this magic number 3?
        //     let y = (BabyBearElem::new(3) * x).pow(1 << po2);
        //     let ret = tot * (y - BabyBearElem::new(1)).inv();

        //     // SAFETY: This conversion is to make the check slice mutable, which should be
        //     // safe because each thread access will not overlap with each other.
        //     let check = unsafe {
        //         std::slice::from_raw_parts_mut(check.as_ptr() as *mut BabyBearElem, check.len())
        //     };
        //     for i in 0..BabyBearExtElem::EXT_SIZE {
        //         check[i * domain + cycle] = ret.elems()[i];
        //     }
        // });
        todo!()
    }

    fn accumulate(
        &self,
        preflight: &AccumPreflight,
        ctrl: &CpuBuffer<BabyBearElem>,
        global: &CpuBuffer<BabyBearElem>,
        data: &CpuBuffer<BabyBearElem>,
        mix: &CpuBuffer<BabyBearElem>,
        accum: &CpuBuffer<BabyBearElem>,
        steps: usize,
    ) {
        scope!("accumulate");

        // {
        //     let args = &[
        //         ctrl.as_slice_sync(),
        //         io.as_slice_sync(),
        //         data.as_slice_sync(),
        //         mix.as_slice_sync(),
        //         accum.as_slice_sync(),
        //     ];

        //     let accum_ctx = CIRCUIT.alloc_accum_ctx(steps);

        //     scope!("step_compute_accum", {
        //         (0..steps - ZK_CYCLES).into_par_iter().for_each(|cycle| {
        //             CIRCUIT
        //                 .par_step_compute_accum(steps, cycle, &accum_ctx, args)
        //                 .unwrap();
        //         });
        //     });
        //     scope!("calc_prefix_products", {
        //         CIRCUIT.calc_prefix_products(&accum_ctx).unwrap();
        //     });
        //     scope!("step_verify_accum", {
        //         (0..steps - ZK_CYCLES).into_par_iter().for_each(|cycle| {
        //             CIRCUIT
        //                 .par_step_verify_accum(steps, cycle, &accum_ctx, args)
        //                 .unwrap();
        //         });
        //     });
        // }

        // {
        //     // Zero out 'invalid' entries in accum and io
        //     let mut accum_slice = accum.as_slice_mut();
        //     let mut io = io.as_slice_mut();
        //     for value in accum_slice.iter_mut().chain(io.iter_mut()) {
        //         *value = value.valid_or_zero();
        //     }
        // }
        todo!()
    }
}

pub fn segment_prover() -> Result<Box<dyn SegmentProver>> {
    let suite = Poseidon2HashSuite::new_suite();
    let hal = Rc::new(CpuHal::new(suite));
    let circuit_hal = Rc::new(CpuCircuitHal::new());
    Ok(Box::new(SegmentProverImpl::new(hal, circuit_hal)))
}
