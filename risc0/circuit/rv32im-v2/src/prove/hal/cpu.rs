// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

use std::rc::Rc;

use anyhow::Result;
use rayon::prelude::*;
use risc0_circuit_rv32im_v2_sys::{
    risc0_circuit_rv32im_v2_cpu_accum, risc0_circuit_rv32im_v2_cpu_poly_fp,
    risc0_circuit_rv32im_v2_cpu_witgen, RawAccumBuffers, RawBuffer, RawExecBuffers,
    RawPreflightTrace,
};
use risc0_core::scope;
use risc0_sys::ffi_wrap;
use risc0_zkp::{
    core::{hash::poseidon2::Poseidon2HashSuite, log2_ceil},
    field::{
        baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
        map_pow, Elem, ExtElem as _, RootsOfUnity as _,
    },
    hal::{cpu::CpuBuffer, AccumPreflight, CircuitHal},
    INV_RATE,
};

use super::{
    CircuitAccumulator, CircuitWitnessGenerator, MetaBuffer, SegmentProver, SegmentProverImpl,
    StepMode,
};
use crate::{
    prove::{witgen::preflight::PreflightTrace, GLOBAL_MIX, GLOBAL_OUT},
    zirgen::{
        circuit::{REGISTER_GROUP_ACCUM, REGISTER_GROUP_CODE, REGISTER_GROUP_DATA},
        info::POLY_MIX_POWERS,
    },
};

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
        global: &MetaBuffer<CpuHal>,
        data: &MetaBuffer<CpuHal>,
    ) -> Result<()> {
        scope!("cpu_witgen");
        let cycles = preflight.cycles.len();
        tracing::debug!("witgen: {cycles}");
        let global_buf = global.buf.as_slice();
        let data_buf = data.buf.as_slice();
        let buffers = RawExecBuffers {
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
            risc0_circuit_rv32im_v2_cpu_witgen(mode as u32, &buffers, &preflight, cycles as u32)
        })
    }
}

impl CircuitAccumulator<CpuHal> for CpuCircuitHal {
    fn step_accum(
        &self,
        preflight: &PreflightTrace,
        data: &MetaBuffer<CpuHal>,
        accum: &MetaBuffer<CpuHal>,
        mix: &MetaBuffer<CpuHal>,
    ) -> Result<()> {
        scope!("accumulate");
        let cycles = preflight.cycles.len();
        tracing::debug!("accumulate: {cycles}");
        let data_buf = data.buf.as_slice();
        let accum_buf = accum.buf.as_slice();
        let mix_buf = mix.buf.as_slice();
        let buffers = RawAccumBuffers {
            data: RawBuffer {
                buf: data_buf.as_ptr(),
                rows: data.rows,
                cols: data.cols,
                checked_reads: data.checked_reads,
            },
            accum: RawBuffer {
                buf: accum_buf.as_ptr(),
                rows: accum.rows,
                cols: accum.cols,
                checked_reads: accum.checked_reads,
            },
            mix: RawBuffer {
                buf: mix_buf.as_ptr(),
                rows: mix.rows,
                cols: mix.cols,
                checked_reads: mix.checked_reads,
            },
        };
        let preflight = RawPreflightTrace {
            cycles: preflight.cycles.as_ptr(),
            txns: preflight.txns.as_ptr(),
            table_split_cycle: preflight.table_split_cycle,
        };
        ffi_wrap(|| unsafe {
            risc0_circuit_rv32im_v2_cpu_accum(&buffers, &preflight, cycles as u32)
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

        const EXP_PO2: usize = log2_ceil(INV_RATE);
        let domain = steps * INV_RATE;
        let poly_mix_pows = map_pow(poly_mix, POLY_MIX_POWERS);

        // SAFETY: Convert a borrow of a cell into a raw const slice so that we can pass
        // it over the thread boundary. This should be safe because the scope of the
        // usage is within this function and each thread access will not overlap with
        // each other.

        let code = groups[REGISTER_GROUP_CODE].as_slice();
        let data = groups[REGISTER_GROUP_DATA].as_slice();
        let accum = groups[REGISTER_GROUP_ACCUM].as_slice();
        let mix = globals[GLOBAL_MIX].as_slice();
        let out = globals[GLOBAL_OUT].as_slice();
        let check = check.as_slice();

        let code = unsafe { std::slice::from_raw_parts(code.as_ptr(), code.len()) };
        let data = unsafe { std::slice::from_raw_parts(data.as_ptr(), data.len()) };
        let accum = unsafe { std::slice::from_raw_parts(accum.as_ptr(), accum.len()) };
        let mix = unsafe { std::slice::from_raw_parts(mix.as_ptr(), mix.len()) };
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr(), out.len()) };
        let check = unsafe { std::slice::from_raw_parts(check.as_ptr(), check.len()) };
        let poly_mix_pows = poly_mix_pows.as_slice();

        let args: &[&[BabyBearElem]] = &[accum, code, data, out, mix];

        (0..domain).into_par_iter().for_each(|cycle| {
            let args: Vec<*const BabyBearElem> = args.iter().map(|x| (*x).as_ptr()).collect();
            let mut tot = BabyBearExtElem::ZERO;
            unsafe {
                risc0_circuit_rv32im_v2_cpu_poly_fp(
                    cycle,
                    domain,
                    poly_mix_pows.as_ptr(),
                    args.as_ptr(),
                    &mut tot,
                )
            };
            let x = BabyBearElem::ROU_FWD[po2 + EXP_PO2].pow(cycle);
            // TODO: what is this magic number 3?
            let y = (BabyBearElem::new(3) * x).pow(1 << po2);
            let ret = tot * (y - BabyBearElem::new(1)).inv();

            // SAFETY: This conversion is to make the check slice mutable, which should be
            // safe because each thread access will not overlap with each other.
            let check = unsafe {
                std::slice::from_raw_parts_mut(check.as_ptr() as *mut BabyBearElem, check.len())
            };
            for i in 0..BabyBearExtElem::EXT_SIZE {
                check[i * domain + cycle] = ret.elems()[i];
            }
        });
    }

    fn accumulate(
        &self,
        _preflight: &AccumPreflight,
        _ctrl: &CpuBuffer<BabyBearElem>,
        _global: &CpuBuffer<BabyBearElem>,
        _data: &CpuBuffer<BabyBearElem>,
        _mix: &CpuBuffer<BabyBearElem>,
        _accum: &CpuBuffer<BabyBearElem>,
        _steps: usize,
    ) {
        unimplemented!()
    }
}

pub fn segment_prover() -> Result<Box<dyn SegmentProver>> {
    let suite = Poseidon2HashSuite::new_suite();
    let hal = Rc::new(CpuHal::new(suite));
    let circuit_hal = Rc::new(CpuCircuitHal::new());
    Ok(Box::new(SegmentProverImpl::new(hal, circuit_hal)))
}
