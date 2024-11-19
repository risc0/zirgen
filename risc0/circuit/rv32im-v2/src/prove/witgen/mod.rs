// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

pub(crate) mod preflight;
#[cfg(test)]
mod tests;

use anyhow::Result;
use risc0_circuit_rv32im_v2_sys::RawPreflightCycle;
use risc0_core::scope;
use risc0_zkp::{
    core::digest::DIGEST_WORDS,
    field::{
        baby_bear::{BabyBear, BabyBearElem, BabyBearExtElem},
        Elem as _,
    },
    hal::Hal,
};

use super::hal::{CircuitWitnessGenerator, StepMode, WitnessBuffer};
use crate::{
    execute::{platform::LOOKUP_TABLE_CYCLES, segment::Segment},
    prove::{REG_COUNT_DATA, REG_COUNT_GLOBAL},
    zirgen::circuit::{LAYOUT_GLOBAL, LAYOUT_TOP},
};

const TOP_ECALL0_STATE_COL: usize = LAYOUT_TOP.inst_result.arm8.s0._super.offset;

const TOP_POSEIDON_STATE_COL: usize = LAYOUT_TOP
    .inst_result
    .arm9
    .state
    ._super
    .has_state
    ._super
    .offset;

const GLOBAL_INPUT_BASE: usize = LAYOUT_GLOBAL.input.values[0].low._super.offset;

const GLOBAL_IS_TERMINATE: usize = LAYOUT_GLOBAL.is_terminate._super.offset;

const GLOBAL_STATE_IN_BASE: usize = LAYOUT_GLOBAL.state_in.values[0].low._super.offset;

pub(crate) struct WitnessGenerator<H: Hal> {
    pub cycles: usize,
    pub global: WitnessBuffer<H>,
    pub data: WitnessBuffer<H>,
}

impl<H: Hal> WitnessGenerator<H> {
    pub fn new<C>(hal: &H, circuit_hal: &C, segment: &Segment, mode: StepMode) -> Result<Self>
    where
        H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
        C: CircuitWitnessGenerator<H>,
    {
        scope!("witgen");

        let trace = segment.preflight()?;
        let cycles = trace.cycles.len();

        tracing::trace!("{segment:#?}");
        tracing::trace!("{trace:#?}");

        assert_eq!(
            segment.suspend_cycle + segment.paging_cycles + LOOKUP_TABLE_CYCLES as u32 + 1,
            cycles as u32,
            "suspend_cycle: {} + paging_cycles: {} + {LOOKUP_TABLE_CYCLES} + 1 == trace.cycles",
            segment.suspend_cycle,
            segment.paging_cycles
        );
        // assert_eq!(cycles, 1 << segment.po2, "cycles == 1 << segment.po2");
        assert!(cycles <= 1 << segment.po2, "cycles <= 1 << segment.po2");
        let cycles = 1 << segment.po2;

        let mut global = vec![BabyBearElem::INVALID; REG_COUNT_GLOBAL];

        for i in 0..DIGEST_WORDS {
            // state in
            let low = segment.pre_digest.as_words()[i] & 0xffff;
            let high = segment.pre_digest.as_words()[i] >> 16;
            global[GLOBAL_STATE_IN_BASE + 2 * i + 0] = low.into();
            global[GLOBAL_STATE_IN_BASE + 2 * i + 1] = high.into();

            // input digest
            let low = 0u32;
            let high = 0u32;
            global[GLOBAL_INPUT_BASE + 2 * i + 0] = low.into();
            global[GLOBAL_INPUT_BASE + 2 * i + 1] = high.into();
        }

        // is_terminate
        global[GLOBAL_IS_TERMINATE] = 1u32.into();

        let global = WitnessBuffer {
            buf: hal.copy_from_elem("global", &global),
            rows: 1,
            cols: REG_COUNT_GLOBAL,
            checked_reads: true,
        };

        let data = scope!(
            "alloc(data)",
            WitnessBuffer {
                buf: hal.alloc_elem_init("data", cycles * REG_COUNT_DATA, BabyBearElem::INVALID),
                rows: cycles,
                cols: REG_COUNT_DATA,
                checked_reads: true,
            }
        );

        let mut injector = Injector::new(REG_COUNT_DATA);

        // Set stateful columns from 'top'
        for (i, cycle) in trace.cycles.iter().enumerate() {
            // tracing::trace!("[{i}] pc: {:#010x}, state: {}", cycle.pc, cycle.state);
            let extra_start = cycle.extra_idx as usize;
            let extra_end = if i == trace.cycles.len() - 1 {
                trace.extras.len()
            } else {
                trace.cycles[i + 1].extra_idx as usize
            };
            injector.set_cycle(i, cycle, &trace.extras[extra_start..extra_end]);
        }

        hal.scatter(
            &data.buf,
            &injector.index,
            &injector.offsets,
            &injector.values,
        );

        circuit_hal.generate_witness(mode, &trace, trace.cycles.len(), &global, &data)?;

        // Zero out 'invalid' entries in data and output.
        scope!("zeroize", {
            hal.eltwise_zeroize_elem(&global.buf);
            hal.eltwise_zeroize_elem(&data.buf);
        });

        Ok(Self {
            cycles,
            global,
            data,
        })
    }
}

struct Injector {
    cols: usize,
    offsets: Vec<u32>,
    values: Vec<BabyBearElem>,
    index: Vec<u32>,
}

impl Injector {
    pub fn new(cols: usize) -> Self {
        Self {
            cols,
            offsets: vec![],
            values: vec![],
            index: vec![],
        }
    }

    pub fn set_cycle(&mut self, row: usize, cycle: &RawPreflightCycle, extras: &[u32]) {
        self.set(row, LAYOUT_TOP.next_pc_low._super.offset, cycle.pc & 0xffff);
        self.set(
            row,
            LAYOUT_TOP.next_pc_high._super.offset + 1,
            cycle.pc >> 16,
        );
        self.set(row, LAYOUT_TOP.next_state_0._super.offset, cycle.state);
        self.set(
            row,
            LAYOUT_TOP.next_machine_mode._super.offset,
            cycle.machine_mode as u32,
        );
        let base = if extras.len() == 3 {
            TOP_ECALL0_STATE_COL
        } else {
            TOP_POSEIDON_STATE_COL
        };
        for (i, &extra) in extras.iter().enumerate() {
            // tracing::trace!("set({base} + {i}, {extra:#010x})");
            self.set(row, base + i, extra);
        }
        self.index.push(self.offsets.len() as u32);
    }

    fn set(&mut self, row: usize, col: usize, value: u32) {
        let idx = row * self.cols + col;
        self.offsets.push(idx as u32);
        self.values.push(value.into());
    }
}
