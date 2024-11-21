// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

pub(crate) mod poseidon2;
pub(crate) mod preflight;
#[cfg(test)]
mod tests;

use std::iter::zip;

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

use self::poseidon2::Poseidon2State;
use super::hal::{CircuitWitnessGenerator, StepMode, WitnessBuffer};
use crate::{
    execute::{
        addr::WordAddr,
        platform::{LOOKUP_TABLE_CYCLES, MERKLE_TREE_END_ADDR},
        segment::Segment,
    },
    prove::{REG_COUNT_DATA, REG_COUNT_GLOBAL},
    zirgen::circuit::{LAYOUT_GLOBAL, LAYOUT_TOP},
};

pub(crate) struct WitnessGenerator<H: Hal> {
    pub cycles: usize,
    pub global: WitnessBuffer<H>,
    pub data: WitnessBuffer<H>,
}

impl<H: Hal> WitnessGenerator<H> {
    pub fn new<C>(
        hal: &H,
        circuit_hal: &C,
        segment: &Segment,
        mode: StepMode,
        nonce: BabyBearExtElem,
    ) -> Result<Self>
    where
        H: Hal<Field = BabyBear, Elem = BabyBearElem, ExtElem = BabyBearExtElem>,
        C: CircuitWitnessGenerator<H>,
    {
        scope!("witgen");

        let trace = segment.preflight(nonce)?;
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
            global[LAYOUT_GLOBAL.state_in.values[i].low._super.offset] = low.into();
            global[LAYOUT_GLOBAL.state_in.values[i].high._super.offset] = high.into();

            // input digest
            let low = 0u32;
            let high = 0u32;
            global[LAYOUT_GLOBAL.input.values[i].low._super.offset] = low.into();
            global[LAYOUT_GLOBAL.input.values[i].high._super.offset] = high.into();
        }

        // rng
        for (i, &elem) in trace.nonce.elems().iter().enumerate() {
            global[LAYOUT_GLOBAL.rng._super.offset + i] = elem;
        }

        // is_terminate
        global[LAYOUT_GLOBAL.is_terminate._super.offset] = 1u32.into();

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
        for (row, back) in trace.backs.iter().enumerate() {
            let cycle = &trace.cycles[row];
            // tracing::trace!("[{i}] pc: {:#010x}, state: {}", cycle.pc, cycle.state);
            match back {
                preflight::Back::None => {}
                preflight::Back::Ecall(s0, s1, s2) => {
                    injector.set(row, LAYOUT_TOP.inst_result.arm8.s0._super.offset, *s0);
                    injector.set(row, LAYOUT_TOP.inst_result.arm8.s1._super.offset, *s1);
                    injector.set(row, LAYOUT_TOP.inst_result.arm8.s2._super.offset, *s2);
                }
                preflight::Back::Poseidon2(p2_state) => {
                    for (col, value) in zip(Poseidon2State::offsets(), p2_state.as_array()) {
                        injector.set(row, col, value);
                    }
                }
            }
            injector.set_cycle(row, cycle);
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

    pub fn set_cycle(&mut self, row: usize, cycle: &RawPreflightCycle) {
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
        self.index.push(self.offsets.len() as u32);
    }

    fn set(&mut self, row: usize, col: usize, value: u32) {
        let idx = row * self.cols + col;
        self.offsets.push(idx as u32);
        self.values.push(value.into());
    }
}

fn node_idx_to_addr(idx: u32) -> WordAddr {
    MERKLE_TREE_END_ADDR - idx * DIGEST_WORDS as u32
}

fn node_addr_to_idx(addr: WordAddr) -> u32 {
    (MERKLE_TREE_END_ADDR - addr).0 / DIGEST_WORDS as u32
}
