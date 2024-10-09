// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use crate::keccak_circuit;
use anyhow::{anyhow, Result};
use core::cell::RefCell;
use keccak_circuit::{CircuitField, ExtVal, MixState, Val};
use risc0_zirgen_dsl::{CycleContext, CycleRow, GlobalRow};
use risc0_zkp::{
    field::Elem,
    hal::cpu::{CpuBuffer, CpuHal},
};
use std::collections::VecDeque;

pub struct CpuCircuitHal {
    mem: RefCell<Vec<Val>>,
    input: RefCell<VecDeque<u32>>,
}

fn val_array<const SIZE: usize>(vals: [usize; SIZE]) -> [Val; SIZE] {
    vals.map(|val| Val::new(val as u32))
}

impl CpuCircuitHal {
    pub fn new(input: VecDeque<u32>) -> Self {
        Self {
            mem: RefCell::new(Vec::new()),
            input: RefCell::new(input),
        }
    }
}

pub struct CpuExecContext<'a> {
    cycle: usize,
    tot_cycles: usize,
    mem: &'a RefCell<Vec<Val>>,

    elems_per_word: &'a RefCell<usize>,
    input: &'a RefCell<VecDeque<u32>>,
    input_elems: &'a RefCell<VecDeque<Val>>,
}

impl<'a> CycleContext for CpuExecContext<'a> {
    fn cycle(&self) -> usize {
        self.cycle
    }
    fn tot_cycles(&self) -> usize {
        self.tot_cycles
    }
}

impl<'a> CpuExecContext<'a> {
    pub fn get_cycle(&self) -> Result<Val> {
        Ok(Val::new(self.cycle as u32))
    }

    pub fn simple_memory_peek(&self, addr: Val) -> Result<Val> {
        let addr = u32::from(addr) as usize;
        self.mem
            .borrow()
            .get(addr)
            .as_deref()
            .ok_or(anyhow!("invalid address {addr}"))
            .copied()
    }

    pub fn simple_memory_poke(&self, addr: Val, val: Val) -> Result<()> {
        let addr = u32::from(addr) as usize;
        let mut mem = self.mem.borrow_mut();
        if mem.len() <= addr {
            mem.resize(addr + 1, Val::ZERO);
        }
        mem[addr] = val;
        Ok(())
    }

    pub fn configure_input(&self, elems_per_word: Val) -> Result<()> {
        assert!(self.input_elems.borrow().is_empty());
        (*self.elems_per_word.borrow_mut()) = u32::from(elems_per_word) as usize;
        Ok(())
    }
    pub fn read_input(&self) -> Result<Val> {
        let mut elems = self.input_elems.borrow_mut();
        if elems.is_empty() {
            let word = self.input.borrow_mut().pop_front().expect("Input underrun");
            match *self.elems_per_word.borrow() {
                1 => elems.push_back(Val::from(word)),
                2 => elems.extend([word & 0xFF, word >> 16].map(Val::new)),
                4 => elems.extend(word.to_le_bytes().map(u32::from).map(Val::new)),
                elems_per_word @ _ => panic!("Unknown input configuration {}", elems_per_word),
            }
        }

        Ok(elems.pop_front().expect("Input underrun"))
    }
    pub fn log(&self, message: &str, x: impl AsRef<[Val]>) -> Result<()> {
        risc0_zirgen_dsl::codegen::default_log(message, x.as_ref())
    }

    // Stubs so we can compile with calculator circuit for rapid iteration
    pub fn get_val_from_user(&self) -> Result<Val> {
        unimplemented!()
    }
    pub fn output_to_user(&self, _ov: Val) -> Result<()> {
        unimplemented!()
    }
}

impl<'a> keccak_circuit::CircuitHal<'a, CpuHal<CircuitField>> for CpuCircuitHal {
    fn step_exec(
        &self,
        tot_cycles: usize,
        data: &CpuBuffer<Val>,
        global: &CpuBuffer<Val>,
    ) -> Result<()> {
        let elems_per_word = &RefCell::new(0);
        let input_elems: &RefCell<VecDeque<Val>> = &RefCell::new(Default::default());

        let data = &data.as_slice_sync();
        let data = CycleRow { buf: data };
        let global = global.as_slice_sync();
        let global = GlobalRow { buf: &global };

        for cycle in 0..tot_cycles {
            let exec_context = CpuExecContext {
                mem: &self.mem,
                cycle,
                tot_cycles,
                elems_per_word,
                input: &self.input,
                input_elems,
            };

            keccak_circuit::step_top(&exec_context, &data, &global)?;
        }
        Ok(())
    }

    fn step_accum(
        &self,
        _tot_cycles: usize,
        _accum: &CpuBuffer<Val>,
        _data: &CpuBuffer<Val>,
        _global: &CpuBuffer<Val>,
    ) -> Result<()> {
        // The calculator circuit has no arguments, so there's no work to do here
        Ok(())
    }
}

impl risc0_zkp::hal::CircuitHal<CpuHal<CircuitField>> for CpuCircuitHal {
    fn accumulate(
        &self,
        ctrl: &CpuBuffer<Val>,
        io: &CpuBuffer<Val>,
        data: &CpuBuffer<Val>,
        mix: &CpuBuffer<Val>,
        accum: &CpuBuffer<Val>,
        steps: usize,
    ) {
        unimplemented!()
    }

    fn eval_check(
        &self,
        check: &CpuBuffer<Val>,
        groups: &[&CpuBuffer<Val>],
        globals: &[&CpuBuffer<Val>],
        poly_mix: ExtVal,
        po2: usize,
        cycles: usize,
    ) {
        /*        let buffers = keccak_circuit::get_named_buffers(
            risc0_zirgen_dsl::eval_check_named_buffers(groups, globals),
        );

        risc0_zirgen_dsl::cpu::eval_check(
            check,
            buffers,
            poly_mix,
            po2,
            cycles,
            |ctx, poly_mix| -> Result<MixState> {
                keccak_circuit::validity_regs(
                    &ctx,
                    poly_mix,
                    keccak_circuit::get_accum_buffer(&ctx),
                    keccak_circuit::get_data_buffer(&ctx),
                    keccak_circuit::get_global_buffer(&ctx),
                    keccak_circuit::get_mix_buffer(&ctx),
                )
            },
        )
        .expect("Expected eval check to succeed");*/
        unimplemented!()
    }
}
