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

use super::calc_circuit;
use anyhow::Result;
use calc_circuit::{CircuitField, ExtVal, MixState, Val};
use core::cell::RefCell;
use risc0_zkp::hal::cpu::{CpuBuffer, CpuHal};
use std::collections::VecDeque;

pub struct CpuCircuitHal {
    from_user: VecDeque<Val>,
    to_user: VecDeque<Val>,
}

fn val_array<const SIZE: usize>(vals: [usize; SIZE]) -> [Val; SIZE] {
    vals.map(|val| Val::new(val as u32))
}

impl CpuCircuitHal {
    pub fn new(op: usize, lhs: usize, rhs: usize) -> Self {
        Self {
            from_user: val_array([op, lhs, rhs]).into(),
            to_user: [].into(),
        }
    }
}

pub struct CpuExecContext {
    from_user: RefCell<VecDeque<Val>>,
    to_user: RefCell<VecDeque<Val>>,
}

impl CpuExecContext {
    pub fn get_val_from_user(&self) -> Result<Val> {
        let mut from_user = RefCell::borrow_mut(&self.from_user);
        eprintln!("get_val_from_user, from_user = {from_user:?}");
        Ok(from_user.pop_front().unwrap())
    }

    pub fn output_to_user(&self, val: Val) -> Result<()> {
        let mut to_user = RefCell::borrow_mut(&self.to_user);
        to_user.push_back(val);
        eprintln!("output_to_user, to_user = {to_user:?}");
        Ok(())
    }

    pub fn log(&self, message: &str, x: &[Val]) -> Result<()> {
        zirgen_dsl::codegen::default_log(message, x)
    }
}

impl<'a> calc_circuit::CircuitHal<'a, CpuHal<CircuitField>> for CpuCircuitHal {
    fn step_exec(
        &self,
        tot_cycles: usize,
        data: &CpuBuffer<Val>,
        global: &CpuBuffer<Val>,
    ) -> Result<()> {
        zirgen_dsl::cpu::run_serial(
            calc_circuit::get_named_buffers([("data", data), ("global", global)]),
            tot_cycles,
            |ctx, _cycle| -> Result<()> {
                // Clone it so we run with identical inputs and outputs for each cycle.
                let exec_context = CpuExecContext {
                    from_user: RefCell::new(self.from_user.clone()),
                    to_user: RefCell::new(self.to_user.clone()),
                };
                let ctx = ctx.wrap(&exec_context);

                calc_circuit::step_top(
                    &ctx,
                    calc_circuit::get_data_buffer(&ctx),
                    calc_circuit::get_global_buffer(&ctx),
                )?;

                Ok(())
            },
        )
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
        let buffers =
            calc_circuit::get_named_buffers(zirgen_dsl::eval_check_named_buffers(groups, globals));

        zirgen_dsl::cpu::eval_check(
            check,
            buffers,
            poly_mix,
            po2,
            cycles,
            |ctx, poly_mix| -> Result<MixState> {
                calc_circuit::validity_regs(
                    &ctx,
                    poly_mix,
                    calc_circuit::get_data_buffer(&ctx),
                    calc_circuit::get_global_buffer(&ctx),
                )
            },
        )
        .expect("Expected eval check to succeed");
    }
}
