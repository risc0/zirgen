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

use crate::micro_circuit;
use anyhow::{bail, Result};
use core::cell::RefCell;
use micro_circuit::{CircuitField, ExtVal, MixState, Val};
use risc0_binfmt::MemoryImage;
use risc0_zkp::{
    field::Elem,
    hal::cpu::{CpuBuffer, CpuHal},
};
use risc0_zkvm::Segment;
use risc0_zkvm_platform::WORD_SIZE;
use zirgen_dsl::BoundLayout;

pub struct MicroCircuitCpuHal<'a> {
    segment: &'a Segment,
}

impl<'a> MicroCircuitCpuHal<'a> {
    pub fn new(segment: &'a Segment) -> Self {
        Self { segment }
    }
}

// Ugh, gross hack to access the "pre_image" from the Segment which is private.
// TODO: figure out what to do here; should we make Segment members `pub'?
fn get_segment_pre_image(segment: &Segment) -> Result<MemoryImage> {
    let json = serde_json::to_value(segment)?;
    if let serde_json::Value::Object(fields) = json {
        if let Some(image) = fields.get("pre_image") {
            let mem_image: MemoryImage = serde_json::from_value(image.clone())?;
            return Ok(mem_image);
        }
    }
    bail!("Unable to find pre image in segment")
}

impl<'a> micro_circuit::CircuitHal<'a, CpuHal<CircuitField>> for MicroCircuitCpuHal<'a> {
    fn step_exec(
        &self,
        tot_cycles: usize,
        data: &CpuBuffer<Val>,
        global: &CpuBuffer<Val>,
    ) -> Result<()> {
        let image = &RefCell::new(get_segment_pre_image(self.segment)?);

        zirgen_dsl::cpu::run_serial(
            micro_circuit::get_named_buffers([("data", data), ("global", global)]),
            tot_cycles,
            |ctx, cycle| -> Result<()> {
                // Clone it so we run with identical inputs and outputs for each cycle.
                let exec_context = CpuExecContext { image, cycle };
                let ctx = ctx.wrap(&exec_context);

                let data = micro_circuit::get_data_buffer(&ctx);
                micro_circuit::exec_top(&ctx, BoundLayout::new(micro_circuit::LAYOUT_TOP, data))?;

                Ok(())
            },
        )
    }

    fn step_accum(
        &self,
        tot_cycles: usize,
        accum: &CpuBuffer<Val>,
        data: &CpuBuffer<Val>,
        global: &CpuBuffer<Val>,
    ) -> Result<()> {
        let image = &RefCell::new(get_segment_pre_image(self.segment)?);

        zirgen_dsl::cpu::run_serial(
            micro_circuit::get_named_buffers([
                ("accum", accum),
                ("data", data),
                ("global", global),
            ]),
            tot_cycles,
            |ctx, cycle| -> Result<()> {
                // Clone it so we run with identical inputs and outputs for each cycle.
                let exec_context = CpuExecContext { image, cycle };
                let ctx = ctx.wrap(&exec_context);

                let data = micro_circuit::get_data_buffer(&ctx);
                let accum = micro_circuit::get_accum_buffer(&ctx);
                micro_circuit::exec_top_accum(
                    &ctx,
                    BoundLayout::new(micro_circuit::LAYOUT_TOP, data),
                    BoundLayout::new(micro_circuit::LAYOUT_TOP_ACCUM, accum),
                )?;

                Ok(())
            },
        )
    }
}

pub struct CpuExecContext<'a> {
    image: &'a RefCell<MemoryImage>,
    cycle: usize,
}

fn to_u16_vals(orig: u32) -> (Val, Val) {
    let lo = orig & 0xFFFF;
    let hi = orig >> 16;
    (Val::new(lo), Val::new(hi))
}

impl<'a> CpuExecContext<'a> {
    fn read_ram(&self, addr: u32) -> u32 {
        let mut bytes = [0u8; WORD_SIZE];
        self.image
            .borrow_mut()
            .load_region_in_page(addr * 4, &mut bytes)
            .unwrap();
        u32::from_le_bytes(bytes)
    }

    fn write_ram(&self, addr: u32, data: u32) {
        self.image
            .borrow_mut()
            .store_region_in_page(addr * 4, &data.to_le_bytes())
    }

    pub(crate) fn divide(
        &self,
        numer_lo: Val,
        numer_hi: Val,
        denom_lo: Val,
        denom_hi: Val,
        sign_type: Val,
    ) -> Result<(Val, Val, Val, Val)> {
        let mut numer: u32 = numer_lo.as_u32() | (numer_hi.as_u32() << 16);
        let mut denom: u32 = denom_lo.as_u32() | (denom_hi.as_u32() << 16);
        let sign: u32 = sign_type.as_u32();
        // log::debug!("divide: [{sign}] {numer} / {denom}");
        let ones_comp = (sign == 2) as u32;
        let neg_numer = sign != 0 && (numer as i32) < 0;
        let neg_denom = sign == 1 && (denom as i32) < 0;
        if neg_numer {
            numer = (!numer).overflowing_add(1 - ones_comp).0;
        }
        if neg_denom {
            denom = (!denom).overflowing_add(1 - ones_comp).0;
        }
        let (mut quot, mut rem) = if denom == 0 {
            (0xffffffff, numer)
        } else {
            (numer / denom, numer % denom)
        };
        let quot_neg_out =
            (neg_numer as u32 ^ neg_denom as u32) - ((denom == 0) as u32 * neg_numer as u32);
        if quot_neg_out != 0 {
            quot = (!quot).overflowing_add(1 - ones_comp).0;
        }
        if neg_numer {
            rem = (!rem).overflowing_add(1 - ones_comp).0;
        }
        // log::debug!("  quot: {quot}, rem: {rem}");
        let (quot_lo, quot_hi) = to_u16_vals(quot);
        let (rem_lo, rem_hi) = to_u16_vals(rem);
        Ok((quot_lo, quot_hi, rem_lo, rem_hi))
    }

    pub(crate) fn is_first_cycle(&self) -> Result<Val> {
        Ok(if self.cycle == 0 { Val::ONE } else { Val::ZERO })
    }
    pub(crate) fn get_cycle(&self) -> Result<Val> {
        Ok(Val::new(self.cycle as u32))
    }
    pub(crate) fn memory_peek(&self, addr: Val) -> Result<(Val, Val)> {
        let addr = addr.as_u32();
        if addr >= 0x40000000 {
            panic!("Peek address 0x{:08x} * 4 out of range", addr)
        }
        let data = self.read_ram(addr);
        log::debug!("Peek address 0x{:08x} -> 0x{:08x}", addr * 4, data);
        Ok(to_u16_vals(data))
    }
    pub(crate) fn memory_poke(&self, addr: Val, data_lo: Val, data_hi: Val) -> Result<()> {
        let addr = addr.as_u32();
        if addr >= 0x40000000 {
            bail!("Poke address out of range")
        }
        let data_lo: u32 = data_lo.as_u32();
        if data_lo >= 0x10000 {
            bail!("Poke data lo out of range");
        }
        let data_hi: u32 = data_hi.as_u32();
        if data_hi >= 0x10000 {
            bail!("Poke data hi out of range");
        }
        self.write_ram(addr, (data_hi << 16) | data_lo);
        Ok(())
    }

    pub(crate) fn host_read_prepare(&self, _fp: Val, _len: Val) -> Result<Val> {
        // TODO: Implement
        Ok(Val::new(0))
    }

    pub(crate) fn host_read_words(
        &self,
        _coutn: Val,
    ) -> Result<(Val, Val, Val, Val, Val, Val, Val, Val)> {
        // TODO: Implement
        let zero = Val::new(0);
        Ok((zero, zero, zero, zero, zero, zero, zero, zero))
    }

    pub(crate) fn host_write(
        &self,
        _fp: Val,
        _addrLow: Val,
        _addrHigh: Val,
        _len: Val,
    ) -> Result<Val> {
        // TODO: Implement
        Ok(Val::new(0))
    }

    pub(crate) fn get_major_minor(
        &self,
        _state: Val,
        _mode: Val,
        _instLow: Val,
        _instHigh: Val,
    ) -> Result<(Val, Val)> {
        // TODO: Implement
        Ok((Val::new(0), Val::new(0)))
    }

    pub(crate) fn lookup_delta(&self, _arg1: Val, _arg2: Val, _arg3: Val) -> Result<()> {
        // TODO: Implement
        Ok(())
    }
    pub(crate) fn print(&self, _arg1: Val) -> Result<()> {
        // TODO: Implement
        Ok(())
    }
    pub(crate) fn log_inst_info(&self, _arg1: Val, _arg2: Val, _arg3: Val) -> Result<()> {
        // TODO: Implement
        Ok(())
    }

    pub(crate) fn log(&self, message: &str, x: &[Val]) -> Result<()> {
        zirgen_dsl::codegen::default_log(message, x)
    }
}

impl<'a> risc0_zkp::hal::CircuitHal<CpuHal<CircuitField>> for MicroCircuitCpuHal<'a> {
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
            micro_circuit::get_named_buffers(zirgen_dsl::eval_check_named_buffers(groups, globals));

        zirgen_dsl::cpu::eval_check(
            check,
            buffers,
            poly_mix,
            po2,
            cycles,
            |ctx, poly_mix| -> Result<MixState> { micro_circuit::validity_regs_(&ctx, poly_mix) },
        )
        .expect("Expected eval check to succeed");
    }
}
