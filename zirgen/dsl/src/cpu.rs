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

use crate::{BufferRow, Buffers};
use anyhow::Result;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use risc0_core::field::{Elem, ExtElem, RootsOfUnity};
use risc0_zkp::adapter::MixState;
use risc0_zkp::hal::cpu::{CpuBuffer, SyncSlice};
use risc0_zkp::INV_RATE;

pub type CpuBuffers<'a, Val, Context> = Buffers<RowSlice<'a, Val>, GlobalsRow<'a, Val>, Context>;

#[derive(Clone)]
pub struct RowSlice<'a, ValType: Default + Clone> {
    buf: &'a SyncSlice<'a, ValType>,
    cycle: usize,
    tot_cycles: usize,
    // TODO: Row slice should probably be parameterized based on
    // back_factor so that it can be optimized away or converted into
    // faster operations than multiply.
    back_factor: usize,
}

impl<'a, ValType: Default + Clone> RowSlice<'a, ValType> {
    pub fn new(
        buf: &'a SyncSlice<'a, ValType>,
        cycle: usize,
        tot_cycles: usize,
        back_factor: usize,
    ) -> Self {
        assert_eq!(
            tot_cycles & (tot_cycles - 1),
            0,
            "{tot_cycles} must be a power of 2"
        );
        Self {
            buf,
            cycle,
            tot_cycles,
            back_factor,
        }
    }

    fn resolve_offset(&self, offset: usize, back: usize) -> usize {
        offset * self.tot_cycles
            + (self.cycle.wrapping_sub(back * self.back_factor) & (self.tot_cycles - 1))
    }
}

impl<'a, ValType: Default + Clone + Copy> BufferRow for RowSlice<'a, ValType> {
    type ValType = ValType;

    fn load(&self, offset: usize, back: usize) -> ValType {
        self.buf.get(self.resolve_offset(offset, back))
    }

    fn store(&self, offset: usize, val: ValType) {
        self.buf.set(self.resolve_offset(offset, 0), val);
    }
}

#[derive(Clone)]
pub struct GlobalsRow<'a, ValType: Default + Clone + Copy> {
    buf: &'a SyncSlice<'a, ValType>,
}

impl<'a, ValType: Default + Clone + Copy> GlobalsRow<'a, ValType> {
    pub fn new(buf: &'a SyncSlice<'a, ValType>) -> Self {
        Self { buf }
    }
}

impl<'a, ValType: Default + Clone + Copy> BufferRow for GlobalsRow<'a, ValType> {
    type ValType = ValType;

    fn load(&self, offset: usize, back: usize) -> ValType {
        assert_eq!(back, 0);
        self.buf.get(offset)
    }

    fn store(&self, offset: usize, val: ValType) {
        self.buf.set(offset, val);
    }
}

pub fn eval_check<
    CircuitFn: Fn(
            /*ctx=*/ Buffers<RowSlice<Val>, GlobalsRow<Val>, ()>,
            /*poly_mix=*/ ExtVal,
        ) -> Result<MixState<ExtVal>>
        + Sync,
    Val: Elem + RootsOfUnity,
    ExtVal: ExtElem<SubElem = Val>,
>(
    check: &CpuBuffer<Val>,
    buffers: Buffers<&CpuBuffer<Val>, &CpuBuffer<Val>, ()>,
    poly_mix: ExtVal,
    po2: usize,
    cycles: usize,
    circuit_fn: CircuitFn,
) -> Result<()> {
    assert_eq!(1 << po2, cycles);

    const EXP_PO2: usize = risc0_zkp::core::log2_ceil(INV_RATE);
    let domain = cycles * INV_RATE;

    let check_slice = check.as_slice_sync();

    let buffers = buffers
        .map_rows(CpuBuffer::as_slice_sync)
        .map_globals(CpuBuffer::as_slice_sync);

    (0..domain)
        .into_par_iter()
        .try_for_each(|cycle| -> Result<()> {
            let cycle_buffers = buffers
                .as_ref()
                .map_rows(|buf| RowSlice::new(buf, cycle, domain, INV_RATE))
                .map_globals(GlobalsRow::new)
                .wrap(());

            let res = circuit_fn(cycle_buffers, poly_mix)?;
            let x = Val::ROU_FWD[po2 + EXP_PO2].pow(cycle);
            // TODO: what is this magic number 3?
            let y = (Val::from_u64(3) * x).pow(1 << po2);
            let ret = res.tot * (y - Val::from_u64(1)).inv();

            for i in 0..ExtVal::EXT_SIZE {
                check_slice.set(i * domain + cycle, ret.subelems()[i]);
            }
            Ok(())
        })?;

    Ok(())
}

pub fn run_serial<Val: Elem>(
    buffers: Buffers<&CpuBuffer<Val>, &CpuBuffer<Val>, ()>,
    tot_cycles: usize,
    mut circuit_fn: impl for<'a> FnMut(
        Buffers<RowSlice<'a, Val>, GlobalsRow<'a, Val>, ()>,
        /*cycle=*/ usize,
    ) -> Result<()>,
) -> Result<()> {
    let buffers = buffers
        .map_rows(CpuBuffer::as_slice_sync)
        .map_globals(CpuBuffer::as_slice_sync);

    for cycle in 0..tot_cycles {
        let cycle_ctx = buffers
            .as_ref()
            .map_globals(GlobalsRow::new)
            .map_rows(|buf| RowSlice::new(buf, cycle, tot_cycles, /*back_factor=*/ 1));
        circuit_fn(cycle_ctx, cycle)?
    }

    Ok(())
}
