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

#[macro_use]
pub mod codegen;

//pub mod cpu;

mod buffers;
pub use buffers::{BufferSpec, Buffers};
use risc0_core::{field::Elem,field::ExtElem};
use risc0_zkp::{hal::cpu::SyncSlice, layout::Reg};

/// This trait represents the set of registers available for a
/// specific cycle.  For instance, this can either be a row in the
/// execution trace or a set of global registers that are common to
/// all cycles.
pub trait BufferRow: Clone {
    type ValType: Clone + Copy + Default;

    fn load(&self, ctx: &impl CycleContext, offset: usize, back: usize) -> Self::ValType;
    fn store(&self, ctx: &impl CycleContext, offset: usize, val: Self::ValType);
}

pub trait CycleContext {
    fn cycle(&self) -> usize;
    fn tot_cycles(&self) ->usize;
    
    fn offset_this_cycle(&self, offset: usize, back: usize) -> usize {
        let tot_cycles = self.tot_cycles();
        let cycle = (self.cycle() + tot_cycles - back) % tot_cycles;
        offset * tot_cycles + cycle
    }
}

#[derive(Clone)]
pub struct GlobalRow<'a, E: Default + Clone + Copy> {
    pub buf: &'a SyncSlice<'a, E>
}

impl<'a, E: Copy + Clone + Default> BufferRow for GlobalRow<'a, E> {
    type ValType = E;

    fn load(&self, _ctx: &impl CycleContext, offset: usize, back: usize) -> E {
        assert_eq!(back, 0);
        self.buf.get(offset)
    }
    fn store(&self, _ctx: &impl CycleContext, offset: usize, val: E) {
        self.buf.set(offset, val)
    }
}

#[derive(Clone)]
pub struct CycleRow<'a, E: Default + Clone + Copy> {
    pub buf: &'a SyncSlice<'a, E>,
}

impl<'a, E: Copy + Clone + Default> BufferRow for CycleRow<'a, E> {
    type ValType = E;

    fn load(&self, ctx: &impl CycleContext, offset: usize, back: usize) -> E {
        self.buf.get(ctx.offset_this_cycle(offset, back))
    }
    fn store(&self, ctx: &impl CycleContext, offset: usize, val: E) {
        self.buf.set(ctx.offset_this_cycle(offset, 0) , val)
    }

}

/// This represents a layout bound to a buffer.
#[derive(Clone, Copy)]
pub struct BoundLayout<'a, L: 'static, B: BufferRow> {
    pub layout: &'static L,
    pub buf: &'a B,
}

/*// We can't derive these since derive(Copy) also puts bounds on BufferRow.
impl<'a, L, B: BufferRow> Clone for BoundLayout<'a, L, B> {
    fn clone(&self) -> Self {
        BoundLayout {
            buf: self.buf,
            layout: self.layout,
        }
    }
}*/

impl<'a, L: PartialEq, B: BufferRow> PartialEq for BoundLayout<'a, L, B> {
    fn eq(&self, other: &BoundLayout<L, B>) -> bool {
        // We only need to compare the layout values
        self.layout == other.layout
    }
}

impl<'a, L, B: BufferRow> BoundLayout<'a, L, B> {
    pub fn new(layout: &'static L, buf: &'a B) -> Self {
        Self { layout, buf }
    }
    pub fn map<NewL, F: FnOnce(&'static L) -> &'static NewL>(
        &self,
        f: F,
    ) -> BoundLayout<'a, NewL, B> {
        BoundLayout {
            layout: f(self.layout),
            buf: self.buf,
        }
    }


    pub fn layout(&self) -> &'static L {
        self.layout
    }
    pub fn buf(&self) -> &B {
        &self.buf
    }
}


impl<'a, B: BufferRow> BoundLayout<'a, Reg, B> {
    pub fn load_ext<E: ExtElem<SubElem = B::ValType>>(&self,
        ctx: &impl CycleContext,
        back: usize
    ) -> E {
        E::from_subelems(
            (0..E::EXT_SIZE)
                .map(|idx| self.buf.load(ctx, self.layout.offset + idx, back)),
        )
    }
    pub fn store_ext<E: ExtElem<SubElem = B::ValType>>(&self,ctx: &impl CycleContext, val: E) {
        for (idx, elem) in val.subelems().into_iter().enumerate() {
            self.buf.store(ctx, self.layout.offset + idx, *elem)
        }
    }
}
     
impl<'a, E: Elem,  B: BufferRow<ValType=E> >BoundLayout<'a, Reg,  B> {
    pub fn load(&self, ctx: &impl CycleContext, back: usize) -> E {
        self.buf.load(ctx, self.layout.offset, back)
    }
    pub fn store(&self, ctx: &impl CycleContext, val: E) {
        self.buf.store(ctx, self.layout.offset, val);
    }
}

/*
// Takes the groups and globals provided in the
// risc0_zkp::hal::CircuitHal::eval_check API and returns them with
// names.
//
// Note that this order is different than the order required by poly_ext.
//
// TODO: This mapping should not be hardcoded.
pub fn eval_check_named_buffers<'a, T: Copy>(
    groups: &'a [T],
    globals: &'a [T],
) -> impl IntoIterator<Item = (&'static str, T)> + 'a {
    const GROUP_NAMES: &[&'static str] = &["accum", "code", "data"];
    const GLOBAL_NAMES: &[&'static str] = &["mix", "global"];
    assert_eq!(GROUP_NAMES.len(), groups.len());
    assert_eq!(GLOBAL_NAMES.len(), globals.len());

    GROUP_NAMES
        .iter()
        .copied()
        .zip(groups.iter().copied())
        .chain(GLOBAL_NAMES.iter().copied().zip(globals.iter().copied()))
}

// Takes the groups and globals provided to
// the risc0_zkp::adapter::PolyExt::poly_ext API in the "args" parameter and returns
// them with names.
//
// Note that this order is different than the order required by eval_check.
//
// TODO: This mapping should not be hardcoded.
pub fn poly_ext_named_buffers<'a, T: Copy>(
    args: &'a [T],
) -> impl IntoIterator<Item = (&'static str, T)> + 'a {
    const GLOBAL_NAMES: &[&'static str] = &["global", "mix"];
    assert_eq!(GLOBAL_NAMES.len(), args.len());

    GLOBAL_NAMES.iter().copied().zip(args.iter().copied())
}
*/
