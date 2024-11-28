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
mod buffers;

pub use buffers::{BufferSpec, Buffers};
use core::fmt::Debug;
use risc0_core::{field::Elem, field::ExtElem};
use risc0_zkp::{hal::cpu::SyncSlice, layout::Reg};

/// This trait represents the set of registers available for a
/// specific cycle.  For instance, this can either be a row in the
/// execution trace or a set of global registers that are common to
/// all cycles.
#[derive(Clone, Copy)]
pub struct BufferRow<'a, E: Default + Clone> {
    pub buf: &'a SyncSlice<'a, E>,
    kind: BufferKind,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum BufferKind {
    Global,
    Cycle,
}

impl<'a, E: Default + Clone> Debug for BufferRow<'a, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::result::Result<(), std::fmt::Error> {
        write!(f, "BufferRow({:?})", self.kind)
    }
}

impl<'a, E: Default + Clone> PartialEq for BufferRow<'a, E> {
    fn eq(&self, rhs: &Self) -> bool {
        self.kind == rhs.kind && ((self.buf as *const _) == (rhs.buf as *const _))
    }
}

impl<'a, E: Elem> BufferRow<'a, E> {
    pub fn global(buf: &'a SyncSlice<'a, E>) -> Self {
        Self {
            buf,
            kind: BufferKind::Global,
        }
    }

    pub fn cycle(buf: &'a SyncSlice<'a, E>) -> Self {
        Self {
            buf,
            kind: BufferKind::Cycle,
        }
    }

    pub fn load(&self, ctx: &impl CycleContext, offset: usize, back: usize) -> E {
        match self.kind {
            BufferKind::Global => {
                assert_eq!(back, 0);
                let val = self.buf.get(offset);
                tracing::trace!("Load {val:?} from global offset {offset}");
                debug_assert!(val.is_valid(), "Global offset {offset}");
                val
            }
            BufferKind::Cycle => {
                let adj_offset = ctx.offset_this_cycle(offset, back);
                let val = self.buf.get(adj_offset);
                tracing::trace!("Load {val:?} from offset {offset} back {back}");
                val
            }
        }
    }
    pub fn store(&self, ctx: &impl CycleContext, offset: usize, val: E) {
        match self.kind {
            BufferKind::Global => {
                if cfg!(debug_assertions) {
                    let old_val = self.buf.get(offset);
                    assert!(
                        !old_val.is_valid() || old_val == val,
                        "Global offset {offset}, Old value: {old_val:?}, New value: {val:?}"
                    );
                }
                tracing::trace!("Store {val:?} to global offset {offset}");
                self.buf.set(offset, val)
            }
            BufferKind::Cycle => {
                if cfg!(debug_assertions) {
                    let old_val = self.buf.get(ctx.offset_this_cycle(offset, 0));
                    assert!(
                        !old_val.is_valid() || old_val == val,
                        "Old value: {old_val:?}, New value: {val:?}, offset: {offset}"
                    );
                }
                tracing::trace!("Store {val:?} to offset {offset}");
                self.buf.set(ctx.offset_this_cycle(offset, 0), val)
            }
        }
    }
}

pub trait CycleContext {
    fn cycle(&self) -> usize;
    fn tot_cycles(&self) -> usize;

    fn offset_this_cycle(&self, offset: usize, back: usize) -> usize {
        let tot_cycles = self.tot_cycles();
        let cycle = (self.cycle() + tot_cycles - back) % tot_cycles;
        offset * tot_cycles + cycle
    }
}

/// This represents a layout bound to a buffer.
#[derive(PartialEq)]
pub struct BoundLayout<'a, L: 'static, E: Elem> {
    pub layout: &'static L,
    pub buf: BufferRow<'a, E>,
}

impl<'a, L: 'static, E: Elem> Copy for BoundLayout<'a, L, E> {}
impl<'a, L: 'static, E: Elem> Clone for BoundLayout<'a, L, E> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, L: risc0_zkp::layout::Component + 'static, E: Elem> Debug for BoundLayout<'a, L, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::result::Result<(), std::fmt::Error> {
        write!(f, "BoundLayout({}, {:?})", self.layout.ty_name(), self.buf)
    }
}

impl<'a, L, E: Elem> BoundLayout<'a, L, E> {
    pub fn new(layout: &'static L, buf: BufferRow<'a, E>) -> Self {
        Self { layout, buf }
    }

    pub fn map<NewL, F: FnOnce(&'static L) -> &'static NewL>(
        &self,
        f: F,
    ) -> BoundLayout<'a, NewL, E> {
        BoundLayout {
            layout: f(self.layout),
            buf: self.buf,
        }
    }

    pub fn layout(&self) -> &'static L {
        self.layout
    }
    pub fn buf(&self) -> BufferRow<E> {
        self.buf
    }
}

impl<'a, E: Elem> BoundLayout<'a, Reg, E> {
    pub fn load_ext<EE: ExtElem<SubElem = E>>(&self, ctx: &impl CycleContext, back: usize) -> EE {
        let subelems =
            (0..EE::EXT_SIZE).map(|idx| self.buf.load(ctx, self.layout.offset + idx, back));

        if subelems.clone().any(|elem: EE::SubElem| !elem.is_valid()) {
            EE::INVALID
        } else {
            EE::from_subelems(subelems)
        }
    }
    pub fn load_unchecked_ext<EE: ExtElem<SubElem = E>>(
        &self,
        ctx: &impl CycleContext,
        back: usize,
    ) -> EE {
        self.load_ext::<EE>(ctx, back).valid_or_zero()
    }
    pub fn store_ext<EE: ExtElem<SubElem = E>>(&self, ctx: &impl CycleContext, val: EE) {
        for (idx, elem) in val.subelems().into_iter().enumerate() {
            self.buf.store(ctx, self.layout.offset + idx, *elem)
        }
    }
}

impl<'a, E: Elem + Clone + Default + Copy> BoundLayout<'a, Reg, E> {
    pub fn load(&self, ctx: &impl CycleContext, back: usize) -> E {
        self.buf.load(ctx, self.layout.offset, back)
    }

    pub fn load_unchecked(&self, ctx: &impl CycleContext, back: usize) -> E {
        self.load(ctx, back).valid_or_zero()
    }

    pub fn store(&self, ctx: &impl CycleContext, val: E) {
        self.buf.store(ctx, self.layout.offset, val);
    }
}
