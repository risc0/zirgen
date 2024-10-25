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

pub mod cpu;

mod buffers;
pub use buffers::{BufferSpec, Buffers};

/// This trait represents the set of registers available for a
/// specific cycle.  For instance, this can either be a row in the
/// execution trace or a set of global registers that are common to
/// all cycles.
pub trait BufferRow: Clone {
    type ValType: Clone + Copy + Default;

    fn load(&self, offset: usize, back: usize) -> Self::ValType;
    fn store(&self, offset: usize, val: Self::ValType);
}

impl<T: Clone + Copy + Default> BufferRow for &[T] {
    type ValType = T;

    fn load(&self, offset: usize, back: usize) -> T {
        assert_eq!(back, 0, "Unexpected back when accessing a flat buffer");
        self[offset]
    }
    fn store(&self, _offset: usize, _val: Self::ValType) {
        panic!("Attempt to write to read-only buffer")
    }
}

/// This represents a layout bound to a buffer.
pub struct BoundLayout<'a, L: 'static, B: BufferRow> {
    pub layout: &'static L,
    pub buf: &'a B,
}

// We can't derive these since derive(Copy) also puts bounds on BufferRow.
impl<'a, L, B: BufferRow> Clone for BoundLayout<'a, L, B> {
    fn clone(&self) -> Self {
        BoundLayout {
            buf: self.buf,
            layout: self.layout,
        }
    }
}
impl<'a, L, B: BufferRow> Copy for BoundLayout<'a, L, B> {}

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
