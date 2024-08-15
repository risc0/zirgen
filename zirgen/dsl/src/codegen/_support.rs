// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

// TODO: First pass, lots of cleanup to be done here!

use std::array::from_fn;

use crate::{BoundLayout, BufferRow};
use anyhow::Result;
use risc0_core::field::{Elem, ExtElem, Field};
use risc0_zkp::adapter::MixState;

pub type Index = usize;
pub type TapGroupName = &'static str;

pub fn trivial_constraint<EE: ExtElem>() -> Result<MixState<EE>> {
    Ok(MixState {
        tot: EE::ZERO,
        mul: EE::ONE,
    })
}

pub fn and_cond_generic<F: Field, C: core::fmt::Debug>(
    x: MixState<F::ExtElem>,
    cond: C,
    inner: MixState<F::ExtElem>,
) -> Result<MixState<F::ExtElem>>
where
    F::ExtElem:
        core::ops::Mul<C, Output = F::ExtElem> + core::ops::Mul<F::Elem, Output = F::ExtElem>,
{
    Ok(MixState {
        tot: x.tot + inner.tot * x.mul * cond,
        mul: x.mul * inner.mul,
    })
}

pub fn and_eqz_generic<F: Field, V: core::fmt::Debug>(
    poly_mix: F::ExtElem,
    x: MixState<F::ExtElem>,
    val: V,
) -> Result<MixState<F::ExtElem>>
where
    F::ExtElem:
        core::ops::Mul<V, Output = F::ExtElem> + core::ops::Mul<F::Elem, Output = F::ExtElem>,
{
    Ok(MixState {
        tot: x.tot + x.mul * val,
        mul: x.mul * poly_mix,
    })
}

pub fn layout_map<B: BufferRow, Layout, NewLayout>(
    layout: BoundLayout<Layout, B>,
    f: impl Fn(&'static Layout) -> &'static NewLayout,
) -> BoundLayout<NewLayout, B>
where
{
    layout.map(f)
}

pub fn eqz(v: impl Elem) -> Result<()> {
    if v.to_u32_words().into_iter().all(|v| v == 0) {
        Ok(())
    } else {
        anyhow::bail!("Eqz failed: {:?}", v)
    }
}

pub fn isz<E: Elem>(v: E) -> Result<E> {
    if v == E::ZERO {
        Ok(E::ONE)
    } else {
        Ok(E::ZERO)
    }
}

pub fn map<T, U, F, const N: usize>(arr: [T; N], f: F) -> Result<[U; N]>
where
    T: Copy,
    F: FnMut(T) -> Result<U>,
{
    // Unfortunately, we have to convert from an Array to an Iterator to collect
    // the sequence of results into a result of a sequence. When we convert back,
    // the compiler doesn't know that the size always matches, so we are forced
    // to unwrap with an unreachable error handler.
    // Once it is stable, this function can be replaced with arr.try_map(f).
    arr.map(f)
        .into_iter()
        .collect::<Result<Vec<U>>>()
        .map(|mapped_arr| {
            mapped_arr
                .try_into()
                .unwrap_or_else(|_| panic!("unreachable"))
        })
}

pub fn map_layout<T, Layout, U, F, RowType, const N: usize>(
    arr: [T; N],
    layouts: BoundLayout<[&'static Layout; N], RowType>,
    mut f: F,
) -> Result<[U; N]>
where
    T: Copy,
    F: FnMut(T, BoundLayout<Layout, RowType>) -> Result<U>,
    RowType: BufferRow,
{
    // Unfortunately, we have to convert from an Array to an Iterator to collect
    // the sequence of results into a result of a sequence. When we convert back,
    // the compiler doesn't know that the size always matches, so we are forced
    // to unwrap with an unreachable error handler.
    // Once it is stable, this function's implementation can be replaced with:
    // try_from_fn(|i| f(&arr[i], &layouts[i]))
    let mapped: [Result<U>; N] = from_fn(|i| f(arr[i], layouts.map(|layout| layout[i])));
    mapped
        .into_iter()
        .collect::<Result<Vec<U>>>()
        .map(|mapped_arr| {
            mapped_arr
                .try_into()
                .unwrap_or_else(|_| panic!("unreachable"))
        })
}

pub fn reduce<T, U, F, const N: usize>(arr: [T; N], init: U, mut f: F) -> Result<U>
where
    T: Copy,
    F: FnMut(U, T) -> Result<U>,
{
    let mut output = init;
    for element in arr {
        output = f(output, element)?;
    }
    Ok(output)
}

pub fn reduce_layout<T, Layout, U, F, RowType, const N: usize>(
    arr: [T; N],
    init: U,
    layouts: BoundLayout<[&'static Layout; N], RowType>,
    mut f: F,
) -> Result<U>
where
    T: Copy,
    F: FnMut(U, T, BoundLayout<Layout, RowType>) -> Result<U>,
    RowType: BufferRow,
{
    let mut output = init;
    for (element, layout) in arr.iter().zip(layouts.layout().iter()) {
        let bound_layout = BoundLayout {
            layout: *layout,
            buf: layouts.buf(),
        };
        output = f(output, *element, bound_layout)?;
    }
    Ok(output)
}

pub fn is_nonzero(val: impl Elem) -> bool {
    val.to_u32_words().into_iter().any(|v| v != 0)
}

pub fn inv<E: Elem>(v: E) -> Result<E> {
    Ok(v.inv())
}

pub fn neg<E: Elem>(v: E) -> Result<E> {
    Ok(E::ZERO - v)
}

pub fn bit_and<E: Elem + Into<u32> + From<u32>>(lhs: E, rhs: E) -> Result<E> {
    let lhs = lhs.into();
    let rhs = rhs.into();
    Ok(E::from(lhs & rhs))
}

pub fn in_range<E: Elem + Into<u32>>(low: E, mid: E, high: E) -> Result<E> {
    let low = low.into();
    let mid = mid.into();
    let high = high.into();
    assert!(low <= high);
    if low <= mid && mid < high {
        Ok(E::ONE)
    } else {
        Ok(E::ZERO)
    }
}

// Locally import all the codegen_* macros without the codegen_ prefix.
// TODO: When macros 2.0 arrives we should be able to simplify this.
macro_rules! import_macros {
    ($($name:ident),*) => {
        paste::paste! {
            $( pub use crate::[< codegen_ $name >] as $name; )*
        }
    }
}

import_macros! {
    make_ref, make_tap, invoke_extern, make_val, make_val_ext,
    set_field, load, load_as_ext, load_ext, store, store_ext, define_tap_buffer,
    define_global_buffer, define_buffer, define_buffer_list, bind_layout,
    layout_subscript, layout_lookup, eqz, get_buffer
}
