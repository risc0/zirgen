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

/// Support for generated code.  Not intended for public use; only
/// marked "pub" so that it's accessible by macros defined here.
#[doc(hidden)]
pub mod _support;

/// Support for generated code.  Not intended for public use; only
/// marked "pub" so that it's accessible by macros defined here.
#[doc(hidden)]
#[macro_use]
pub mod _support_macros;

pub mod taps;

// Re-export macros we depend on so that users don't have to add these
// crates to their direct dependencies.
pub use paste::paste;
pub use strum::IntoStaticStr;

/// Inhibits warnings triggered by generated code.
/// Usage:
///  zirgen_inhibit_warnings!{
///    pub mod my_circuit_module_name {
///      ...
///    }
///  }
#[macro_export]
macro_rules! zirgen_inhibit_warnings {
    {$($tt:tt)*} => {
        #[allow(unused_parens)]
        #[allow(unused_assignments)]
        #[allow(unused_variables)]
        #[allow(dead_code)]
        #[allow(non_camel_case_types)]
        #[allow(non_snake_case)]
        $($tt)*
    };
}

/// Usage:
///  pub mod my_circuit_module_name {
///    zirgen_preamble!(b"example_name____");
///    include!{"my_generated_circuit.rs.inc"};
///    struct ExecContext...;
///    struct ValidityRegsContext...;
///    struct ValidityTapsContext...;
#[macro_export]
macro_rules! zirgen_preamble {
    ($protocol_info:expr) => {
        use anyhow::{bail, Context, Result};
        use risc0_zkp::adapter::ProtocolInfo;
        use $crate::codegen::_support::*;
        use $crate::codegen::taps::{make_taps, Tap};
        use $crate::{BoundLayout, BufferRow, BufferSpec, Buffers};

        lazy_static::lazy_static! {
            pub static ref TAPS : risc0_zkp::taps::TapSet<'static> = make_taps(TAP_LIST.as_slice(),
                                                                               TAP_GROUP_NAMES);
        }

        // Explicitly instantiate calls that cause rustc to be very slow
        // when processing large generated code.
        fn and_cond(x: MixState, cond: Val, inner: MixState) -> Result<MixState> {
            and_cond_generic::<CircuitField, Val>(x, cond, inner)
        }
        fn and_cond_ext(x: MixState, cond: ExtVal, inner: MixState) -> Result<MixState> {
            and_cond_generic::<CircuitField, ExtVal>(x, cond, inner)
        }
        fn and_eqz(poly_mix: ExtVal, x: MixState, val: Val) -> Result<MixState> {
            and_eqz_generic::<CircuitField, Val>(poly_mix, x, val)
        }
        fn and_eqz_ext(poly_mix: ExtVal, x: MixState, val: ExtVal) -> Result<MixState> {
            and_eqz_generic::<CircuitField, ExtVal>(poly_mix, x, val)
        }
        fn load(buf: BoundLayout<Reg, impl BufferRow<ValType = Val>>, back: usize) -> Val {
            buf.buf().load(buf.layout().offset, back)
        }
        fn load_ext(buf: BoundLayout<Reg, impl BufferRow<ValType = Val>>, back: usize) -> ExtVal {
            ExtVal::new(
                buf.buf().load(buf.layout().offset + 0, back),
                buf.buf().load(buf.layout().offset + 1, back),
                buf.buf().load(buf.layout().offset + 2, back),
                buf.buf().load(buf.layout().offset + 3, back),
            )
        }
        fn store(buf: BoundLayout<Reg, impl BufferRow<ValType = Val>>, val: Val) {
            buf.buf().store(buf.layout().offset, val)
        }
        fn store_ext(buf: BoundLayout<Reg, impl BufferRow<ValType = Val>>, val: ExtVal) {
            for (i, coef) in val.elems().iter().enumerate() {
                buf.buf().store(buf.layout().offset + i, *coef);
            }
        }
        fn alias_layout<Layout: PartialEq, B: BufferRow>(
            x: BoundLayout<Layout, B>,
            y: BoundLayout<Layout, B>,
        ) -> Result<()> {
            if x == y {
                Ok(())
            } else {
                bail!("layouts did not match at runtime!")
            }
        }

        // risc0_zkp-compatible CircuitDef
        pub struct CircuitDef;
        type ValidityRegsContext<'a> = $crate::cpu::CpuBuffers<'a, Val, ()>;
        type ValidityTapsContext<'a> = $crate::Buffers<(), &'a [Val], ()>;
        impl risc0_zkp::adapter::CircuitInfo for CircuitDef {
            const CIRCUIT_INFO: ProtocolInfo = ProtocolInfo($protocol_info);
            const OUTPUT_SIZE: usize = REGCOUNT_GLOBAL;
            const MIX_SIZE: usize = REGCOUNT_MIX;
        }
        impl risc0_zkp::adapter::PolyExt<CircuitField> for CircuitDef {
            fn poly_ext(&self, mix: &ExtVal, u: &[ExtVal], args: &[&[Val]]) -> MixState {
                use risc0_zkp::field::Elem;

                let raw_buffers = get_named_buffers($crate::poly_ext_named_buffers(args));
                let buffers =
                    raw_buffers.map_rows(|x| -> () { panic!("Unexpected tap in poly_ext") });
                assert_eq!(u.len(), TAP_LIST.len());

                let res = validity_taps_(&buffers, *mix, u.try_into().unwrap()).unwrap();

                res
            }
        }

        impl risc0_zkp::adapter::TapsProvider for CircuitDef {
            fn get_taps(&self) -> &'static risc0_zkp::taps::TapSet<'static> {
                &*TAPS
            }
        }

        impl risc0_zkp::adapter::CircuitCoreDef<CircuitField> for CircuitDef {}

        #[derive(Debug, Copy, Clone, PartialEq)]
        pub struct Reg {
            pub offset: usize,
        }

        impl risc0_zkp::layout::Component for Reg {
            fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {
                v.visit_reg("Reg", self.offset)
            }
            fn ty_name(&self) -> &'static str {
                "reg"
            }
        }

        // Eventually we want to generate this trait based on what functions are available,
        // but for now we can hardcode it.
        pub trait CircuitHal<'a, H: risc0_zkp::hal::Hal<Elem = Val>> {
            fn step_exec(
                &self,
                tot_cycles: usize,
                data: &H::Buffer<H::Elem>,
                global: &H::Buffer<H::Elem>,
            ) -> Result<()>;

            fn step_accum(
                &self,
                tot_cycles: usize,
                accum: &H::Buffer<H::Elem>,
                data: &H::Buffer<H::Elem>,
                global: &H::Buffer<H::Elem>,
            ) -> Result<()>;
        }
    };
}

/// Default implementation of "log" for circuits.  Circuits can take advantage
/// of this implementation by supplying a forwarding function such as this on
/// a context object:
///    pub fn log(& self, message: &str, x: &[Val]) -> Result<()> {
///        default_log(message, x)
///    }
pub fn default_log<E: risc0_zkp::field::Elem + Into<u32>>(
    message: &str,
    x: &[E],
) -> anyhow::Result<()> {
    let mut arg_num = 0;
    let mut next_arg = || {
        if arg_num >= x.len() {
            panic!("Ran out of arguments in log");
        }
        let result = x[arg_num];
        arg_num += 1;
        result
    };

    let mut bytes = message.bytes();
    while let Some(mut p) = bytes.next() {
        let mut next_char = || -> u8 { bytes.next().unwrap() };
        if p == b'%' {
            p = next_char();
            let mut _len = 0;
            while p >= b'0' && p <= b'9' {
                _len *= 10;
                _len += p - b'0';
                p = next_char();
            }
            if p == b'%' {
                print!("%");
            } else if p == b'x' {
                print!("{:x}", next_arg().into()); // TODO: use len?
            } else if p == b'u' {
                print!("{}", next_arg().into()); // TODO: use len?
            } else if p == b'w' || p == b'e' {
                let mut vals: [u64; 4] = [0; 4];
                let mut is_u32 = true;
                let mut u32val: u32 = 0;
                for i in 0..4 {
                    vals[i] = next_arg().into() as u64;
                    if vals[i] > 0xff {
                        is_u32 = false;
                    } else {
                        u32val >>= 8;
                        u32val |= (vals[i] as u32) << 24;
                    }
                }
                if p == b'e' {
                    is_u32 = false;
                }

                if is_u32 {
                    print!("{:x}", u32val);
                } else {
                    print!("[");
                    for i in 0..4 {
                        if i != 0 {
                            print!(", ");
                        }
                        print!("{}", vals[i]);
                    }
                    print!("]");
                }
            }
        } else {
            print!("{}", p as char);
        }
    }
    print!("\n");
    Ok(())
}
