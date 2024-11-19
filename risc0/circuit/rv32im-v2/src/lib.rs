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

#[cfg(feature = "prove")]
mod prove;
#[cfg(test)]
mod riscv_tests;
mod zirgen;

pub struct CircuitImpl;

pub const CIRCUIT: CircuitImpl = CircuitImpl::new();

// use anyhow::Result;
// use micro_circuit::{CircuitField, ExtVal, Val};
// use risc0_zkp::{
//     core::digest::Digest,
//     core::hash::HashSuite,
//     hal::{Buffer, Hal},
// };

// zirgen_dsl::zirgen_inhibit_warnings! {

// pub mod micro_circuit {
//     zirgen_dsl::zirgen_preamble!(*b"micro_circuit___");

//     include!{"../dsl/micro.rs.inc"}

//     type ExecContext<'a> = zirgen_dsl::cpu::CpuBuffers<'a, Val, &'a super::cpu::CpuExecContext<'a>>;
// }

// }

// pub fn prove<
//     'a,
//     H: Hal + 'a,
//     MH: micro_circuit::CircuitHal<'a, H> + risc0_zkp::hal::CircuitHal<H>,
// >(
//     hal: &H,
//     micro_hal: &MH,
//     segment: &Segment,
// ) -> Result<Vec<u32>>
// where
//     H: Hal<Elem = Val, ExtElem = ExtVal>,
// {
//     let tot_cycles = 1 << segment.po2();

//     let data = hal.alloc_elem("data", micro_circuit::REGCOUNT_DATA * tot_cycles);
//     let code = hal.alloc_elem("code", micro_circuit::REGCOUNT_CODE * tot_cycles);
//     let global = hal.alloc_elem("global", micro_circuit::REGCOUNT_GLOBAL);
//     let accum = hal.alloc_elem("accum", micro_circuit::REGCOUNT_ACCUM * tot_cycles);

//     micro_hal.step_exec(tot_cycles, &data, &global)?;

//     let mut prover = risc0_zkp::prove::Prover::new(hal, &*crate::micro_circuit::TAPS);
//     prover.set_po2(segment.po2() as usize);

//     global.view(|out_slice| prover.iop().write_field_elem_slice(out_slice));
//     prover.iop().write_u32_slice(&[segment.po2() as u32]);

//     // ZKP library's verification requires that we have 3 hardcoded
//     // buffers, numbered 0, 1, and 2, all of which are nonempty, have
//     // a size of a multiple of the cycle count, and are committed in a
//     // specific order.
//     //
//     // TODO: Remove this restriction and make buffer orders and such per-circuit.
//     prover.commit_group(micro_circuit::REGISTER_GROUP_CODE, &code);
//     prover.commit_group(micro_circuit::REGISTER_GROUP_DATA, &data);
//     let mix: [H::Elem; micro_circuit::REGCOUNT_MIX] =
//         std::array::from_fn(|_| prover.iop().random_elem());
//     let mix = hal.copy_from_elem("mix", mix.as_slice());
//     micro_hal.step_accum(tot_cycles, &accum, &data, &global)?;
//     prover.commit_group(micro_circuit::REGISTER_GROUP_ACCUM, &accum);
//     let seal = prover.finalize(&[&mix, &global], micro_hal);

//     Ok(seal)
// }

// pub fn verify(seal: &[u32], hash_suite: &HashSuite<CircuitField>) -> Result<()> {
//     // We don't have a `code' buffer to verify.
//     let check_code_fn = |_: u32, _: &Digest| Ok(());

//     Ok(risc0_zkp::verify::verify(
//         &micro_circuit::CircuitDef,
//         hash_suite,
//         seal,
//         check_code_fn,
//     )?)
// }
