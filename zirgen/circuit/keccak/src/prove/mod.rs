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

mod cpp;
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub mod testutil;
#[cfg(not(feature = "make_control_ids"))]
pub mod zkr;

use std::{collections::VecDeque, rc::Rc};

use anyhow::Result;
use cfg_if::cfg_if;
use risc0_core::{field::Elem, scope};
use risc0_zkp::{
    adapter::{CircuitCoreDef, CircuitInfo, TapsProvider, PROOF_SYSTEM_INFO},
    core::{digest::Digest, hash::poseidon2::Poseidon2HashSuite},
    field::baby_bear::BabyBear,
    hal::{cpu::CpuHal, Buffer, Hal},
    taps::TapSet,
};

use self::cpu::CpuCircuitHal;
use super::{taps, CircuitImpl};
use crate::CIRCUIT;

risc0_zirgen_dsl::zirgen_inhibit_warnings! {

mod keccak_circuit {
    risc0_zirgen_dsl::zirgen_preamble!{}

    type ExecContext<'a> = super::cpu::CpuExecContext<'a>;

    include!{"../types.rs.inc"}
    include!{"../defs.rs.inc"}
    include!{"../layout.rs.inc"}
    include!{"../steps.rs.inc"}
}
}

pub use keccak_circuit::REGCOUNT_GLOBAL;
use keccak_circuit::{CircuitField, CircuitHal, ExtVal, Val};

const GLOBAL_MIX: usize = 0;
const GLOBAL_OUT: usize = 1;

pub type Seal = Vec<u32>;

pub trait KeccakProver {
    fn prove(&self, input: VecDeque<u32>, po2: usize) -> Result<Seal>;

    fn verify(&self, seal: &Seal) -> Result<()> {
        let hash_suite = Poseidon2HashSuite::new_suite();

        // We don't have a `code' buffer to verify.
        let check_code_fn = |_: u32, _: &Digest| Ok(());

        Ok(risc0_zkp::verify::verify(
            &CIRCUIT,
            &hash_suite,
            seal,
            check_code_fn,
        )?)
    }
}

pub fn keccak_prover() -> Result<Box<dyn KeccakProver>> {
    cfg_if! {
        if #[cfg(feature = "cuda")] {
            self::cuda::keccak_prover()
        // } else if #[cfg(any(all(target_os = "macos", target_arch = "aarch64"), target_os = "ios"))] {
        //     self::metal::keccak_prover()
        } else {
            self::cpu::keccak_prover()
        }
    }
}

pub(crate) struct KeccakProverImpl<H, C>
where
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal>,
    C: risc0_zkp::hal::CircuitHal<H>,
{
    hal: Rc<H>,
    circuit_hal: Rc<C>,
}

impl<H, C> KeccakProver for KeccakProverImpl<H, C>
where
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal>,
    C: risc0_zkp::hal::CircuitHal<H>,
{
    fn prove(&self, input: VecDeque<u32>, po2: usize) -> Result<Seal> {
        scope!("prove");

        let tot_cycles: usize = 1 << po2;

        let cpu_hal = CpuHal::new(self.hal.get_hash_suite().clone());
        let cpu_circuit_hal = CpuCircuitHal::new(input);

        let alloc_elem = |name, size| {
            if cfg!(debug_assertions) {
                cpu_hal.copy_from_elem(name, vec![H::Elem::INVALID; size].as_slice())
            } else {
                cpu_hal.alloc_elem(name, size)
            }
        };

        let cpu_data = scope!(
            "alloc(data)",
            alloc_elem("data", keccak_circuit::REGCOUNT_DATA * tot_cycles)
        );
        let cpu_code = scope!(
            "alloc(code)",
            alloc_elem("code", keccak_circuit::REGCOUNT_CODE * tot_cycles)
        );
        let cpu_global = scope!(
            "alloc(global)",
            alloc_elem("global", keccak_circuit::REGCOUNT_GLOBAL)
        );
        let cpu_accum = scope!(
            "alloc(accum)",
            alloc_elem("accum", keccak_circuit::REGCOUNT_ACCUM * tot_cycles)
        );

        cpu_circuit_hal.step_exec(tot_cycles, &cpu_data, &cpu_global)?;
        scope!("zeroize(code)", cpu_hal.eltwise_zeroize_elem(&cpu_code));
        scope!("zeroize(data)", cpu_hal.eltwise_zeroize_elem(&cpu_data));
        scope!("zeroize(global)", cpu_hal.eltwise_zeroize_elem(&cpu_global));

        let mut prover = risc0_zkp::prove::Prover::new(self.hal.as_ref(), &*crate::taps::TAPSET);

        // At the start of the protocol, seed the Fiat-Shamir transcript with context information
        // about the proof system and circuit.
        let hashfn = &self.hal.get_hash_suite().hashfn;
        prover
            .iop()
            .commit(&hashfn.hash_elem_slice(&PROOF_SYSTEM_INFO.encode()));
        prover
            .iop()
            .commit(&hashfn.hash_elem_slice(&CircuitImpl::CIRCUIT_INFO.encode()));
        prover.set_po2(po2);

        // Concat io (i.e. globals) and po2 into a vector.
        cpu_global.view(|out_slice| {
            let vec: Vec<_> = out_slice
                .iter()
                .chain(Elem::from_u32_slice(&[po2 as u32]))
                .copied()
                .collect();

            prover
                .iop()
                .commit(&self.hal.get_hash_suite().hashfn.hash_elem_slice(&vec));
            prover.iop().write_field_elem_slice(vec.as_slice());
        });

        let hal_code = self.hal.copy_from_elem("code", &cpu_code.as_slice());
        let hal_data = self.hal.copy_from_elem("data", &cpu_data.as_slice());

        prover.commit_group(keccak_circuit::REGISTER_GROUP_CODE, &hal_code);
        prover.commit_group(keccak_circuit::REGISTER_GROUP_DATA, &hal_data);

        let mix: [Val; keccak_circuit::REGCOUNT_MIX] =
            std::array::from_fn(|_| prover.iop().random_elem());
        let cpu_mix = cpu_hal.copy_from_elem("mix", mix.as_slice());
        cpu_circuit_hal.step_accum(tot_cycles, &cpu_accum, &cpu_data, &cpu_mix)?;

        let hal_accum = self.hal.copy_from_elem("accum", &cpu_accum.as_slice());
        scope!("zeroize(accum)", self.hal.eltwise_zeroize_elem(&hal_accum));
        prover.commit_group(keccak_circuit::REGISTER_GROUP_ACCUM, &hal_accum);

        let hal_mix = self.hal.copy_from_elem("mix", &cpu_mix.as_slice());

        let hal_global = self.hal.copy_from_elem("global", &cpu_global.as_slice());

        let seal = prover.finalize(&[&hal_mix, &hal_global], self.circuit_hal.as_ref());

        Ok(seal)
    }
}

impl CircuitCoreDef<BabyBear> for CircuitImpl {}

impl TapsProvider for CircuitImpl {
    fn get_taps(&self) -> &'static TapSet<'static> {
        self::taps::TAPSET
    }
}
