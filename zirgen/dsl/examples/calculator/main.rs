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

//! Performs a simple test of 123 + 456 using the generated code for the calculator circuit.
mod cpu;

use anyhow::Result;
use clap::Parser;
use risc0_zkp::core::digest::Digest;
use risc0_zkp::core::hash::{poseidon::PoseidonHashSuite, HashSuite};
use risc0_zkp::hal::{cpu::CpuHal, Buffer, Hal};
use std::path::PathBuf;

zirgen_dsl::zirgen_inhibit_warnings! {
mod calc_circuit {
    zirgen_dsl::zirgen_preamble!(*b"calculator______");

    include!{"../../examples/calculator.rs.inc"}

    type ExecContext<'a> = zirgen_dsl::cpu::CpuBuffers<'a, Val, &'a super::cpu::CpuExecContext>;
}
}
use calc_circuit::{CircuitField, CircuitHal, ExtVal, Val};

pub const OP_ADD: usize = 0;
pub const OP_SUB: usize = 1;

fn prove<
    'a,
    H: Hal<Field = CircuitField, Elem = Val, ExtElem = ExtVal>,
    CH: CircuitHal<'a, H> + risc0_zkp::hal::CircuitHal<H>,
>(
    hal: &H,
    circuit_hal: &CH,
) -> Result<Vec<u32>> {
    // TODO: Make it so we can use a smaller PO2 for testing without
    // having everything break in mysterious ways.  We should at least
    // have a more informative error message than "seal fails
    // verifaciton".
    const PO2: usize = 5;
    const TOT_CYCLES: usize = 1 << PO2;

    let data = hal.alloc_elem("data", calc_circuit::REGCOUNT_DATA * TOT_CYCLES);
    let code = hal.alloc_elem("code", calc_circuit::REGCOUNT_CODE * TOT_CYCLES);
    let global = hal.alloc_elem("global", calc_circuit::REGCOUNT_GLOBAL);
    let accum = hal.alloc_elem("accum", calc_circuit::REGCOUNT_ACCUM * TOT_CYCLES);

    circuit_hal.step_exec(TOT_CYCLES, &data, &global)?;

    let mut prover = risc0_zkp::prove::Prover::new(hal, &*crate::calc_circuit::TAPS);
    prover.set_po2(PO2);

    global.view(|out_slice| prover.iop().write_field_elem_slice(out_slice));
    prover.iop().write_u32_slice(&[/*PO2=*/ PO2 as u32]);

    // ZKP library's verification requires that we have 3 hardcoded
    // buffers, numbered 0, 1, and 2, all of which are nonempty, have
    // a size of a multiple of the cycle count, and are committed in a
    // specific order.
    //
    // TODO: Remove this restriction and make buffer orders and such per-circuit.
    prover.commit_group(calc_circuit::REGISTER_GROUP_CODE, &code);
    prover.commit_group(calc_circuit::REGISTER_GROUP_DATA, &data);
    let mix: [Val; calc_circuit::REGCOUNT_MIX] =
        std::array::from_fn(|_| prover.iop().random_elem());
    let mix = hal.copy_from_elem("mix", mix.as_slice());
    circuit_hal.step_accum(TOT_CYCLES, &accum, &data, &global)?;
    prover.commit_group(calc_circuit::REGISTER_GROUP_ACCUM, &accum);
    let seal = prover.finalize(&[&mix, &global], circuit_hal);

    Ok(seal)
}

pub fn verify(seal: &[u32], hash_suite: &HashSuite<CircuitField>) -> Result<()> {
    // We don't have a `code' buffer to verify.
    let check_code_fn = |_: u32, _: &Digest| Ok(());

    Ok(risc0_zkp::verify::verify(
        &calc_circuit::CircuitDef,
        hash_suite,
        seal,
        check_code_fn,
    )?)
}

#[derive(Parser)]
struct Args {
    /// Filename in which to write an output seal.
    #[clap(long)]
    seal: PathBuf,
}

pub fn main() {
    env_logger::init();

    let args = Args::parse();
    let hash_suite = PoseidonHashSuite::new_suite();
    let hal = CpuHal::new(hash_suite.clone());
    let circuit_hal = cpu::CpuCircuitHal::new(OP_ADD, 456, 123);
    let seal = prove(&hal, &circuit_hal).unwrap();
    std::fs::write(args.seal, bytemuck::cast_slice(seal.as_slice())).expect("Writing seal failed");

    verify(seal.as_slice(), &hash_suite).expect("Verification failed");
}

#[cfg(test)]
mod tests {
    use super::*;
    use risc0_zkp::field::Elem;
    use test_log::test;

    fn run_test(op: usize) {
        let hash_suite = PoseidonHashSuite::new_suite();
        let hal = CpuHal::new(hash_suite.clone());
        let circuit_hal = cpu::CpuCircuitHal::new(op, 456, 123);
        let seal = prove(&hal, &circuit_hal).unwrap();

        verify(seal.as_slice(), &hash_suite).expect("Verification failed");

        let expected_result = match op {
            OP_ADD => Val::new(456 + 123),
            OP_SUB => Val::new(456 - 123),
            _ => panic!("Unexpected op"),
        };

        let seal_vals = Val::from_u32_slice(&seal[..calc_circuit::REGCOUNT_GLOBAL]);
        assert_eq!(
            seal_vals[calc_circuit::LAYOUT_GLOBAL.result._super.offset],
            expected_result
        );
    }

    #[test]
    fn test_cpu_add() {
        run_test(OP_ADD);
    }

    #[test]
    fn test_cpu_sub() {
        run_test(OP_SUB);
    }
}
