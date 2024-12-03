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

use std::process::Command;

use anyhow::Result;
use risc0_circuit_keccak::{prove::zkr::get_zkr_u32s, KECCAK_PO2, RECURSION_PO2};
use risc0_circuit_recursion::prove::Program;
use risc0_zkp::core::{digest::Digest, hash::poseidon2::Poseidon2HashSuite};
use risc0_zkvm::recursion::MerkleGroup;

const CONTROL_ID_PATH: &str = "risc0/circuit/keccak/src/control_id.rs";

fn compute_control_id(po2: usize) -> Result<Digest> {
    let encoded_program = get_zkr_u32s(&format!("keccak_lift_{}.zkr", po2))?;
    let program = Program::from_encoded(&encoded_program, RECURSION_PO2);
    let hash_suite = Poseidon2HashSuite::new_suite();
    Ok(program.compute_control_id(hash_suite))
}

fn compute_control_root(control_id: Digest) -> Result<Digest> {
    let hash_suite = Poseidon2HashSuite::new_suite();
    let hashfn = hash_suite.hashfn.as_ref();
    let group = MerkleGroup::new(vec![control_id])?;
    Ok(group.calc_root(hashfn))
}

pub fn main() {
    let control_id = compute_control_id(KECCAK_PO2).unwrap();
    let control_root = compute_control_root(control_id).unwrap();
    let contents = format!(
        include_str!("templates/control_id.rs"),
        control_id, control_root
    );
    std::fs::write(CONTROL_ID_PATH, contents).unwrap();

    // Use rustfmt to format the file.
    Command::new("rustfmt")
        .arg(CONTROL_ID_PATH)
        .status()
        .expect("failed to format {CONTROL_ID_PATH}");
}
