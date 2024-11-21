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

use hex_literal::hex;
use test_log::test;

use super::*;

fn run_test(input: &[u8], po2: usize) {
    let input: Vec<u32> = input
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let prover = keccak_prover().unwrap();
    let seal = prover.prove(input.into(), po2).unwrap();
    prover.verify(&seal).expect("Verification failed");
}

#[test]
fn test_keccak1() {
    run_test(
        &hex!("010000000000000054686520717569636B2062726F776E20666F78206A756D7073206F76657220746865206C617A7920646F672E0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000080578951e24efd62a3d63a86f7cd19aaa53c898fe287d2552133220370240b572d0000000000000000"),
        /*po2=*/12,
    );
}

#[test]
fn test_keccak2() {
    run_test(
        &hex!("0600000000000000436F6D6D616E64657220526F64657269636B20426C61696E65206C6F6F6B6564206672616E746963616C6C792061726F756E6420746865206272696467652E20776865726520686973206F66666963657273207765726520646972656374696E6720726570616972732077697468206C6F7720616E6420757267656E7420766F696365732C2073757267656F6E7320617373697374696E67206174206120646966666963756C74206F7065726174696F6E2E20546865206772617920737465656C20636F6D706172746D656E7420776173206120636F6E667573696F6E206F6620616374697669746965732C2065616368206F726465726C7920627920697473656C662062757420746865206F766572616C6C20696D7072657373696F6E20776173206F66206368616F732E2053637265656E732061626F7665206F6E652068656C6D736D616E27732073746174696F6E2073686F7765642074686520706C616E65742062656C6F7720616E6420746865206F746865722C20736869707320696E206F72626974206E656172204D61634172746875722C20627574206576657279776865726520656C7365207468652070616E656C20636F7665727320686164206265656E2072656D6F7665642066726F6D20636F6E736F6C65732C207465737420696E737472756D656E7473207765726520636C697070656420696E746F20746865697220696E73696465732C20616E6420746563686E696369616E732073746F6F64206279207769746820636F6C6F722D636F64656420656C656374726F6E696320617373656D626C69657320746F207265706C6163652065766572797468696E672074686174207365656D656420646F75627466756C2E205468756D707320616E64207768696E657320736F756E646564207468726F75676820746865207368697020383920736F6D657768657265206166742074686520656E67696E656572696E67206372657720776F726B6564206F6E207468652068756C6C2E01000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008028c3f5c69c21be780e5508d355ebf7d5e060f203ca8717447b71cb44544df5c70200000000000000546865736520776F7264732077657265207574746572656420696E204A756C79203138303520627920416E6E61205061766C6F766E6120536368657265722C20612064697374696E67756973686564206C616479206F662074686520636F7572742C20616E6420636F6E666964656E7469616C206D6169642D6F662D686F6E6F757220746F2074686520456D7072657373204D617279612046796F646F726F766E612E2049742077617320686572206772656574696E6720746F205072696E63652056617373696C792C2061206D616E206869676820696E2072616E6B20616E64206F66666963652C2077686F207761732074686520666972737420746F2061727269766520617401000000000000804bdc1874a3125f1f911fe8c76ac8443a6ec623ef91bc58eabf54c5762097894d0000000000000000"),
        /*po2=*/15,
    );
}