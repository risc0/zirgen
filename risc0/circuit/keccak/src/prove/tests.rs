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

use test_log::test;

use super::*;

fn test_inputs() -> Vec<KeccakState> {
    let mut state = KeccakState::default();
    let mut pows = 987654321_u64;
    for part in state.as_mut_slice() {
        *part = pows;
        pows = pows.wrapping_mul(123456789);
    }
    vec![state]
}

#[test]
fn test_keccak1() {
    let inputs = test_inputs();
    let po2 = 8; // 256
    let prover = keccak_prover().unwrap();
    let seal = prover.prove(inputs, po2).unwrap();
    prover.verify(&seal).expect("Verification failed");
}

#[test]
fn test_fwd_rev_ab() {
    cfg_if! {
        if #[cfg(feature = "cuda")] {
            use risc0_zkp::hal::cuda::CudaHalSha256;
            use crate::prove::hal::cuda::CudaCircuitHalSha256;
            let hal = Rc::new(CudaHalSha256::new());
            let circuit_hal = CudaCircuitHalSha256::new(hal.clone());
        // } else if #[cfg(any(all(target_os = "macos", target_arch = "aarch64"), target_os = "ios"))] {
        //     use risc0_zkp::hal::metal::MetalHalSha256;
        //     use crate::prove::hal::metal::MetalCircuitHal;
        //     let hal = Rc::new(MetalHalSha256::new());
        //     let circuit_hal = MetalCircuitHal::new(hal.clone());
        } else {
            use risc0_zkp::{core::hash::sha::Sha256HashSuite, hal::cpu::CpuHal};
            use crate::prove::hal::cpu::CpuCircuitHal;
            let suite = Sha256HashSuite::new_suite();
            let hal = Rc::new(CpuHal::new(suite));
            let circuit_hal = CpuCircuitHal;
        }
    }

    let inputs = test_inputs();
    let po2 = 8;
    let cycles: usize = 1 << po2;
    let preflight = PreflightTrace::new(inputs, cycles);

    let fwd_data = {
        let global = MetaBuffer::new("global", hal.as_ref(), 1, REGCOUNT_GLOBAL, true);
        let data = MetaBuffer::new("data", hal.as_ref(), cycles, REGCOUNT_DATA, true);
        circuit_hal
            .generate_witness(StepMode::SeqForward, &preflight, &global, &data)
            .unwrap();
        hal.eltwise_zeroize_elem(&data.buf);
        data.buf.to_vec()
    };

    let rev_data = {
        let global = MetaBuffer::new("global", hal.as_ref(), 1, REGCOUNT_GLOBAL, true);
        let data = MetaBuffer::new("data", hal.as_ref(), cycles, REGCOUNT_DATA, true);
        circuit_hal
            .scatter_preflight(&data, &preflight.scatter, &preflight.data)
            .unwrap();
        circuit_hal
            .generate_witness(StepMode::SeqReverse, &preflight, &global, &data)
            .unwrap();
        hal.eltwise_zeroize_elem(&data.buf);
        data.buf.to_vec()
    };

    assert_eq!(fwd_data, rev_data);
}
