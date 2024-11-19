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

use derive_more::Debug;
use risc0_core::field::baby_bear::BabyBearElem;

#[derive(Clone, Debug, PartialEq)]
#[repr(C)]
pub struct RawMemoryTransaction {
    #[debug("{addr:#010x}")]
    pub addr: u32,
    pub cycle: u32,
    #[debug("{word:#010x}")]
    pub word: u32,
    pub prev_cycle: u32,
    #[debug("{word:#010x}")]
    pub prev_word: u32,
}

#[derive(Clone, Debug, PartialEq)]
#[repr(C)]
pub struct RawPreflightCycle {
    pub state: u32,
    #[debug("{pc:#010x}")]
    pub pc: u32,
    pub major: u8,
    pub minor: u8,
    pub machine_mode: u8,
    #[debug(skip)]
    pub padding: u8,
    pub user_cycle: u32,
    pub txn_idx: u32,
    pub extra_idx: u32,
    pub diff_count: u32,
}

#[repr(C)]
pub struct RawPreflightTrace {
    pub cycles: *const RawPreflightCycle,
    pub txns: *const RawMemoryTransaction,
    pub extras: *const u32,
    pub table_split_cycle: u32,
}

#[repr(C)]
pub struct RawBuffer {
    pub buf: *const BabyBearElem,
    pub rows: usize,
    pub cols: usize,
    pub checked_reads: bool,
}

#[repr(C)]
pub struct RawExecutionTrace {
    pub global: RawBuffer,
    pub data: RawBuffer,
}

extern "C" {
    pub fn risc0_circuit_rv32im_v2_cpu_witgen(
        exec_trace: *const RawExecutionTrace,
        preflight: *const RawPreflightTrace,
        cycles: u32,
    ) -> *const std::os::raw::c_char;
}
