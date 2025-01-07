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

#pragma once

#include <cstdint>

// Platform specific constants

namespace zirgen::rv32im_v2 {

constexpr uint32_t ZERO_PAGE_START_ADDR = 0x0;
constexpr uint32_t ZERO_PAGE_START_WORD = ZERO_PAGE_START_ADDR / 4;
constexpr uint32_t ZERO_PAGE_END_ADDR = 0x10000;
constexpr uint32_t ZERO_PAGE_END_WORD = ZERO_PAGE_END_ADDR / 4;
constexpr uint32_t USER_START_ADDR = 0x10000;
constexpr uint32_t USER_START_WORD = USER_START_ADDR / 4;
constexpr uint32_t USER_END_ADDR = 0xc0000000;
constexpr uint32_t USER_END_WORD = USER_END_ADDR / 4;
constexpr uint32_t KERNEL_START_ADDR = 0xc0000000;
constexpr uint32_t KERNEL_START_WORD = KERNEL_START_ADDR / 4;
constexpr uint32_t KERNEL_END_ADDR = 0xff000000;
constexpr uint32_t KERNEL_END_WORD = KERNEL_END_ADDR / 4;

constexpr uint32_t MACHINE_REGS_ADDR = 0xffff0000;
constexpr uint32_t MACHINE_REGS_WORD = MACHINE_REGS_ADDR / 4;
constexpr uint32_t USER_REGS_ADDR = 0xffff0080;
constexpr uint32_t USER_REGS_WORD = USER_REGS_ADDR / 4;
constexpr uint32_t MEPC_ADDR = 0xffff0200;
constexpr uint32_t MEPC_WORD = MEPC_ADDR / 4;
constexpr uint32_t SUSPEND_PC_ADDR = 0xffff0210;
constexpr uint32_t SUSPEND_PC_WORD = SUSPEND_PC_ADDR / 4;
constexpr uint32_t SUSPEND_MODE_ADDR = 0xffff0214;
constexpr uint32_t SUSPEND_MODE_WORD = SUSPEND_MODE_ADDR / 4;
constexpr uint32_t SUSPEND_CYCLE_LOW_ADDR = 0xffff0218;
constexpr uint32_t SUSPEND_CYCLE_LOW_WORD = SUSPEND_CYCLE_LOW_ADDR / 4;
constexpr uint32_t SUSPEND_CYCLE_HIGH_ADDR = 0xffff021c;
constexpr uint32_t SUSPEND_CYCLE_HIGH_WORD = SUSPEND_CYCLE_HIGH_ADDR / 4;
constexpr uint32_t OUTPUT_ADDR = 0xffff0240;
constexpr uint32_t OUTPUT_WORD = OUTPUT_ADDR / 4;
constexpr uint32_t INPUT_ADDR = 0xffff0260;
constexpr uint32_t INPUT_WORD = INPUT_ADDR / 4;

constexpr uint32_t ECALL_DISPATCH_ADDR = 0xffff1000;
constexpr uint32_t ECALL_DISPATCH_WORD = ECALL_DISPATCH_ADDR / 4;
constexpr uint32_t TRAP_DISPATCH_ADDR = 0xffff2000;
constexpr uint32_t TRAP_DISPATCH_WORD = TRAP_DISPATCH_ADDR / 4;

constexpr uint32_t REG_ZERO = 0;
constexpr uint32_t REG_RA = 1;
constexpr uint32_t REG_SP = 2;
constexpr uint32_t REG_GP = 3;
constexpr uint32_t REG_TP = 4;
constexpr uint32_t REG_T0 = 5;
constexpr uint32_t REG_T1 = 6;
constexpr uint32_t REG_T2 = 7;
constexpr uint32_t REG_S0 = 8;
constexpr uint32_t REG_FP = 8;
constexpr uint32_t REG_S1 = 9;
constexpr uint32_t REG_A0 = 10;
constexpr uint32_t REG_A1 = 11;
constexpr uint32_t REG_A2 = 12;
constexpr uint32_t REG_A3 = 13;
constexpr uint32_t REG_A4 = 14;
constexpr uint32_t REG_A5 = 15;
constexpr uint32_t REG_A6 = 16;
constexpr uint32_t REG_A7 = 17;
constexpr uint32_t REG_S2 = 18;
constexpr uint32_t REG_S3 = 19;
constexpr uint32_t REG_S4 = 20;
constexpr uint32_t REG_S5 = 21;
constexpr uint32_t REG_S6 = 22;
constexpr uint32_t REG_S7 = 23;
constexpr uint32_t REG_S8 = 24;
constexpr uint32_t REG_S9 = 25;
constexpr uint32_t REG_S10 = 26;
constexpr uint32_t REG_S11 = 27;
constexpr uint32_t REG_T3 = 28;
constexpr uint32_t REG_T4 = 29;
constexpr uint32_t REG_T5 = 30;
constexpr uint32_t REG_T6 = 31;

constexpr uint32_t HOST_ECALL_TERMINATE = 0;
constexpr uint32_t HOST_ECALL_READ = 1;
constexpr uint32_t HOST_ECALL_WRITE = 2;
constexpr uint32_t HOST_ECALL_POSEIDON2 = 3;
constexpr uint32_t HOST_ECALL_SHA2 = 4;

constexpr uint32_t PFLAG_IS_ELEM = 0x80000000;
constexpr uint32_t PFLAG_CHECK_OUT = 0x40000000;

namespace MajorType {
constexpr uint32_t MISC0 = 0;
constexpr uint32_t MISC1 = 1;
constexpr uint32_t MISC2 = 2;
constexpr uint32_t MUL0 = 3;
constexpr uint32_t DIV0 = 4;
constexpr uint32_t MEM0 = 5;
constexpr uint32_t MEM1 = 6;
constexpr uint32_t CONTROL0 = 7;
constexpr uint32_t ECALL0 = 8;
constexpr uint32_t POSEIDON0 = 9;
constexpr uint32_t POSEIDON1 = 10;
} // namespace MajorType

// State of 32 -> Decode (major / minor determined by instruction)
// State 0-32-> major = 7 + state / 8, minor = state % 8

constexpr uint32_t STATE_LOAD_ROOT = 0;
constexpr uint32_t STATE_RESUME = 1;
constexpr uint32_t STATE_SUSPEND = 4;
constexpr uint32_t STATE_STORE_ROOT = 5;
constexpr uint32_t STATE_CONTROL_TABLE = 6;
constexpr uint32_t STATE_CONTROL_DONE = 7;

constexpr uint32_t STATE_MACHINE_ECALL = 8;
constexpr uint32_t STATE_TERMINATE = 9;
constexpr uint32_t STATE_HOST_READ_SETUP = 10;
constexpr uint32_t STATE_HOST_WRITE = 11;
constexpr uint32_t STATE_HOST_READ_BYTES = 12;
constexpr uint32_t STATE_HOST_READ_WORDS = 13;

constexpr uint32_t STATE_POSEIDON_ENTRY = 16;
constexpr uint32_t STATE_POSEIDON_LOAD_STATE = 17;
constexpr uint32_t STATE_POSEIDON_LOAD_IN = 18;
constexpr uint32_t STATE_POSEIDON_DO_OUT = 21;
constexpr uint32_t STATE_POSEIDON_PAGING = 22;
constexpr uint32_t STATE_POSEIDON_STORE_STATE = 23;

constexpr uint32_t STATE_POSEIDON_EXT_ROUND = 24;
constexpr uint32_t STATE_POSEIDON_INT_ROUND = 25;

constexpr uint32_t STATE_SHA_ECALL = 32;
constexpr uint32_t STATE_SHA_LOAD_STATE = 33;
constexpr uint32_t STATE_SHA_LOAD_DATA = 34;
constexpr uint32_t STATE_SHA_MIX = 35;
constexpr uint32_t STATE_SHA_STORE_STATE = 36;

constexpr uint32_t STATE_DECODE = 40;

constexpr uint32_t SAFE_WRITE_WORD = 0x3fffc040;

namespace ControlMinorType {
constexpr uint32_t RESUME = 1;
constexpr uint32_t USER_ECALL = 2;
constexpr uint32_t MRET = 3;
} // namespace ControlMinorType

namespace ECallMinorType {
constexpr uint32_t MACHINE_ECALL = 0;
constexpr uint32_t TERMINATE = 1;
constexpr uint32_t HOST_READ_SETUP = 2;
constexpr uint32_t HOST_WRITE = 3;
constexpr uint32_t HOST_READ_BYTES = 4;
constexpr uint32_t HOST_READ_WORDS = 5;
} // namespace ECallMinorType

namespace PoseidonMinorType {
constexpr uint32_t LOAD_STATE = 0;
constexpr uint32_t LOAD_DATA = 1;
constexpr uint32_t EXT_ROUND = 2;
constexpr uint32_t INT_ROUNDS = 3;
constexpr uint32_t STORE_STATE = 4;
} // namespace PoseidonMinorType

namespace TxKind {
constexpr uint32_t READ = 0;
constexpr uint32_t PAGE_IN = 1;
constexpr uint32_t PAGE_OUT = 2;
} // namespace TxKind

} // namespace zirgen::rv32im_v2
