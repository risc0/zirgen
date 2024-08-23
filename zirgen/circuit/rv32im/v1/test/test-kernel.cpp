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

#include <stdint.h>
#include <sys/errno.h>

#include "zirgen/circuit/rv32im/v1/platform/constants.h"

using namespace zirgen::rv32im_v1;

inline void die() {
  asm("fence\n");
}

// Implement machine mode ECALLS we need
inline void machine_halt() {
  register uintptr_t t0 asm("t0") = 0;     // HALT
  register uintptr_t a0 asm("a0") = 0;     // Normal
  register uintptr_t a1 asm("a1") = 0x400; // Write whatever is in lowmem
  asm volatile("ecall"
               :                           // outputs
               : "r"(t0), "r"(a0), "r"(a1) // inputs
               :                           // clobbers
  );
}

inline void mret() {
  asm volatile("li t0, 5\n" // Jump to usermode
               "ecall\n"
               :                // outputs
               :                // inputs
               : "t0", "memory" // clobbers
  );
}

constexpr uint32_t USER_START_ADDR = 1024;     // 1 page in
constexpr uint32_t USER_END_ADDR = 0x08000000; // Halfway through memory

// Helpers to access user registers
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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
inline uint32_t get_ureg(uint32_t reg) {
  return (reinterpret_cast<uint32_t*>(kUserRegisterOffset * 4))[reg];
}
inline void set_ureg(uint32_t reg, uint32_t val) {
  (reinterpret_cast<uint32_t*>(kUserRegisterOffset * 4))[reg] = val;
}
#pragma GCC diagnostic pop

// Implement system calls

inline uint32_t sys_exit(uint32_t code) {
  machine_halt();
  return 0;
}

inline uint32_t sys_read(uint32_t fd, uint32_t buf, uint32_t len) {
  if (fd != 0) {
    return -EBADF;
  }
  if (buf + len < buf) {
    return -EINVAL;
  }
  if (buf < USER_START_ADDR || buf + len >= USER_END_ADDR) {
    return -EFAULT;
  }
  // Fake out read for now
  if (len != 4) {
    die();
  }
  // Always read 5
  *reinterpret_cast<uint32_t*>(buf) = 5;
  return 4;
}

inline uint32_t sys_write(uint32_t fd, uint32_t buf, uint32_t len) {
  if (fd != 1) {
    return -EBADF;
  }
  if (buf + len < buf) {
    return -EINVAL;
  }
  if (buf < USER_START_ADDR || buf + len >= USER_END_ADDR) {
    return -EFAULT;
  }
  // Fake out write for now
  if (len != 4) {
    die();
  }
  // Make sure we write 100
  if (*reinterpret_cast<uint32_t*>(buf) != 100) {
    die();
  }
  return 4;
}

void ecall_entry() {
  uint32_t out;
  switch (get_ureg(REG_A7)) {
  case 63:
    out = sys_read(get_ureg(REG_A0), get_ureg(REG_A1), get_ureg(REG_A2));
    break;
  case 64:
    out = sys_write(get_ureg(REG_A0), get_ureg(REG_A1), get_ureg(REG_A2));
    break;
  case 93:
    out = sys_exit(get_ureg(REG_A0));
    break;
  default:
    out = -EINVAL;
  }
  set_ureg(REG_A0, out);
  mret();
}

extern "C" void start() {
  // Set up user stack
  set_ureg(REG_SP, 0x07ff0000);
  // Set ecall entry
  *reinterpret_cast<uint32_t*>(kECallEntry) = reinterpret_cast<uint32_t>(ecall_entry);
  mret();
}

asm(R"(
.section .text._start;
.globl _start;
_start:
    .option push;
    .option norelax;
    la gp, __global_pointer$;
    .option pop;
    li sp, 0x0bff0000;
    jal ra, start
)");
