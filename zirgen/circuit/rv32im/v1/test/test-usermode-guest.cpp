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

#include <cstdint>

#include "guest.h"
#include "zirgen/circuit/rv32im/v1/platform/constants.h"

using namespace zirgen::rv32im_v1;

uint32_t buf[3] = {0, 0, 0};

inline void sys_ecall(uint32_t val) {
  register uintptr_t a0 asm("a0") = val;
  asm volatile("ecall"
               :         // outputs
               : "r"(a0) // inputs
               :         // clobbers
  );
}

void ecall_entry() {
  // Should we make UserRegAddr be an address instead of a word # ?
  uint32_t userA0 = *reinterpret_cast<uint32_t*>(UserRegAddr::kA0 * 4);
  if (userA0 == 0) {
    if (buf[0] != 17) {
      fail();
    }
    if (buf[1] != 23) {
      fail();
    }
    if (buf[2] != 5) {
      fail();
    }
    sys_halt();
  } else {
    buf[2] = userA0;
  }
  sys_usermode(); // Never returns
}

void user_start() {
  buf[0] = 17;
  sys_ecall(5);
  buf[1] = 23;
  sys_ecall(0);
}

extern "C" void start() {
  *reinterpret_cast<uint32_t*>(kECallEntry) = reinterpret_cast<uint32_t>(ecall_entry);
  *reinterpret_cast<uint32_t*>(kUserPC) = reinterpret_cast<uint32_t>(user_start);
  sys_usermode(); // Never returns
}
