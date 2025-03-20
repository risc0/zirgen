// Copyright 2025 RISC Zero, Inc.
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

#include "zirgen/circuit/rv32im/v2/platform/constants.h"

using namespace zirgen::rv32im_v2;

inline void die() {
  asm("fence\n");
}

// Implement machine mode ECALLS

inline void terminate(uint32_t val) {
  register uintptr_t a0 asm("a0") = val;
  register uintptr_t a7 asm("a7") = 0;
  asm volatile("ecall\n"
               :                  // no outputs
               : "r"(a0), "r"(a7) // inputs
               :                  // no clobbers
  );
}

inline void sys_bigint2(uint32_t* entry,
                        const uint32_t* a = nullptr,
                        const uint32_t* b = nullptr,
                        const uint32_t* c = nullptr,
                        const uint32_t* d = nullptr) {
  uint32_t* nondetProg = entry + 4;
  uint32_t* verifyProg = nondetProg + entry[0];
  uint32_t* consts = verifyProg + entry[1];
  uint32_t tmpSpace = entry[3] * 4;

  asm volatile("li a7, 5\n"
               "li t0, 1\n"
               "mv t1, %0\n"
               "mv t2, %1\n"
               "mv t3, %2\n"
               "sub sp, sp, %3\n"
               "mv a0, %4\n"
               "mv a1, %5\n"
               "mv a2, %6\n"
               "mv a3, %7\n"
               "mv a4, %8\n"
               "ecall\n"
               "add sp, sp, %3\n"
               : // outputs
               : "r"(nondetProg),
                 "r"(verifyProg),
                 "r"(consts),
                 "r"(tmpSpace),
                 "r"(entry),
                 "r"(a),
                 "r"(b),
                 "r"(c),
                 "r"(d)                                                               // inputs
               : "t0", "t1", "t2", "t3", "a0", "a1", "a2", "a3", "a4", "a7", "memory" // clobbers
  );
}

uint32_t code[] = {
#include "zirgen/circuit/rv32im/v1/test/bigint2.inc"
};

uint32_t point_g[] = {
    0x16F81798,
    0x59F2815B,
    0x2DCE28D9,
    0x029BFCDB,
    0xCE870B07,
    0x55A06295,
    0xF9DCBBAC,
    0x79BE667E,

    0xFB10D4B8,
    0x9C47D08F,
    0xA6855419,
    0xFD17B448,
    0x0E1108A8,
    0x5DA4FBFC,
    0x26A3C465,
    0x483ADA77,
};

extern "C" void start() {
  uint32_t point_g2[16];
  sys_bigint2(code, point_g, point_g2);

  terminate(0);
}
