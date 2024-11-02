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
  // Run some test code
  uint32_t point_g2[16];
  sys_bigint2(code, point_g, point_g2);

  sys_halt();
}
