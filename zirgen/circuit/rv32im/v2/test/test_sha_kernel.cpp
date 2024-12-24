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

static constexpr uint32_t SHA_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

constexpr uint32_t swap(uint32_t n) {
  return (((n & 0x000000ff) << 24) | ((n & 0x0000ff00) << 8) | ((n & 0x00ff0000) >> 8) |
          ((n & 0xff000000) >> 24));
}

static constexpr uint32_t SHA_INIT[8] = {
    swap(0x6a09e667),
    swap(0xbb67ae85),
    swap(0x3c6ef372),
    swap(0xa54ff53a),
    swap(0x510e527f),
    swap(0x9b05688c),
    swap(0x1f83d9ab),
    swap(0x5be0cd19),
};

inline void do_sha2(uint32_t stateIn, uint32_t stateOut, uint32_t data, uint32_t count) {
  register uintptr_t a0 asm("a0") = stateIn;
  register uintptr_t a1 asm("a1") = stateOut;
  register uintptr_t a2 asm("a2") = data;
  register uintptr_t a3 asm("a3") = count;
  register uintptr_t a4 asm("a4") = (uint32_t)SHA_K;
  register uintptr_t a7 asm("a7") = 4;
  asm volatile("ecall\n"
               :                                                      // no outputs
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a7) // inputs
               :                                                      // no clobbers
  );
}

void test_sha_basic() {
  uint8_t state[32];
  uint8_t data[64];
  for (size_t i = 0; i < 32; i++) {
    data[i] = 0;
  }
  data[0] = 0x61; // a
  data[1] = 0x62; // b
  data[2] = 0x63; // c
  data[3] = 0x80; // One bit
  data[63] = 24;  // 24 bits
  constexpr uint8_t goal[32] = {
      0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40,
      0xde, 0x5d, 0xae, 0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17,
      0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
  };
  do_sha2((uint32_t)SHA_INIT, (uint32_t)state, (uint32_t)data, 1);
  for (size_t i = 0; i < 32; i++) {
    if (state[i] != goal[i]) {
      die();
    }
  }
}

extern "C" void start() {
  test_sha_basic();
  terminate(0);
}
