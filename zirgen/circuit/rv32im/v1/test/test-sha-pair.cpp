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

uint8_t sha_init[] = {
    0x6a, 0x09, 0xe6, 0x67, //
    0xbb, 0x67, 0xae, 0x85, //
    0x3c, 0x6e, 0xf3, 0x72, //
    0xa5, 0x4f, 0xf5, 0x3a, //
    0x51, 0x0e, 0x52, 0x7f, //
    0x9b, 0x05, 0x68, 0x8c, //
    0x1f, 0x83, 0xd9, 0xab, //
    0x5b, 0xe0, 0xcd, 0x19, //
};

uint32_t test_a[] = {0x80, 0, 0, 0, 0, 0, 0, 0};
uint32_t test_b[] = {0, 0, 0, 0, 0, 0, 0, 0};

uint8_t golden[] = {
    0xe3, 0xb0, 0xc4, 0x42, //
    0x98, 0xfc, 0x1c, 0x14, //
    0x9a, 0xfb, 0xf4, 0xc8, //
    0x99, 0x6f, 0xb9, 0x24, //
    0x27, 0xae, 0x41, 0xe4, //
    0x64, 0x9b, 0x93, 0x4c, //
    0xa4, 0x95, 0x99, 0x1b, //
    0x78, 0x52, 0xb8, 0x55, //
};

extern "C" void start() {
  uint8_t out[8 * 4];
  sys_sha_pair((uint32_t*)out, (uint32_t*)sha_init, test_a, test_b);
  for (int i = 0; i < 8 * 4; i++) {
    if (out[i] != golden[i]) {
      fail();
    }
  }
  sys_halt();
}
