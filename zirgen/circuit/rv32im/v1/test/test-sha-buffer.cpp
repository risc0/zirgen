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

uint32_t test[] = {
    1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0, //
    1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0,
};

uint8_t golden1[] = {
    0x04, 0x10, 0x50, 0x05, //
    0x05, 0xeb, 0x63, 0x60, //
    0x8d, 0xef, 0x98, 0x4e, //
    0xcc, 0x0b, 0x78, 0x20, //
    0xcb, 0xa1, 0x01, 0x25, //
    0x70, 0xe3, 0xd2, 0x88, //
    0xc4, 0x83, 0xf3, 0x50, //
    0x21, 0xc9, 0x71, 0xa6, //
};

uint8_t golden2[] = {
    0x03, 0x43, 0xd5, 0x00, //
    0x09, 0x7e, 0x63, 0x12, //
    0x3d, 0x3c, 0x7f, 0x41, //
    0x8f, 0x46, 0x5b, 0xfd, //
    0x22, 0x53, 0x65, 0x2f, //
    0x35, 0x1c, 0x90, 0xc7, //
    0x5a, 0x05, 0xcb, 0x33, //
    0x94, 0x6e, 0x71, 0xf1, //
};

extern "C" void start() {
  uint8_t out[8 * 4];
  sys_sha_buffer((uint32_t*)out, (uint32_t*)sha_init, (char*)test, 1);
  for (int i = 0; i < 8 * 4; i++) {
    if (out[i] != golden1[i]) {
      fail();
    }
  }
  sys_sha_buffer((uint32_t*)out, (uint32_t*)sha_init, (char*)test, 2);
  for (int i = 0; i < 8 * 4; i++) {
    if (out[i] != golden2[i]) {
      fail();
    }
  }
  sys_halt();
}
