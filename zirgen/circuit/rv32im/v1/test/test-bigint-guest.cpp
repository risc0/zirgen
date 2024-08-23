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

constexpr uint32_t zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
constexpr uint32_t one[8] = {1, 0, 0, 0, 0, 0, 0, 0};

constexpr uint32_t x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
constexpr uint32_t y[8] = {9, 10, 11, 12, 13, 14, 15, 16};

// Half-width (128-bit) inputs to test the "checked multiply" mode.
constexpr uint32_t xHalf[8] = {1, 2, 3, 4, 0, 0, 0, 0};
constexpr uint32_t yHalf[8] = {9, 10, 11, 12, 0, 0, 0, 0};

constexpr uint32_t mod[8] = {17, 18, 19, 20, 21, 22, 23, 24};
constexpr uint32_t xyModN[8] = {
    725175305,
    3727701367,
    3590724717,
    3360403226,
    4071262838,
    2077883780,
    2470597663,
    13,
};
constexpr uint32_t xyHalf[8] = {9, 28, 58, 100, 97, 80, 48, 0};

extern "C" void start() {
  uint32_t result[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  // Check multiplication of x and y.
  sys_bigint(result, x, y, mod);
  for (uint32_t i = 0; i < 8; i++) {
    if (result[i] != xyModN[i]) {
      fail();
    }
  }

  // Check multiplication by 1.
  sys_bigint(result, x, one, mod);
  for (uint32_t i = 0; i < 8; i++) {
    if (result[i] != x[i]) {
      fail();
    }
  }

  // Check multiplication by 0.
  sys_bigint(result, x, zero, mod);
  for (uint32_t i = 0; i < 8; i++) {
    if (result[i] != zero[i]) {
      fail();
    }
  }

  // Check multiplication with modulus of zero..
  sys_bigint(result, xHalf, yHalf, zero);
  for (uint32_t i = 0; i < 8; i++) {
    if (result[i] != xyHalf[i]) {
      fail();
    }
  }
  sys_halt();
}
