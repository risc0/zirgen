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

#include "zirgen/compiler/zkp/poseidon2.h"
#include "zirgen/compiler/zkp/baby_bear.h"
#include <gtest/gtest.h>

namespace zirgen {

TEST(zkp, poseidon2) {
  uint32_t input[16];
  for (uint32_t i = 0; i < 16; i++) {
    input[i] = i;
  }
  Digest out = poseidon2Hash(input, 16);

  Digest goal = {toMontgomery(1749308481),
                 toMontgomery(879447913),
                 toMontgomery(499502012),
                 toMontgomery(1842374203),
                 toMontgomery(1869354733),
                 toMontgomery(71489094),
                 toMontgomery(19273002),
                 toMontgomery(690566044)};
  ASSERT_EQ(out, goal);
}

TEST(zkp, poseidon2_long) {
  uint32_t input[32];
  for (uint32_t i = 0; i < 32; i++) {
    input[i] = i;
  }
  Digest out = poseidon2Hash(input, 32);

  Digest goal = {toMontgomery(1257374621),
                 toMontgomery(1235708219),
                 toMontgomery(1590109606),
                 toMontgomery(1571950965),
                 toMontgomery(936452277),
                 toMontgomery(615799448),
                 toMontgomery(844422484),
                 toMontgomery(1109152478)};
  ASSERT_EQ(out, goal);
}

TEST(zkp, poseidon2_unaligned) {
  uint32_t input[20];
  for (uint32_t i = 0; i < 20; i++) {
    input[i] = i;
  }
  Digest out = poseidon2Hash(input, 20);

  Digest goal = {toMontgomery(604605911),
                 toMontgomery(966112445),
                 toMontgomery(161941635),
                 toMontgomery(1104782587),
                 toMontgomery(1646345168),
                 toMontgomery(1860725044),
                 toMontgomery(90293413),
                 toMontgomery(311999068)};
  ASSERT_EQ(out, goal);
}

TEST(zkp, poseidon2_empty) {
  uint32_t input[1] = {0};
  Digest out = poseidon2Hash(input, 0);

  Digest goal = {0x6c506ad3,
                 0x5f9bf6f5,
                 0x52e0e2d3,
                 0x62b9418a,
                 0x08710488,
                 0x2eae5f3a,
                 0x503eb8c2,
                 0x4e1f9cd9};
  ASSERT_EQ(out, goal);
}

} // namespace zirgen
