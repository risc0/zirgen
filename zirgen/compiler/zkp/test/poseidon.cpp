// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/compiler/zkp/poseidon.h"
#include "zirgen/compiler/zkp/baby_bear.h"
#include <gtest/gtest.h>

namespace zirgen {

TEST(zkp, poseidon) {
  uint32_t input[16];
  for (uint32_t i = 0; i < 16; i++) {
    input[i] = i;
  }
  Digest out = poseidonHash(input, 16);

  Digest goal = {toMontgomery(165799421),
                 toMontgomery(446443103),
                 toMontgomery(1242624592),
                 toMontgomery(791266679),
                 toMontgomery(1939888497),
                 toMontgomery(1437820613),
                 toMontgomery(893076101),
                 toMontgomery(95764709)};
  ASSERT_EQ(out, goal);
}

} // namespace zirgen
