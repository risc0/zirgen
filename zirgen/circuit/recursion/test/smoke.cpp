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

#include "zirgen/circuit/recursion/encode.h"
#include "zirgen/circuit/recursion/test/runner.h"
#include "zirgen/compiler/zkp/baby_bear.h"

#include <gtest/gtest.h>

namespace zirgen::recursion {

TEST(RECURSION, Smoke) {
  Runner runner;
  Module module;
  module.addFunc<2>("test", {gbuf(8), ioparg()}, [&](Buffer out, ReadIopVal iop) {
    Val a = iop.readBaseVals(1)[0];
    Val b = iop.readBaseVals(1)[0];
    out[0] = a + b;
  });
  module.optimize();
  module.dump();
  auto func = module.getModule().lookupSymbol<mlir::func::FuncOp>("test");
  std::vector<uint32_t> code = encode(recursion::HashType::SHA256, &func.front());
  std::vector<uint32_t> proof = {(kBabyBearToMontgomery * 1) % kBabyBearP,
                                 (kBabyBearToMontgomery * 2) % kBabyBearP};
  runner.setup(code, proof);
  runner.run();
  auto expected = llvm::SmallVector<uint64_t, 4>({3});
  ASSERT_EQ(runner.out[0], expected);
  runner.done();
}

// BIT_AND_ELEM constrains its inputs to have canonical 31-bit decompositions.  Check that honest
// values at the edges of that check still work: values below 2^31 - P (which would otherwise have a
// second decomposition x + P), values with all of the top 4 bits set, and P - 1 itself.
TEST(RECURSION, BitAndElemCanonicalBoundary) {
  constexpr uint32_t kTop4 = 0x78000000; // bits 27-30 set, i.e. P - 1
  std::vector<std::pair<uint32_t, uint32_t>> cases = {
      {0, 0xffff},
      {1, 0xffff},
      {(1u << 27) - 2, 0xffff},
      {(1u << 27) - 1, 0xffff},
      {kTop4, 0xffff},
      {kTop4, kTop4},
      {kTop4, 0x3fffffff},
      {0x12345678, 0x7800ffff - 0x10000}, // 0x77ffffff: top 4 bits are 0b1110
  };
  for (auto [a, b] : cases) {
    ASSERT_LT(a, kBabyBearP);
    ASSERT_LT(b, kBabyBearP);
    // Each SET_GLOBAL writes a zero padded half digest, so run one case per program.
    Runner runner;
    Module module;
    module.addFunc<2>("test", {gbuf(8), ioparg()}, [&](Buffer out, ReadIopVal iop) {
      Val x = iop.readBaseVals(1)[0];
      Val y = iop.readBaseVals(1)[0];
      out[0] = x & y;
    });
    module.optimize();
    auto func = module.getModule().lookupSymbol<mlir::func::FuncOp>("test");
    std::vector<uint32_t> code = encode(recursion::HashType::SHA256, &func.front());
    std::vector<uint32_t> proof = {uint32_t((uint64_t(kBabyBearToMontgomery) * a) % kBabyBearP),
                                   uint32_t((uint64_t(kBabyBearToMontgomery) * b) % kBabyBearP)};
    runner.setup(code, proof);
    runner.run();
    auto expected = llvm::SmallVector<uint64_t, 4>({a & b});
    EXPECT_EQ(runner.out[0], expected) << "a = " << a << ", b = " << b;
    runner.done();
  }
}

} // namespace zirgen::recursion

// Hello
