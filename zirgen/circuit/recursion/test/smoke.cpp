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

} // namespace zirgen::recursion

// Hello
