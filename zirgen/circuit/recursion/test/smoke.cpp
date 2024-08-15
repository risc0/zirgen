// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
