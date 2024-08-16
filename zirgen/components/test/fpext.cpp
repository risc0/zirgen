// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/components/fpext.h"

#include "risc0/core/rng.h"
#include "risc0/zkp/core/fp4.h"

#include <gtest/gtest.h>

namespace zirgen {

TEST(FpExt, basic) {
  Module module;
  module.addFunc<2>("test_func", {cbuf(2 * kExtSize), mbuf(kExtSize)}, [](Buffer in, Buffer regs) {
    CompContext::init({});
    CompContext::addBuffer("code", in);
    CompContext::addBuffer("data", regs);

    FpExtReg a("code");
    FpExtReg b("code");
    FpExtReg out;
    out->set(((a * b) + b) * a - b);

    CompContext::fini();
  });
  module.optimize();
  auto run = [&](risc0::Fp4 a, risc0::Fp4 b) {
    std::vector<uint64_t> in(2 * kExtSize);
    std::vector<uint64_t> out(kExtSize, kFieldInvalid);
    for (size_t i = 0; i < kExtSize; i++) {
      in[i] = a.elems[i].asUInt32();
      in[kExtSize + i] = b.elems[i].asUInt32();
    }
    module.runFunc("test_func", {in, out});
    risc0::Fp4 fpOut;
    for (size_t i = 0; i < kExtSize; i++) {
      fpOut.elems[i] = out[i];
    }
    return fpOut;
  };
  risc0::PsuedoRng rng(2);
  for (size_t i = 0; i < 100; i++) {
    auto a = risc0::Fp4::random(rng);
    auto b = risc0::Fp4::random(rng);
    ASSERT_EQ(run(a, b), ((a * b) + b) * a - b);
  }
}

} // namespace zirgen
