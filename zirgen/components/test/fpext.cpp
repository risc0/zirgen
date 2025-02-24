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
