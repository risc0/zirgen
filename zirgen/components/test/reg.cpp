// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/components/reg.h"
#include "zirgen/components/fpext.h"

#include <gtest/gtest.h>

using namespace zirgen::Zll;

namespace zirgen {

TEST(Reg, basic) {
  Module module;
  module.addFunc<2>(
      "test_func", {cbuf(2, kExtSize), mbuf(1, kExtSize)}, [](Buffer in, Buffer regs) {
        CompContext::init({});
        CompContext::addBuffer("code", in);
        CompContext::addBuffer("data", regs);

        Reg a("code");
        Reg b("code");
        Reg out;
        out->set(((a * b) + b) * a + b);

        CompContext::fini();
      });
  module.dump();
  module.optimize();
  auto run = [&](Interpreter::Polynomial a, Interpreter::Polynomial b) {
    Interpreter::Buffer in(2);
    Interpreter::Buffer out(1, Interpreter::Polynomial(kExtSize, kFieldInvalid));
    in[0] = a;
    in[1] = b;
    module.runFunc("test_func", {in, out});
    return out;
  };
  Interpreter::Polynomial a = {0, 3, 0, 0};
  Interpreter::Polynomial b = {5, 2, 0, 0};
  ASSERT_EQ(run(a, b), Interpreter::Buffer({{5, 17, 51, 18}}));
}

} // namespace zirgen
