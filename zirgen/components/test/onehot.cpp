// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/components/onehot.h"

#include <gtest/gtest.h>

namespace zirgen {
using namespace Zll;

TEST(OneHot, basic) {
  Module module;
  module.addFunc<2>("test_func", {cbuf(1), mbuf(3)}, [](Buffer in, Buffer regs) {
    CompContext::init({});
    CompContext::addBuffer("code", in);
    CompContext::addBuffer("data", regs);

    Reg val("code");
    OneHot<3> oh;
    oh->set(val);

    CompContext::fini();
  });
  module.optimize();
  // module.dump();
  auto run = [&](uint64_t val) {
    Interpreter::Buffer in = {{val}};
    Interpreter::Buffer out(3, {kFieldInvalid});
    module.runFunc("test_func", {in, out});
    return out;
  };
  ASSERT_EQ(run(0), Interpreter::Buffer({{1}, {0}, {0}}));
  ASSERT_EQ(run(1), Interpreter::Buffer({{0}, {1}, {0}}));
  ASSERT_EQ(run(2), Interpreter::Buffer({{0}, {0}, {1}}));
  ASSERT_THROW(run(3), std::exception);
}

} // namespace zirgen
