// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/components/bits.h"

#include <gtest/gtest.h>

using namespace zirgen::Zll;

namespace zirgen {

TEST(Bits, basic) {
  Module module;
  module.addFunc<2>("test_func", {cbuf(2), mbuf(2)}, [](Buffer code, Buffer data) {
    CompContext::init({});
    CompContext::addBuffer("code", code);
    CompContext::addBuffer("data", data);

    Reg in1("code");
    Reg in2("code");
    Bit theBit;
    TwitPrepare<1> twitPrep;
    Twit theTwit;
    theBit->set(in1);
    theTwit->set(in2);

    CompContext::fini();
  });
  module.optimize();
  // module.dump();
  auto run = [&](uint64_t a, uint64_t b) {
    Interpreter::Buffer in = {{a}, {b}};
    Interpreter::Buffer out(2, {kFieldInvalid});
    module.runFunc("test_func", {in, out});
    return out;
  };
  ASSERT_EQ(run(0, 1), Interpreter::Buffer({{0}, {1}}));
  ASSERT_EQ(run(0, 3), Interpreter::Buffer({{0}, {3}}));
  ASSERT_EQ(run(1, 1), Interpreter::Buffer({{1}, {1}}));
  ASSERT_THROW(run(0, 4), std::exception);
  ASSERT_THROW(run(2, 0), std::exception);
}

} // namespace zirgen
