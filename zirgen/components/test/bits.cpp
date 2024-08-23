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
