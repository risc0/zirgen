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
