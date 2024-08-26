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

#include "zirgen/components/mux.h"

#include <gtest/gtest.h>

using namespace zirgen::Zll;

namespace zirgen {

struct Arm1Impl : public CompImpl<Arm1Impl> {
  void set(Reg out, Val aVal, Val bVal) {
    a->set(aVal);
    b->set(bVal);
    out->set(a * a + b);
  }
  Reg a;
  Bit b;
};
using Arm1 = Comp<Arm1Impl>;

struct Arm2Impl : public CompImpl<Arm2Impl> {
  void set(Reg out, Val aVal, Val bVal) {
    a->set(aVal);
    b->set(bVal);
    out->set(a * a);
  }
  Twit a;
  Reg b;
};
using Arm2 = Comp<Arm2Impl>;

TEST(Mux, basic) {
  Module module;
  module.addFunc<2>("test_func", {cbuf(3), mbuf(6)}, [](Buffer code, Buffer data) {
    CompContext::init({});
    CompContext::addBuffer("code", code);
    CompContext::addBuffer("data", data);

    Reg which("code");
    Reg aIn("code");
    Reg bIn("code");
    Reg out;
    OneHot<2> oh;
    TwitPrepare<1> twitPrep;
    Mux<Arm1, Arm2> mux(oh);

    oh->set(which);
    mux->doMux([&](auto arm) { arm->set(out, aIn, bIn); });

    CompContext::fini();
  });
  module.optimize();
  // module.dump();
  auto run = [&](uint64_t which, uint64_t a, uint64_t b) {
    Interpreter::Buffer in = {{which}, {a}, {b}};
    Interpreter::Buffer out(6, {kFieldInvalid});
    module.runFunc("test_func", {in, out});
    return out[0][0];
  };
  ASSERT_EQ(run(0, 17, 1), 17 * 17 + 1);
  ASSERT_EQ(run(1, 3, 10), 9);
  ASSERT_THROW(run(0, 17, 2), std::exception);
  ASSERT_THROW(run(1, 4, 10), std::exception);
}

} // namespace zirgen
