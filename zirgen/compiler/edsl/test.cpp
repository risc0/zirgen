// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "edsl.h"

using namespace zirgen;

int main() {
  Module module;
  module.addFunc<2>("test_func", {cbuf(2), mbuf(3)}, [](Buffer in, Buffer regs) {
    // clang-format off
    Val x = 1 / in[1];
    NONDET {
      regs[2] = 13;
    }
    IF(x - 7) {
      regs[1] = 7;
    }
    Val a = 3;
    Val c = -a;
    Val b = 4;
    regs[0] = in[0] * in[0] + x + (a + b) * c;
    // clang-format on
  });
  module.optimize();
  module.dump();
}
