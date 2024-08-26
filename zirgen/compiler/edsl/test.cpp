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
