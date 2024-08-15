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

// This code is automatically generated

#include "ffi.h"
#include "fp.h"

#include <array>
#include <cassert>
#include <stdexcept>

// clang-format off
namespace risc0::circuit::fib {

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

Fp step_exec(void* ctx, HostBridge host, size_t steps, size_t cycle, Fp** args) {
  size_t mask = steps - 1;
  std::array<Fp, 96> host_args;
  std::array<Fp, 32> host_outs;
  // loc(unknown)
  Fp x0(1);
  // loc("zirgen/circuit/fib/fib.cpp":20:0)
  auto x1 = args[0][0 * steps + ((cycle - 0) & mask)];
  assert(x1 != Fp::invalid());
  if (x1 != 0) {
    // loc("zirgen/circuit/fib/fib.cpp":21:0)
    {
      auto& reg = args[2][0 * steps + cycle];
      assert(reg == Fp::invalid() || reg == x0);
      reg = x0;
    }
  }
  // loc("zirgen/circuit/fib/fib.cpp":23:0)
  auto x2 = args[0][1 * steps + ((cycle - 0) & mask)];
  assert(x2 != Fp::invalid());
  if (x2 != 0) {
    // loc("zirgen/circuit/fib/fib.cpp":24:0)
    auto x3 = args[2][0 * steps + ((cycle - 2) & mask)];
    assert(x3 != Fp::invalid());
    // loc("zirgen/circuit/fib/fib.cpp":24:0)
    auto x4 = args[2][0 * steps + ((cycle - 1) & mask)];
    assert(x4 != Fp::invalid());
    // loc("zirgen/circuit/fib/fib.cpp":24:0)
    auto x5 = x4 + x3;
    // loc("zirgen/circuit/fib/fib.cpp":24:0)
    {
      auto& reg = args[2][0 * steps + cycle];
      assert(reg == Fp::invalid() || reg == x5);
      reg = x5;
    }
  }
  // loc("zirgen/circuit/fib/fib.cpp":26:0)
  auto x6 = args[0][2 * steps + ((cycle - 0) & mask)];
  assert(x6 != Fp::invalid());
  if (x6 != 0) {
    // loc("zirgen/circuit/fib/fib.cpp":28:0)
    auto x7 = args[2][0 * steps + ((cycle - 0) & mask)];
    assert(x7 != Fp::invalid());
    // loc("zirgen/circuit/fib/fib.cpp":28:0)
    args[1][0] = x7;
  }
  // loc("zirgen/circuit/fib/fib.cpp":30:0)
  auto x8 = x0 - x6;
  return x8;
}

} // namespace risc0::circuit::fib
// clang-format on
