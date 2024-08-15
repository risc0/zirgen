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

#include "fp.h"
#include "fpext.h"

#include <cstdint>

constexpr size_t kInvRate = 4;

// clang-format off
namespace risc0::circuit::fib {

FpExt poly_fp(size_t cycle, size_t steps, FpExt* poly_mix, Fp** args) {
  size_t mask = steps - 1;
  // loc(unknown)
  Fp x0(1);
  // loc("zirgen/circuit/fib/fib.cpp":38:0)
  FpExt x1 = FpExt(0);
  // loc("zirgen/circuit/fib/fib.cpp":20:0)
  auto x2 = args[0][0 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc("zirgen/circuit/fib/fib.cpp":21:0)
  auto x3 = args[2][0 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc("zirgen/circuit/fib/fib.cpp":21:0)
  auto x4 = x3 - x0;
  // loc("zirgen/circuit/fib/fib.cpp":21:0)
  FpExt x5 = x1 + x4 * poly_mix[0];
  // loc("zirgen/circuit/fib/fib.cpp":20:0)
  FpExt x6 = x1 + x2 * x5 * poly_mix[0];
  // loc("zirgen/circuit/fib/fib.cpp":23:0)
  auto x7 = args[0][1 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc("zirgen/circuit/fib/fib.cpp":24:0)
  auto x8 = args[2][0 * steps + ((cycle - kInvRate * 2) & mask)];
  // loc("zirgen/circuit/fib/fib.cpp":24:0)
  auto x9 = args[2][0 * steps + ((cycle - kInvRate * 1) & mask)];
  // loc("zirgen/circuit/fib/fib.cpp":24:0)
  auto x10 = x9 + x8;
  // loc("zirgen/circuit/fib/fib.cpp":24:0)
  auto x11 = x3 - x10;
  // loc("zirgen/circuit/fib/fib.cpp":24:0)
  FpExt x12 = x1 + x11 * poly_mix[0];
  // loc("zirgen/circuit/fib/fib.cpp":23:0)
  FpExt x13 = x6 + x7 * x12 * poly_mix[1];
  // loc("zirgen/circuit/fib/fib.cpp":26:0)
  auto x14 = args[0][2 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc("zirgen/circuit/fib/fib.cpp":28:0)
  auto x15 = args[1][0];
  // loc("zirgen/circuit/fib/fib.cpp":28:0)
  auto x16 = x15 - x3;
  // loc("zirgen/circuit/fib/fib.cpp":28:0)
  FpExt x17 = x1 + x16 * poly_mix[0];
  // loc("zirgen/circuit/fib/fib.cpp":26:0)
  FpExt x18 = x13 + x14 * x17 * poly_mix[2];
  // loc("zirgen/circuit/fib/fib.cpp":34:0)
  auto x19 = x2 + x7;
  // loc("zirgen/circuit/fib/fib.cpp":34:0)
  auto x20 = x19 + x14;
  // loc("zirgen/circuit/fib/fib.cpp":35:0)
  auto x21 = args[4][0 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc("zirgen/circuit/fib/fib.cpp":35:0)
  auto x22 = x21 - x0;
  // loc("zirgen/circuit/fib/fib.cpp":35:0)
  FpExt x23 = x1 + x22 * poly_mix[0];
  // loc("zirgen/circuit/fib/fib.cpp":34:0)
  FpExt x24 = x18 + x20 * x23 * poly_mix[3];
  return x24;
}

} // namespace risc0::circuit::fib
// clang-format on
