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
namespace risc0::circuit::keccak {





FpExt poly_fp(size_t cycle, size_t steps, FpExt* poly_mix, Fp** args) {
  size_t mask = steps - 1;
  // loc(unknown)
  constexpr Fp x0(0);
  // loc(unknown)
  constexpr Fp x1(1);
  // loc(unknown)
  constexpr FpExt x2(0,1,0,0);
  // loc(unknown)
  FpExt x3 = FpExt(0);
  // loc(callsite(unknown at callsite("Reg"("<preamble>":4:21) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":20:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown)))))
  auto x4 = /*global=*/args[2][0];
  // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":20:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown))))
  auto x5 = x0 - x4;
  // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":20:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown))))
  FpExt x6 = x3 + x5 * poly_mix[0];
  // loc(callsite(unknown at callsite("Top"("zirgen/dsl/test/simple-accum.zir":21:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown))))
  auto x7 = /*data=*/args[1][0 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite(unknown at callsite("Top"("zirgen/dsl/test/simple-accum.zir":22:14) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown))))
  auto x8 = x1 - x7;
  // loc(callsite(unknown at callsite("Reg"("<preamble>":4:21) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":22:13) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown)))))
  auto x9 = /*data=*/args[1][1 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":22:13) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown))))
  auto x10 = x8 - x9;
  // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":22:13) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:2) at unknown))))
  FpExt x11 = x6 + x10 * poly_mix[1];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x12 = /*mix=*/args[3][3];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x13 = /*mix=*/args[3][2];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x14 = x12 * x2;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x15 = x13 + x14;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x16 = /*mix=*/args[3][1];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x17 = x15 * x2;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x18 = x16 + x17;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x19 = /*mix=*/args[3][0];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x20 = x18 * x2;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x21 = x19 + x20;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x22 = /*mix=*/args[3][7];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x23 = /*mix=*/args[3][6];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x24 = x22 * x2;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x25 = x23 + x24;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x26 = /*mix=*/args[3][5];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x27 = x25 * x2;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x28 = x26 + x27;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x29 = /*mix=*/args[3][4];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x30 = x28 * x2;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":82:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x31 = x29 + x30;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x32 = /*accum=*/args[0][3 * steps + ((cycle - kInvRate * 1) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x33 = /*accum=*/args[0][2 * steps + ((cycle - kInvRate * 1) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x34 = x32 * x2;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x35 = x33 + x34;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x36 = /*accum=*/args[0][1 * steps + ((cycle - kInvRate * 1) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x37 = x35 * x2;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x38 = x36 + x37;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x39 = /*accum=*/args[0][0 * steps + ((cycle - kInvRate * 1) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x40 = x38 * x2;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":91:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x41 = x39 + x40;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":232:47 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x42 = /*data=*/args[1][2 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":141:56 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x43 = /*data=*/args[1][3 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":142:36 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x44 = x21 * x43;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":234:47 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x45 = x44 + x31;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x46 = /*accum=*/args[0][3 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x47 = /*accum=*/args[0][2 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x48 = x46 * x2;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x49 = x47 + x48;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x50 = /*accum=*/args[0][1 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x51 = x49 * x2;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x52 = x50 + x51;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x53 = /*accum=*/args[0][0 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x54 = x52 * x2;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":186:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x55 = x53 + x54;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":173:42 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x56 = x55 - x41;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":174:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x57 = x56 * x45;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":176:42 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  auto x58 = x57 - x42;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":178:33 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  FpExt x59 = x3 + x58 * poly_mix[0];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":439:9 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  FpExt x60 = x11 + x7 * x59 * poly_mix[2];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":439:9 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":401:9 at unknown)))
  FpExt x61 = x60 + x9 * x59 * poly_mix[3];
  return x61;
}

} // namespace risc0::circuit::keccak
// clang-format on
