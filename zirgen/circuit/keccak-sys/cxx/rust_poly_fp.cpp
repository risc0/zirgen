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
  std::cerr << "poly_fp cycle " << cycle << "/" << steps << "\n";
  size_t mask = steps - 1;
  // loc(unknown)
  constexpr Fp x0(11);
  // loc(unknown)
  constexpr Fp x1(1);
  // loc(unknown)
  constexpr Fp x2(2);
  // loc(unknown)
  constexpr Fp x3(3);
  // loc(unknown)
  constexpr Fp x4(2013265919);
  // loc(unknown)
  constexpr FpExt x5(0,1,0,0);
  // loc(unknown)
  FpExt x6 = FpExt(0);
  // loc(callsite(unknown at callsite("Reg"("<preamble>":4:21) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":17:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown)))))
  auto x7 = /*global=*/args[2][0];
  // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":17:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
  auto x8 = x0 - x7;
  // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":17:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
  FpExt x9 = x6 + x8 * poly_mix[0];
  // loc(callsite(unknown at callsite("Top"("zirgen/dsl/test/simple-accum.zir":18:20) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
  auto x10 = /*data=*/args[1][0 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite(unknown at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:14) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
  auto x11 = x1 - x10;
  // loc(callsite(unknown at callsite("Reg"("<preamble>":4:21) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:13) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown)))))
  auto x12 = /*data=*/args[1][1 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:13) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
  auto x13 = x11 - x12;
  // loc(callsite("Reg"("<preamble>":5:7) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:13) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
  FpExt x14 = x9 + x13 * poly_mix[1];
  // loc(callsite(unknown at callsite("NondetMyArgument"("zirgen/dsl/test/simple-accum.zir":5:18) at callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":10:25) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:37) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))))
  auto x15 = /*data=*/args[1][2 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite(unknown at callsite("NondetMyArgument"("zirgen/dsl/test/simple-accum.zir":6:18) at callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":10:25) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:37) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))))
  auto x16 = /*data=*/args[1][3 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":11:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:37) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
  auto x17 = x2 - x15;
  // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":11:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:37) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
  FpExt x18 = x6 + x17 * poly_mix[0];
  // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":12:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:37) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
  auto x19 = x3 - x16;
  // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":12:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:37) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
  FpExt x20 = x18 + x19 * poly_mix[1];
  // loc(callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:22) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown)))
  FpExt x21 = x14 + x10 * x20 * poly_mix[2];
  // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":11:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:54) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
  auto x22 = x4 - x15;
  // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":11:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:54) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
  FpExt x23 = x6 + x22 * poly_mix[0];
  // loc(callsite("MyArgument"("zirgen/dsl/test/simple-accum.zir":12:6) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:54) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown))))
  FpExt x24 = x23 + x19 * poly_mix[1];
  // loc(callsite("Top"("zirgen/dsl/test/simple-accum.zir":19:22) at callsite("Top"("zirgen/dsl/test/simple-accum.zir":16:2) at unknown)))
  FpExt x25 = x21 + x12 * x24 * poly_mix[3];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x26 = /*mix=*/args[3][3];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x27 = /*mix=*/args[3][2];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x28 = x26 * x5;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x29 = x28 + x27;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x30 = /*mix=*/args[3][1];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x31 = x29 * x5;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x32 = x31 + x30;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x33 = /*mix=*/args[3][0];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x34 = x32 * x5;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":44:44 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x35 = x34 + x33;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x36 = /*mix=*/args[3][7];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x37 = /*mix=*/args[3][6];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x38 = x36 * x5;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x39 = x38 + x37;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x40 = /*mix=*/args[3][5];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x41 = x39 * x5;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x42 = x41 + x40;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x43 = /*mix=*/args[3][4];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x44 = x42 * x5;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":81:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x45 = x44 + x43;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x46 = /*accum=*/args[0][3 * steps + ((cycle - kInvRate * 1) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x47 = /*accum=*/args[0][2 * steps + ((cycle - kInvRate * 1) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x48 = x46 * x5;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x49 = x48 + x47;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x50 = /*accum=*/args[0][1 * steps + ((cycle - kInvRate * 1) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x51 = x49 * x5;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x52 = x51 + x50;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x53 = /*accum=*/args[0][0 * steps + ((cycle - kInvRate * 1) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x54 = x52 * x5;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":90:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x55 = x54 + x53;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":141:36 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x56 = x35 * x16;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":233:47 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x57 = x56 + x45;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x58 = /*accum=*/args[0][3 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x59 = /*accum=*/args[0][2 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x60 = x58 * x5;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x61 = x60 + x59;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x62 = /*accum=*/args[0][1 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x63 = x61 * x5;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x64 = x63 + x62;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x65 = /*accum=*/args[0][0 * steps + ((cycle - kInvRate * 0) & mask)];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x66 = x64 * x5;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":185:50 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x67 = x66 + x65;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":172:42 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x68 = x67 - x55;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":173:46 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x69 = x68 * x57;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":175:42 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  auto x70 = x69 - x15;
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":177:33 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  FpExt x71 = x6 + x70 * poly_mix[0];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":438:9 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  FpExt x72 = x25 + x10 * x71 * poly_mix[4];
  // loc(callsite("zirgen/dsl/passes/GenerateAccum.cpp":438:9 at callsite("zirgen/dsl/passes/GenerateAccum.cpp":400:9 at unknown)))
  FpExt x73 = x72 + x12 * x71 * poly_mix[5];
  return x73;
}

} // namespace risc0::circuit::keccak
// clang-format on
