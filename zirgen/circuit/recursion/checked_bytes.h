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

#pragma once

#include "zirgen/circuit/recursion/code.h"
// CheckedBytes includes code for preparing for Poseidon2 full rounds
#include "zirgen/circuit/recursion/poseidon2.h"
#include "zirgen/circuit/recursion/wom.h"

namespace zirgen::recursion {

// Loads bytes, verifies them as bytes, and evaluates a polynomial with them as coefficients
class CheckedBytesImpl : public CompImpl<CheckedBytesImpl> {
public:
  CheckedBytesImpl(Code code, WomHeader header);
  void set(Code code, Val writeAddr);

private:
  using CheckedVals = std::array<FpExt, 16>;

  CheckedVals getPowersOfZ(FpExtReg tmp2, FpExtReg tmp4, FpExtReg tmp10, FpExt in);
  void addPowersOfZConstraint();
  WomBody body;

  WomReg eval_point = Label("eval_point");
  // For each of the 16 input bytes, break into 4 2-bit parts, which can be range-checked with a
  // degree-4 constraint
  std::array<Reg, 16> lowestBits;
  std::array<Reg, 16> midLoBits;
  std::array<Reg, 16> midHiBits;
  std::array<Reg, 16> highestBits;
  // What the polynomial evaluates to at `eval_point`
  // I.e. the "primary" output of this op
  WomReg evaluation;
  // Padding to make output land where Poseidon expects it
  std::array<Reg, 6> padding;
  // The final state @85 to match Poseidon
  Poseidon2Cells output;
  // Temporaries to hold powers of eval so they don't go past degree
  FpExtReg power2;
  FpExtReg power4;
  FpExtReg power10;
};
using CheckedBytes = Comp<CheckedBytesImpl>;

} // namespace zirgen::recursion
