// Copyright 2025 RISC Zero, Inc.
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
#include "zirgen/circuit/recursion/wom.h"

namespace zirgen::recursion {

// Bit operations, tailored to implementation of the SHA-256 PRNG for Fiat-Shamir.
//
// Inputs:
// * operands[0] is the memory location for input A
// * operands[1] is the memory location for input B
// * operands[2] is a expected to be a bit selecting the operation.
//
// Preconditions:
// * A and B are expected to be of the form [x, y, 0, 0].
//   Any values in the upper limbs of the extension field element will be ignored.
// * The non-zero limbs of A and B are expected to be 16-bit values.
//   If they are out of the 16-bit range, the bit decomposition will result in a constraint failure.
//
// Output:
// * If operands[2] is 0, compute XOR: [a, b, 0, 0] ^ [c, d, 0, 0] -> [a ^ c, b ^ d, 0, 0]
// * If operands[2] is 1, compute AND + combine: [a, b, 0, 0] & [c, d, 0, 0] -> [(a & c) + ((b & d)
// << 16), 0, 0, 0]
struct BitOpShortsImpl : public CompImpl<BitOpShortsImpl> {
  BitOpShortsImpl(WomHeader header);
  void set(MacroInst inst, Val writeAddr);

  WomBody body;
  WomReg inA = Label("in_a");
  WomReg inB = Label("in_b");
  WomReg out = Label("out");

  std::array<Bit, 32> bitsA = Label("bits_a");
  std::array<Bit, 32> bitsB = Label("bits_b");
};
using BitOpShorts = Comp<BitOpShortsImpl>;

// Bitwise AND of the least significant limb of two extension field elements.
//
// Inputs:
// * operands[0] is the memory location for input A
// * operands[1] is the memory location for input B
// * operands[2] is unused.
//
// Preconditions:
// * A and B are expected to be of the form [x, 0, 0, 0].
//   I.e. they are expected to be some base field elements.
//   Any other limbs in the extension element will be ignored.
//
// Output:
//
// A base field elem equal to the bitwise AND of the standard, reduced representations of two base
// field inputs. [a, 0, 0, 0] & [b, 0, 0, 0] -> [a & b, 0, 0, 0]
struct BitAndElemImpl : public CompImpl<BitAndElemImpl> {
  BitAndElemImpl(WomHeader header);
  void set(MacroInst inst, Val writeAddr);

  WomBody body;
  WomReg inA = Label("in_a");
  WomReg inB = Label("in_b");
  WomReg out = Label("out");
  std::vector<Bit> bitsA;
  std::vector<Bit> bitsB;
};
using BitAndElem = Comp<BitAndElemImpl>;

} // namespace zirgen::recursion
