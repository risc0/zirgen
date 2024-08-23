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
#include "zirgen/circuit/recursion/wom.h"

namespace zirgen::recursion {

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
