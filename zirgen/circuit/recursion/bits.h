// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

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
