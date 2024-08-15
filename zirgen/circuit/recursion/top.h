// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/circuit/recursion/checked_bytes.h"
#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/recursion/macro.h"
#include "zirgen/circuit/recursion/micro.h"
#include "zirgen/circuit/recursion/poseidon2.h"

namespace zirgen::recursion {

struct TopImpl : public CompImpl<TopImpl> {
  TopImpl();
  void set();

  Code code = Label("code");
  WomHeader womHeader;
  Mux<MicroOps,
      MacroOp,
      Poseidon2Load,
      Poseidon2Full,
      Poseidon2Partial,
      Poseidon2Store,
      CheckedBytes>
      mux;
};
using Top = Comp<TopImpl>;

} // namespace zirgen::recursion
