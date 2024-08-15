// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/components/bits.h"
#include "zirgen/components/reg.h"

namespace zirgen {

class IsZeroImpl : public CompImpl<IsZeroImpl> {
public:
  // Sets a register value
  Val set(Val val);
  // Checks whether the register value is 0
  Val isZero();

private:
  // Records whether the register value is 0
  Bit isZeroBit;
  // Records the finite field multiplicative inverse of the register value
  Reg invVal;
};

using IsZero = Comp<IsZeroImpl>;

} // namespace zirgen
