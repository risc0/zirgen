// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/circuit/rv32im/v1/edsl/decode.h"
#include "zirgen/circuit/rv32im/v1/platform/constants.h"
#include "zirgen/components/ram.h"

namespace zirgen::rv32im_v1 {

class TopImpl;
using Top = Comp<TopImpl>;

class MultiplyCycleImpl : public CompImpl<MultiplyCycleImpl> {
public:
  MultiplyCycleImpl(RamHeader ramHeader);
  void set(Top top);

private:
  RamBody ram;
  RamReg readInst;
  Decoder decoder;
  OneHot<kMinorMuxSize> minorSelect;
  RamReg readRS1;
  RamReg readRS2;
  Twit top2;
  Bit bit6;
  U32Po2 po2;
  U32Mul mul;
  IsZero rdZero;
  RamReg writeRd;
};
using MultiplyCycle = Comp<MultiplyCycleImpl>;

} // namespace zirgen::rv32im_v1
