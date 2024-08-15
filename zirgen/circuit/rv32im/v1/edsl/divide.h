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

class VerifyDivideCycleImpl;

class DivideCycleImpl : public CompImpl<DivideCycleImpl> {
  friend class VerifyDivideCycleImpl;

public:
  DivideCycleImpl(RamHeader ramHeader);
  void set(Top top);

private:
  RamBody ram;
  RamReg readInst;
  Decoder decoder;
  OneHot<8> minorSelect;
  RamReg readRS1;
  RamReg readRS2;
  Twit top2;
  Bit bit6;
  U32Po2 po2;
  Bit isSigned;
  Bit onesComp;
  std::array<ByteReg, 4> denom;
  std::array<ByteReg, 4> quot;
  std::array<ByteReg, 4> rem;
  IsZero rdZero;
  RamReg writeRd;
};
using DivideCycle = Comp<DivideCycleImpl>;

class VerifyDivideCycleImpl : public CompImpl<VerifyDivideCycleImpl> {
public:
  VerifyDivideCycleImpl(RamHeader ramHeader);
  void set(Top top);

private:
  RamPass ram;
  TopBit numerTop;
  TopBit denomTop;
  Reg negNumer;
  Reg negDenom;
  Reg negQuot;
  U32Normalize numerAbs;
  U32Normalize denomAbs;
  IsZeroU32 denomZero;
  U32Normalize quotAbs;
  U32Normalize remAbs;
  U32Normalize denomRemCheck;
  U32MulAcc mul;
};
using VerifyDivideCycle = Comp<VerifyDivideCycleImpl>;

} // namespace zirgen::rv32im_v1
