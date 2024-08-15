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

class MemIOCycleImpl : public CompImpl<MemIOCycleImpl> {
public:
  MemIOCycleImpl(RamHeader ramHeader);
  void set(Top top);

private:
  RamBody ram;
  RamReg readInst;
  Decoder decoder;
  U32Reg immReg;
  OneHot<kMinorMuxSize> minorSelect;
  RamReg readRS1;
  RamReg readRS2;
  IsZero rdZero;
  OneHot<4> lowBits;
  std::array<ByteReg, 4> carry;
  std::array<ByteReg, 3> bytes;
  ByteReg check6;
  std::array<Twit, 2> checkHigh;
  ByteReg highByte;
  ByteReg highBit;
  ByteReg lowBits2;
  U32Reg buffer;
  RamReg readMem;
  RamReg write;
};
using MemIOCycle = Comp<MemIOCycleImpl>;

} // namespace zirgen::rv32im_v1
