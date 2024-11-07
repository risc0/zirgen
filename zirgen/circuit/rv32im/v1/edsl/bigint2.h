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

#include "zirgen/circuit/rv32im/v1/edsl/decode.h"
#include "zirgen/circuit/rv32im/v1/edsl/ecall.h"
#include "zirgen/circuit/rv32im/v1/edsl/global.h"
#include "zirgen/circuit/rv32im/v1/platform/constants.h"
#include "zirgen/components/ram.h"

namespace zirgen::rv32im_v1 {

class TopImpl;
using Top = Comp<TopImpl>;

class BigInt2CycleImpl : public CompImpl<BigInt2CycleImpl> {
public:
  // Constructor
  BigInt2CycleImpl(RamHeader ramHeader);

  // Helpers
  void setByte(Val v, size_t i);
  Val getByte(size_t i);

  // Called during data phase
  void set(Top top);

  // Called during accum phase
  void onAccum();

  // Registers for reading from and writing to RAM.
  RamBody ram;
  RamReg readInst;
  RamReg readRegAddr;
  std::array<RamReg, 4> io;
  OneHot<7> polyOp;
  OneHot<3> memOp;
  Reg isLast;
  Reg offset;
  Reg instWordAddr;
  std::array<Bit, 5> checkReg;
  std::array<Bit, 3> checkCoeff;
  std::array<ByteReg, 13> bytes;
  std::array<TwitByteReg, 3> twitBytes;
  FpExtReg mix;
  FpExtReg poly;
  FpExtReg term;
  FpExtReg tot;
  FpExtReg tmp;
};
using BigInt2Cycle = Comp<BigInt2CycleImpl>;

} // namespace zirgen::rv32im_v1
