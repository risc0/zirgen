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
#include "zirgen/circuit/rv32im/v1/platform/constants.h"
#include "zirgen/components/ram.h"

namespace zirgen::rv32im_v1 {

class TopImpl;
using Top = Comp<TopImpl>;

class ShaCycleImpl : public CompImpl<ShaCycleImpl> {
public:
  ShaCycleImpl(size_t major, RamHeader ramHeader);
  void set(Top top);

private:
  size_t major;

  void setInit(Top top);
  void setLoad(Top top);
  void setMain(Top top);

  void computeW();
  void computeAE();
  void onVerify();

  RamBody ram;
  RamReg io0;
  RamReg io1;

  Reg stateOut;
  Reg stateIn;
  Reg data0;
  Reg data1;
  Reg count;
  IsZero countZero;
  Reg repeat;
  IsZero repeatZero;
  Bit minor;
  Bit finalStage;
  Bit mode; // 0: write result, 1: verify result
  Reg readOp;

  std::vector<Twit> twits;
  std::vector<ByteReg> bytes;

  std::vector<Bit> a;
  std::array<Reg, 2> aRaw;
  Twit aCarryLow;
  Twit aCarryHigh;

  std::vector<Bit> e;
  std::array<Reg, 2> eRaw;
  Twit eCarryLow;
  Twit eCarryHigh;

  std::vector<Bit> w;
  std::array<Reg, 2> wRaw;
  Twit wCarryLow;
  Twit wCarryHigh;
};

using ShaCycle = Comp<ShaCycleImpl>;

template <size_t major> struct ShaWrapImpl : public CompImpl<ShaWrapImpl<major>> {
  ShaWrapImpl(RamHeader ramHeader) : inner(major, ramHeader) {}
  void set(Top top) { inner->set(top); }
  ShaCycle inner;
};

template <size_t major> using ShaWrap = Comp<ShaWrapImpl<major>>;

} // namespace zirgen::rv32im_v1
