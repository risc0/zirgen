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

#include "zirgen/circuit/rv32im/v1/edsl/bigint.h"
#include "zirgen/circuit/rv32im/v1/edsl/bigint2.h"
#include "zirgen/circuit/rv32im/v1/edsl/compute.h"
#include "zirgen/circuit/rv32im/v1/edsl/divide.h"
#include "zirgen/circuit/rv32im/v1/edsl/ecall.h"
#include "zirgen/circuit/rv32im/v1/edsl/global.h"
#include "zirgen/circuit/rv32im/v1/edsl/memio.h"
#include "zirgen/circuit/rv32im/v1/edsl/multiply.h"
#include "zirgen/circuit/rv32im/v1/edsl/page_fault.h"
#include "zirgen/circuit/rv32im/v1/edsl/sha.h"
#include "zirgen/components/ram.h"

#include <deque>

namespace zirgen::rv32im_v1 {

class TopImpl;
using Top = Comp<TopImpl>;

// The PC register must be between 0 and 2^28
class PCRegImpl : public CompImpl<PCRegImpl> {
public:
  PCRegImpl();

  // Set from single value
  void set(Val val, size_t offset = 4);

  // Get as a single value
  Val get();

  // Get as U32
  U32Val getU32();

  // Validate for user mode
  void checkValid(Val userMode);

private:
  std::vector<ByteReg> bytes;
  std::vector<Twit> twits;
  Reg buffer;
};
using PCReg = Comp<PCRegImpl>;

class ResetStepImpl : public CompImpl<ResetStepImpl> {
public:
  ResetStepImpl(BytesHeader bytesHeader);
  void set(Top top);

private:
  Global global;
  BytesBody bytes;
  TwitPrepare<kNumBodyTwits> twits;
  RamHeader ramHeader;
  PCReg pc;      // Program counter
  Bit userMode;  // User mode is set
  Reg nextMajor; // Next cycle major opcode, or MajorType::kMuxSize if decode
  std::array<Reg, MajorType::kMuxSize> padding;
  RamBody ram;
  std::array<RamReg, kDigestWords / 2> imageIdWrites;
  Reg writeAddr;
  OneHot<HaltType::kMuxSize> haltType;
  Reg sysExitCode;
  ByteReg verifyPC;
};
using ResetStep = Comp<ResetStepImpl>;

class HaltCycleImpl : public CompImpl<HaltCycleImpl> {
public:
  HaltCycleImpl(RamHeader ramHeader);
  void set(Top top);

  RamPass ram;
  Reg sysExitCode{Label("sys_exit_code")};
  Reg userExitCode{Label("user_exit_code")};
  Reg writeAddr{Label("write_addr")};
};
using HaltCycle = Comp<HaltCycleImpl>;

class BodyStepImpl : public CompImpl<BodyStepImpl> {
public:
  BodyStepImpl(BytesHeader bytesHeader);
  void set(Top top);

  Global global;
  BytesBody bytes;
  TwitPrepare<kNumBodyTwits> twits;
  RamHeader ramHeader;
  PCReg pc{Label("pc")};
  Bit userMode{Label("user_mode")};
  Reg nextMajor{Label("next_major")};
  OneHot<MajorType::kMuxSize> majorSelect{Label("major_select")};
  Mux<ComputeWrap<0>,
      ComputeWrap<1>,
      ComputeWrap<2>,
      MemIOCycle,
      MultiplyCycle,
      DivideCycle,
      VerifyAndCycle,
      VerifyDivideCycle,
      ECallCycle,
      ShaWrap<MajorType::kShaInit>,
      ShaWrap<MajorType::kShaLoad>,
      ShaWrap<MajorType::kShaMain>,
      PageFaultCycle,
      ECallCopyInCycle,
      BigIntCycle,
      BigInt2Cycle,
      HaltCycle>
      majorMux;
};
using BodyStep = Comp<BodyStepImpl>;

} // namespace zirgen::rv32im_v1
