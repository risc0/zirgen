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
#include "zirgen/circuit/rv32im/v1/edsl/global.h"
#include "zirgen/circuit/rv32im/v1/platform/constants.h"
#include "zirgen/components/ram.h"

namespace zirgen::rv32im_v1 {

class TopImpl;
using Top = Comp<TopImpl>;

struct ECallHaltImpl : public CompImpl<ECallHaltImpl> {
  void set(Top top);

  RamReg readA0; // sys_exit, user_exit, 0, 0
  RamReg readA1; // Addr of 'output'
  Reg writeAddr;
};
using ECallHalt = Comp<ECallHaltImpl>;

class ECallInputImpl : public CompImpl<ECallInputImpl> {
public:
  void set(Top top);

  RamReg readA0;
  RamReg writeA0;
  OneHot<kDigestWords> selector;
  U32Reg word;
};
using ECallInput = Comp<ECallInputImpl>;

class ECallSoftwareImpl : public CompImpl<ECallSoftwareImpl> {
public:
  void set(Top top);

  RamReg readOutputAddr;
  RamReg readOutputWords;

  // Number of chunks of 1-4 words to output.
  Reg outputChunks;
  // Number of words to output in first cycle; all other outputs are full chunks.
  Twit outputFirstCycleWordsMinusOne;

  ByteReg requireAlignedAddr;
  ByteReg requireAlignedBytes;
};
using ECallSoftware = Comp<ECallSoftwareImpl>;

class ECallShaImpl : public CompImpl<ECallShaImpl> {
public:
  void set(Top top);

  RamReg readA0;
  RamReg readA1;
  RamReg readA4;
};
using ECallSha = Comp<ECallShaImpl>;

class ECallBigIntImpl : public CompImpl<ECallBigIntImpl> {
public:
  void set(Top top);

  // a1 is the op selector.
  // Currently constrained to 0.
  RamReg readA1;
};
using ECallBigInt = Comp<ECallBigIntImpl>;

class ECallBigInt2Impl : public CompImpl<ECallBigInt2Impl> {
public:
  void set(Top top);

  // BigInt pc
  RamReg readVerifyAddr;
};
using ECallBigInt2 = Comp<ECallBigInt2Impl>;

// Jump to user mode
class ECallUserImpl : public CompImpl<ECallUserImpl> {
public:
  void set(Top top);

  RamReg readPC;
};
using ECallUser = Comp<ECallUserImpl>;

// All ECalls in user mode jump here
class ECallMachineImpl : public CompImpl<ECallMachineImpl> {
public:
  void set(Top top);

  RamReg writePC;
  RamReg readEntry;
};
using ECallMachine = Comp<ECallMachineImpl>;

class ECallCycleImpl : public CompImpl<ECallCycleImpl> {
public:
  ECallCycleImpl(RamHeader ramHeader);
  void set(Top top);

  RamBody ram;
  RamReg readInst;
  RamReg readSelector;
  OneHot<ECallType::kMuxSize + 1> minorSelect;
  Bit isTrap;
  Mux<ECallHalt,
      ECallInput,
      ECallSoftware,
      ECallSha,
      ECallBigInt,
      ECallUser,
      ECallBigInt2,
      ECallMachine>
      minorMux;
};
using ECallCycle = Comp<ECallCycleImpl>;

class TwitByteRegImpl : public CompImpl<TwitByteRegImpl> {
public:
  TwitByteRegImpl();
  void set(Val val);
  Val get();
  std::array<Twit, 4> twits;
};
using TwitByteReg = Comp<TwitByteRegImpl>;

class ECallCopyInCycleImpl : public CompImpl<ECallCopyInCycleImpl> {
public:
  ECallCopyInCycleImpl(RamHeader ramHeader);
  void set(Top top);

  RamBody ram;

  std::array<RamReg, kIoChunkWords> io;

  Reg chunksRemaining;
  Reg outputAddr;

  // Number of words to output this cycle; 0 means we're done and we
  // should fill a0 and a1 for returning to user.
  OneHot<kIoChunkWords + 1> outputWords;

  // Verifiers for all 16 bytes, last two
  std::array<ByteReg, 14> checkBytes;
  std::array<TwitByteReg, 2> checkBytesTwits;

  IsZero chunksRemainingZ;
};
using ECallCopyInCycle = Comp<ECallCopyInCycleImpl>;

} // namespace zirgen::rv32im_v1
