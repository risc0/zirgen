// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/rv32im/v1/edsl/body.h"
#include "zirgen/circuit/rv32im/v1/edsl/code.h"

#include <deque>

namespace zirgen::rv32im_v1 {

class TopImpl;
using Top = Comp<TopImpl>;

class BytesInitStepImpl : public CompImpl<BytesInitStepImpl> {
public:
  BytesInitStepImpl(BytesHeader bytesHeader);
  void set(Top top);

  BytesInit bytes;
};
using BytesInitStep = Comp<BytesInitStepImpl>;

class BytesSetupStepImpl : public CompImpl<BytesSetupStepImpl> {
public:
  BytesSetupStepImpl(BytesHeader bytesHeader);
  void set(Top top);

  BytesSetup bytes;
};
using BytesSetupStep = Comp<BytesSetupStepImpl>;

class RamInitStepImpl : public CompImpl<RamInitStepImpl> {
public:
  RamInitStepImpl(BytesHeader bytesHeader);
  void set(Top top);

  Global global;
  BytesBody bytes;
  TwitPrepare<kNumBodyTwits> twits;
  RamHeader ramHeader;
  RamInit ram;
};
using RamInitStep = Comp<RamInitStepImpl>;

class RamLoadStepImpl : public CompImpl<RamLoadStepImpl> {
public:
  RamLoadStepImpl(BytesHeader bytesHeader);
  void set(Top top);

  Global global;
  BytesBody bytes;
  TwitPrepare<kNumBodyTwits> twits;
  RamHeader ramHeader;
  std::array<Reg, MajorType::kMuxSize> padding;
  RamBody ram;
  std::array<ByteReg, 4 * kRamLoadStepIOCount> decode;
  std::array<RamReg, kRamLoadStepIOCount> writes;
};
using RamLoadStep = Comp<RamLoadStepImpl>;

class RamFiniStepImpl : public CompImpl<RamFiniStepImpl> {
public:
  RamFiniStepImpl(BytesHeader bytesHeader);
  void set(Top top);

  Global global;
  BytesBody bytes;
  TwitPrepare<kNumBodyTwits> twits;
  RamHeader ramHeader;
  RamFini ram;
};
using RamFiniStep = Comp<RamFiniStepImpl>;

class BytesFiniStepImpl : public CompImpl<BytesFiniStepImpl> {
public:
  BytesFiniStepImpl(BytesHeader bytesHeader);
  void set(Top top);

  BytesFini bytes;
};
using BytesFiniStep = Comp<BytesFiniStepImpl>;

class TopImpl : public CompImpl<TopImpl> {
public:
  TopImpl();
  // Returns 'Halted'
  Val set();

  Code code;
  BytesHeader bytesHeader;
  Reg halted;
  Mux<BytesInitStep,
      BytesSetupStep,
      RamInitStep,
      RamLoadStep,
      ResetStep,
      BodyStep,
      RamFiniStep,
      BytesFiniStep>
      mux;
};

} // namespace zirgen::rv32im_v1
