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
