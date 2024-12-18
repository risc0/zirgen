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

#include "llvm/Support/Casting.h"

#include "risc0/fp/fpext.h"
#include "zirgen/circuit/rv32im/v2/emu/exec.h"

namespace zirgen::rv32im_v2 {

struct MemoryTransaction {
  uint32_t word;
  uint32_t cycle;
  uint32_t val;
  uint32_t prevCycle;
  uint32_t prevVal;
};

enum class CycleType : uint32_t {
  NOP,         // A cycle to ignore (maybe a continuation from hashing)
  INSTRUCTION, // A normal instruction
  CONTROL,     // A control instruction
  TABLE,
};

struct PreflightCycle {
  uint32_t state;
  uint32_t pc;
  uint8_t major;
  uint8_t minor;
  uint8_t machineMode;
  uint8_t padding;
  uint32_t memCycle;
  uint32_t userCycle;
  uint32_t extraPtr;
  uint32_t diffCount[2];
};

struct PreflightTrace {
  std::vector<PreflightCycle> cycles;
  std::vector<MemoryTransaction> txns;
  std::vector<uint32_t> extra;
  uint32_t tableSplitCycle;
  risc0::FpExt rng;
};

PreflightTrace preflightSegment(const Segment& in, size_t segmentSize);

} // namespace zirgen::rv32im_v2
