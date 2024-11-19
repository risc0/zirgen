// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
  uint32_t diffCount;
};

struct PreflightTrace {
  std::vector<PreflightCycle> cycles;
  std::vector<MemoryTransaction> txns;
  std::vector<uint32_t> extra;
  uint32_t tableSplitCycle;
  risc0::FpExt rng;
};

PreflightTrace preflightSegment(const Segment& in);

} // namespace zirgen::rv32im_v2
