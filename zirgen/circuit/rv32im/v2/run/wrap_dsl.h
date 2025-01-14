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

#include "risc0/core/util.h"
#include "zirgen/circuit/rv32im/v2/emu/preflight.h"
#include "zirgen/circuit/rv32im/v2/emu/trace.h"

namespace zirgen::rv32im_v2 {

struct StepHandler {
  virtual std::pair<uint32_t, uint32_t> getMajorMinor() = 0;
  virtual MemoryTransaction getMemoryTxn(uint32_t addr) = 0;
  virtual uint32_t readPrepare(uint32_t fd, uint32_t size) = 0;
  virtual std::vector<uint8_t> readBytes(uint32_t count) = 0;
  virtual uint32_t write(uint32_t fd, uint32_t addr, uint32_t size) = 0;
  virtual std::vector<uint32_t> nextPagingIdx() = 0;
  virtual void lookupDelta(risc0::Fp table, risc0::Fp index, risc0::Fp count) = 0;
  virtual risc0::Fp lookupCurrent(risc0::Fp table, risc0::Fp index) = 0;
  virtual void memoryDelta(uint32_t cycle, uint32_t addr, uint32_t data, risc0::Fp count) = 0;
  virtual uint32_t getDiffCount(uint32_t cycle) = 0;
};

CircuitParams getDslParams();

size_t getCycleCol();
size_t getTopStateCol();
size_t getEcall0StateCol();
size_t getPoseidonStateCol();
size_t getShaStateCol();

void DslStep(StepHandler& stepHandler, ExecutionTrace& trace, size_t cycle);
void DslStepAccum(StepHandler& stepHandler, ExecutionTrace& trace, size_t cycle);

} // namespace zirgen::rv32im_v2
