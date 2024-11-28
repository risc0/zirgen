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
#include "zirgen/circuit/keccak2/cpp/preflight.h"

#include <array>

namespace zirgen::keccak2 {

struct StepHandler {
  StepHandler(const PreflightTrace& trace, size_t cycle) : trace(trace), cycle(cycle) {}
  bool nextPreimage() { return trace.curPreimage[cycle] != trace.preimages.size(); }
  const std::array<uint64_t, 25>& getPreimage() {
    return trace.preimages[trace.curPreimage[cycle]];
  }

private:
  const PreflightTrace& trace;
  size_t cycle;
};

struct LayoutInfo {
  uint32_t bits;
  uint32_t sflat;
  uint32_t kflat;
  uint32_t control;
  uint32_t ctypeOneHot;
};

LayoutInfo getLayoutInfo();

CircuitParams getDslParams();
void DslStep(StepHandler& stepHandler, ExecutionTrace& trace, size_t cycle);

} // namespace zirgen::keccak2
