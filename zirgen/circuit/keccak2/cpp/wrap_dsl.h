// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "risc0/core/util.h"
#include "zirgen/circuit/keccak2/cpp/trace.h"

#include <array>

namespace zirgen::keccak2 {

using KeccakState = std::array<uint64_t, 25>;

struct StepHandler {
  StepHandler(const std::vector<KeccakState>& inputs) : inputs(inputs), idx(0) {}
  bool nextPreimage() {
    idx++;
    return idx < inputs.size();
  }
  const std::array<uint64_t, 25>& getPreimage() {
    return inputs[idx];
  }
private:
  std::vector<KeccakState> inputs;
  size_t idx = 0;
};

CircuitParams getDslParams();
void DslStep(StepHandler& stepHandler, ExecutionTrace& trace, size_t cycle);

} // namespace zirgen::rv32im_v2
