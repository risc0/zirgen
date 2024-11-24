// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include <deque>
#include <iostream>

#include "risc0/core/elf.h"
#include "risc0/core/util.h"
#include "zirgen/circuit/keccak2/cpp/run.h"

namespace zirgen::keccak2{

ExecutionTrace runSegment(const std::vector<KeccakState>& inputs) {
  size_t cycles = 200;
  ExecutionTrace trace(cycles, getDslParams());
  StepHandler ctx(inputs);
  for (size_t i = 0; i < cycles; i++) {
    std::cout << "Running cycle " << i << "\n";
    DslStep(ctx, trace, i);
  }
  std::cout << "Done\n";
  return trace;
}

} // namespace zirgen::rv32im_v2
