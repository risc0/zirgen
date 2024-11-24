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

#include <deque>
#include <iostream>

#include "risc0/core/elf.h"
#include "risc0/core/util.h"
#include "zirgen/circuit/keccak2/cpp/run.h"

namespace zirgen::keccak2 {

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

} // namespace zirgen::keccak2
