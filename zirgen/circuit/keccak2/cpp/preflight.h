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

#include "zirgen/circuit/keccak2/cpp/trace.h"

namespace zirgen::keccak2 {

struct ScatterInfo {
  uint32_t dataOffset; // Place to get the data from (as u32 words)
  uint32_t row;        // Cycle # to write to
  uint16_t column;     // Column number to start at
  uint16_t count;      // Number of words to write
  uint16_t bitPerElem; // How many bits per element
};

struct PreflightTrace {
  // All the preimages
  std::vector<std::array<uint64_t, 25>> preimages;
  // Which 'preimage' each cycle is working on (to answer extern calls)
  std::vector<uint32_t> curPreimage;
  // Raw data for scattering
  std::vector<uint32_t> data;
  // Where to scatter it
  std::vector<ScatterInfo> scatter;
};

PreflightTrace preflightSegment(const std::vector<KeccakState>& inputs, size_t cycles);
void applyPreflight(ExecutionTrace& exec, const PreflightTrace& preflight);

} // namespace zirgen::keccak2
