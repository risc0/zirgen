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

#include "zirgen/circuit/recursion/code.h"

#include "llvm/ADT/TypeSwitch.h"

namespace zirgen::recursion {

enum HashType {
  SHA256,
  POSEIDON2,
  MIXED_POSEIDON2_SHA,
};

// Statistics gathered from encoding
struct EncodeStats {
  size_t totCycles = 0;
  size_t shaCycles = 0;
  size_t poseidon2Cycles = 0;
};

std::vector<uint32_t> encode(HashType hashType,
                             mlir::Block* block,
                             llvm::DenseMap<mlir::Value, uint64_t>* toIdReturn = nullptr,
                             EncodeStats* stats = nullptr);
std::vector<uint32_t> encode(HashType hashType, mlir::Block* block, EncodeStats* stats);

} // namespace zirgen::recursion
