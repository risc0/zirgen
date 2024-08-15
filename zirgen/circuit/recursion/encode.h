// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
