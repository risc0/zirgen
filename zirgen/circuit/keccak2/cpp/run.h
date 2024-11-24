// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include <string>

#include "zirgen/circuit/keccak2/cpp/trace.h"
#include "zirgen/circuit/keccak2/cpp/wrap_dsl.h"

namespace zirgen::keccak2 {

using KeccakState = std::array<uint64_t, 25>;
ExecutionTrace runSegment(const std::vector<KeccakState>& inputs);

} // namespace zirgen::keccak2
