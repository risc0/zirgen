// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "risc0/core/util.h"
#include "zirgen/circuit/keccak2/cpp/trace.h"

namespace zirgen::keccak2 {

struct StepHandler {
};

CircuitParams getDslParams();
void DslStep(StepHandler& stepHandler, ExecutionTrace& trace, size_t cycle);

} // namespace zirgen::rv32im_v2
