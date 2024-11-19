// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include <string>

#include "zirgen/circuit/rv32im/v2/emu/exec.h"
#include "zirgen/circuit/rv32im/v2/emu/trace.h"

namespace zirgen::rv32im_v2 {

ExecutionTrace runSegment(const Segment& segment);

} // namespace zirgen::rv32im_v2
