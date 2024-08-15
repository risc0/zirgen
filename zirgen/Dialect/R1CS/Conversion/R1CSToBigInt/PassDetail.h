// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/Pass/Pass.h"

#include "zirgen/Dialect/R1CS/Conversion/R1CSToBigInt/Passes.h"
#include "zirgen/Dialect/R1CS/IR/R1CS.h"

namespace zirgen::R1CSToBigInt {

#define GEN_PASS_CLASSES
#include "zirgen/Dialect/R1CS/Conversion/R1CSToBigInt/Passes.h.inc"

} // namespace zirgen::R1CSToBigInt
