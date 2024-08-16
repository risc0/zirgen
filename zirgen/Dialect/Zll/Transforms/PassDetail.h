// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/Pass/Pass.h"

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"

namespace zirgen {

#define GEN_PASS_CLASSES
#include "zirgen/Dialect/Zll/Transforms/Passes.h.inc"

} // namespace zirgen
