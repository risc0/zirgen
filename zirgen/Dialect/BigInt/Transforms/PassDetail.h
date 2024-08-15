// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/Pass/Pass.h"

#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/Transforms/Passes.h"

namespace zirgen {

#define GEN_PASS_CLASSES
#include "zirgen/Dialect/BigInt/Transforms/Passes.h.inc"

} // namespace zirgen
