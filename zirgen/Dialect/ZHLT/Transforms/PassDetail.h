// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/Pass/Pass.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/Passes.h"

namespace zirgen::Zhlt {

#define GEN_PASS_CLASSES
#include "zirgen/Dialect/ZHLT/Transforms/Passes.h.inc"

} // namespace zirgen::Zhlt
