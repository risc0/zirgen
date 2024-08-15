// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/Pass/Pass.h"

#include "zirgen/Dialect/Zll/Conversion/ZStructToZll/Passes.h"
#include "zirgen/Dialect/Zll/IR/IR.h"

namespace zirgen::ZStructToZll {

#define GEN_PASS_CLASSES
#include "zirgen/Dialect/Zll/Conversion/ZStructToZll/Passes.h.inc"

} // namespace zirgen::ZStructToZll
