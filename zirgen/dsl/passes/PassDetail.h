// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/Pass/Pass.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/dsl/passes/Passes.h"

namespace zirgen {

#define GEN_PASS_CLASSES
#include "zirgen/dsl/passes/Passes.h.inc"

} // namespace zirgen
