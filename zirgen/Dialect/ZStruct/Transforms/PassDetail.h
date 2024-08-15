// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/Pass/Pass.h"

#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/Passes.h"

namespace zirgen {

#define GEN_PASS_CLASSES
#include "zirgen/Dialect/ZStruct/Transforms/Passes.h.inc"

} // namespace zirgen
