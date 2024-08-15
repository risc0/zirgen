// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/IR/BuiltinOps.h"

namespace zirgen::dsl {

// Displays some circuit-wide statistics for the given module.
void printStats(mlir::ModuleOp moduleOp);

} // namespace zirgen::dsl
