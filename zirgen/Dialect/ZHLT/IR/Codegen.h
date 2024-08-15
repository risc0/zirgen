// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"

namespace zirgen::Zhlt {

// Generates code for a ZHLT module, including extern traits, type definitions, etc.
mlir::LogicalResult emitModule(mlir::ModuleOp module, zirgen::codegen::CodegenEmitter& cg);

} // namespace zirgen::Zhlt
