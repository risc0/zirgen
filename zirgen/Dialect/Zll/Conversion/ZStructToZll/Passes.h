// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace zirgen::ZStructToZll {

// Pass constructors
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createLowerCompositesPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "zirgen/Dialect/Zll/Conversion/ZStructToZll/Passes.h.inc"

} // namespace zirgen::ZStructToZll
