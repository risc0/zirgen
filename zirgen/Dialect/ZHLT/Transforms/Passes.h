// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace zirgen::Zhlt {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createHoistAllocsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createStripTestsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateStepsPass();

#define GEN_PASS_REGISTRATION
#include "zirgen/Dialect/ZHLT/Transforms/Passes.h.inc"

} // namespace zirgen::Zhlt
