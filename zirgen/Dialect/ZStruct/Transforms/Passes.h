// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace zirgen::ZStruct {

// Pass constructors
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createOptimizeLayoutPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createUnrollPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createExpandLayoutPass();
std::unique_ptr<mlir::Pass> createInlineLayoutPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createStripAliasLayoutOpsPass();

void getUnrollPatterns(mlir::RewritePatternSet& patterns, mlir::MLIRContext* ctx);

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "zirgen/Dialect/ZStruct/Transforms/Passes.h.inc"

} // namespace zirgen::ZStruct
