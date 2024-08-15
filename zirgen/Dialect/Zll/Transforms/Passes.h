// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace zirgen::Zll {

// Pass constructors
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createComputeTapsPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createMakePolynomialPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createMakeVerifyTapsPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createSplitStagePass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createDropConstraintsPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createSplitStagePass(unsigned stage);
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createInlineFpExtPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createAddReductionsPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "zirgen/Dialect/Zll/Transforms/Passes.h.inc"

} // namespace zirgen::Zll
