// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace zirgen {
namespace dsl {

// Pass constructors
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateBackPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateExecPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateLayoutPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateCheckPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateTapsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateValidityRegsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateValidityTapsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateAccumPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateGlobalsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createElideTrivialStructsPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "zirgen/dsl/passes/Passes.h.inc"

} // namespace dsl
} // namespace zirgen
