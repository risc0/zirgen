// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace zirgen::Zll {

// Pass constructors
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createComputeTapsPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createMakePolynomialPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createSplitStagePass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createDropConstraintsPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createSplitStagePass(unsigned stage);
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createInlineFpExtPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createAddReductionsPass();
std::unique_ptr<mlir::Pass> createIfToMultiplyPass();
std::unique_ptr<mlir::Pass> createMultiplyToIfPass();
std::unique_ptr<mlir::Pass> createBalancedSplitPass(size_t maxOps = 1000);
std::unique_ptr<mlir::Pass> createSortForReproducibilityPass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "zirgen/Dialect/Zll/Transforms/Passes.h.inc"

} // namespace zirgen::Zll
