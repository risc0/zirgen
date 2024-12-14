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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace zirgen::Zhlt {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createHoistAllocsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createStripTestsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateStepsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createStripAliasLayoutOpsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createLowerStepFuncsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createBuffersToArgsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAnalyzeBuffersPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createOptimizeParWitgenPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createOutlineIfsPass();

#define GEN_PASS_REGISTRATION
#include "zirgen/Dialect/ZHLT/Transforms/Passes.h.inc"

} // namespace zirgen::Zhlt
