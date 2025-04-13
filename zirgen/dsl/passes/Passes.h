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
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"

namespace zirgen {
namespace dsl {

using namespace mlir;
using namespace Zhlt;

// Pass constructors
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createEraseUnusedAspectsPass(bool forTests = false);
std::unique_ptr<OperationPass<ModuleOp>> createElideTrivialStructsPass();
std::unique_ptr<OperationPass<ModuleOp>> createFieldDCEPass();
std::unique_ptr<OperationPass<CheckFuncOp>> createFlattenCheckPass();
std::unique_ptr<OperationPass<ModuleOp>> createGenerateAccumPass();
std::unique_ptr<OperationPass<ModuleOp>> createGenerateBackPass();
std::unique_ptr<OperationPass<ModuleOp>> createGenerateCheckLayoutPass();
std::unique_ptr<OperationPass<ModuleOp>> createGenerateCheckPass();
std::unique_ptr<OperationPass<ModuleOp>> createGenerateExecPass();
std::unique_ptr<OperationPass<ModuleOp>> createGenerateGlobalsPass();
std::unique_ptr<OperationPass<ModuleOp>> createGenerateLayoutPass();
std::unique_ptr<OperationPass<ModuleOp>> createGenerateTapsPass();
std::unique_ptr<OperationPass<ModuleOp>> createGenerateValidityRegsPass();
std::unique_ptr<OperationPass<ModuleOp>> createGenerateValidityTapsPass();
std::unique_ptr<OperationPass<ModuleOp>> createHoistInvariantsPass();
std::unique_ptr<OperationPass<ModuleOp>> createInlineForPicusPass();
std::unique_ptr<OperationPass<ModuleOp>> createInlinePurePass();
std::unique_ptr<Pass> createTopologicalShufflePass();

// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "zirgen/dsl/passes/Passes.h.inc"

} // namespace dsl
} // namespace zirgen
