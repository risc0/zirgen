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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/Transforms/ViewOpGraph.h"
#include "llvm/Support/Debug.h"
#include <limits>
#include <memory>

namespace zirgen {

#define GEN_PASS_DECL_EMITRECURSION
#include "zirgen/compiler/codegen/Passes.h.inc"

/// Creates a pass which outputs recursion predicates to .zkr files
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createEmitRecursionPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createEmitRecursionPass(llvm::StringRef outputDir);

#define GEN_PASS_REGISTRATION
#include "zirgen/compiler/codegen/Passes.h.inc"

} // namespace zirgen
