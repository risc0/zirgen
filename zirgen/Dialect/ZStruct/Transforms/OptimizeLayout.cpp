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

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/Transforms/PassDetail.h"
#include "zirgen/compiler/layout/collect.h"
#include "zirgen/compiler/layout/convert.h"
#include "zirgen/compiler/layout/improve.h"
#include "zirgen/compiler/layout/rebuild.h"
#include "zirgen/compiler/layout/viz.h"

using namespace mlir;

namespace zirgen::ZStruct {

namespace {

// test:
//  bazelisk test //zirgen/Dialect/Zll/IR/test:layout.mlir.test
// run:
//  bazelisk run //zirgen/compiler/tools:zirgen-opt -- --optimize-layout
//    `pwd`/zirgen/Dialect/Zll/IR/test/layout.mlir

struct OptimizeLayoutPass : public OptimizeLayoutBase<OptimizeLayoutPass> {
  void runOnOperation() override;
};

} // namespace

void OptimizeLayoutPass::runOnOperation() {
  ModuleOp module = getOperation();
  // collect the layouts of the components in this circuit
  layout::Circuit circuit(module);
  // optimize the layout of the structs within unions
  layout::improve(circuit);
  // distribute the changes back into the type system
  auto replacements = layout::rebuild(circuit);
  // replace the original types with the improved versions
  if (layout::convert(module, replacements).failed()) {
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createOptimizeLayoutPass() {
  return std::make_unique<OptimizeLayoutPass>();
}

} // namespace zirgen::ZStruct
