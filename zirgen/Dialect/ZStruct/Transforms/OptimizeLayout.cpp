// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
