// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"

#include "zirgen/Dialect/ZStruct/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen::ZStruct {

namespace {

struct StripAliasLayoutOpsPass : public StripAliasLayoutOpsBase<StripAliasLayoutOpsPass> {
  void runOnOperation() override {
    getOperation().walk([](AliasLayoutOp alias) { alias.erase(); });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createStripAliasLayoutOpsPass() {
  return std::make_unique<StripAliasLayoutOpsPass>();
}

} // namespace zirgen::ZStruct
