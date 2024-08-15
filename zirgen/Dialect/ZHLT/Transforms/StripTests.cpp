// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "llvm/ADT/DenseSet.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"

#include <set>
#include <vector>

using namespace mlir;

namespace zirgen::Zhlt {

namespace {
struct StripTestsPass : public StripTestsBase<StripTestsPass> {
  void runOnOperation() override {
    DenseSet<Operation*> toErase;
    for (auto& op : getOperation().getOps()) {
      if (auto symName = SymbolTable::getSymbolName(&op)) {
        if (symName.strref().starts_with("test$") || symName.strref().contains("$test"))
          toErase.insert(&op);
      }
    }

    for (auto op : toErase)
      op->erase();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createStripTestsPass() {
  return std::make_unique<StripTestsPass>();
}

} // namespace zirgen::Zhlt
