// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h"

#include "PassDetail.h"
#include "zirgen/Dialect/Zll/IR/IR.h"

#include <iostream>

using namespace mlir;

namespace zirgen::Zll {

namespace {

struct SplitStagePass : public SplitStageBase<SplitStagePass> {
  SplitStagePass(unsigned stage) { this->stage = stage; }
  void runOnOperation() override {
    auto func = getOperation();
    func.insertResult(0, ValType::get(&getContext(), kFieldPrimeDefault, 1), DictionaryAttr());
    Block* block = &func.front();
    unsigned curStage = 0;
    Value result;
    for (Operation& op : llvm::make_early_inc_range(block->without_terminator())) {
      if (auto bop = dyn_cast<BarrierOp>(op)) {
        if (curStage == stage) {
          result = bop.getIn();
        }
        curStage++;
        bop.erase();
      } else if (curStage != stage && op.getNumResults() == 0) {
        op.erase();
      }
    }
    if (!result) {
      func->emitError("Invalid stage number");
      signalPassFailure();
      return;
    }
    block->getTerminator()->erase();
    auto builder = OpBuilder::atBlockEnd(block);
    builder.create<func::ReturnOp>(func.getLoc(), result);
  }
};

} // End namespace

std::unique_ptr<OperationPass<func::FuncOp>> createSplitStagePass() {
  return std::make_unique<SplitStagePass>(0);
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createSplitStagePass(unsigned stage) {
  return std::make_unique<SplitStagePass>(stage);
}

} // namespace zirgen::Zll
