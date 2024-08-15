// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/PassDetail.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"

using namespace mlir;

namespace zirgen::Zll {

namespace {

struct RemoveEqualZero : public OpRewritePattern<EqualZeroOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(EqualZeroOp op, PatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct DropConstraintsPass : public DropConstraintsBase<DropConstraintsPass> {
  void runOnOperation() override {
    auto ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<RemoveEqualZero>(ctx);
    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)).failed()) {
      return signalPassFailure();
    }
  }
};

} // End namespace

std::unique_ptr<OperationPass<mlir::func::FuncOp>> createDropConstraintsPass() {
  return std::make_unique<DropConstraintsPass>();
}

} // namespace zirgen::Zll
