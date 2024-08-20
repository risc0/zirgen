// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/Transforms/PassDetail.h"
#include "zirgen/Dialect/BigInt/Transforms/Passes.h"

using namespace mlir;

namespace zirgen::BigInt {

namespace {

struct ReplaceModularInv : public OpRewritePattern<ModularInvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ModularInvOp op, PatternRewriter& rewriter) const override {
    // Construct the constant 1
    mlir::Type oneType = rewriter.getIntegerType(1);    // a `1` is bitwidth 1
    auto oneAttr = rewriter.getIntegerAttr(oneType, 1); // value 1
    auto one = rewriter.create<ConstOp>(op.getLoc(), oneAttr);

    auto inv = rewriter.create<NondetInvModOp>(op.getLoc(), op.getLhs(), op.getRhs());
    auto remult = rewriter.create<MulOp>(op.getLoc(), op.getLhs(), inv);
    auto reduced = rewriter.create<ReduceOp>(op.getLoc(), remult, op.getRhs());
    auto diff = rewriter.create<SubOp>(op.getLoc(), reduced, one);
    rewriter.create<EqualZeroOp>(op.getLoc(), diff);
    rewriter.replaceOp(op, inv);
    return success();
  }
};

struct LowerModularInvPass : public LowerModularInvBase<LowerModularInvPass> {
  void runOnOperation() override {
    auto ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceModularInv>(ctx);
    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)).failed()) {
      return signalPassFailure();
    }
  }
};

} // End namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLowerModularInvPass() {
  return std::make_unique<LowerModularInvPass>();
}

} // namespace zirgen::BigInt
