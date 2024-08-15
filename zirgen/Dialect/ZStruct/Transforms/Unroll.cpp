// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZStruct/IR/Types.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/PassDetail.h"

#include <set>
#include <vector>

using namespace mlir;

namespace zirgen::ZStruct {
namespace {

struct UnrollMaps : public OpRewritePattern<MapOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MapOp op, PatternRewriter& rewriter) const final {
    Value in = op.getArray();
    auto inType = mlir::cast<ZStruct::ArrayType>(in.getType());
    auto outType = mlir::cast<ZStruct::ArrayType>(op.getOut().getType());

    llvm::SmallVector<Value, 8> mapped;

    Block& innerBlock = op.getBody().front();
    auto innerValArg = innerBlock.getArgument(0);
    Value innerLayoutArg;
    if (innerBlock.getNumArguments() > 1)
      innerLayoutArg = innerBlock.getArgument(1);

    auto yieldOp = llvm::cast<ZStruct::YieldOp>(innerBlock.getTerminator());
    auto innerReturnVal = yieldOp.getValue();

    for (size_t i = 0; i < inType.getSize(); i++) {
      IRMapping mapping;
      Value idx = rewriter.create<Zll::ConstOp>(op.getLoc(), i);
      Value inVal = rewriter.create<ZStruct::SubscriptOp>(op.getLoc(), in, idx);
      mapping.map(innerValArg, inVal);

      if (op.getLayout()) {
        Value inLayout = rewriter.create<ZStruct::SubscriptOp>(op.getLoc(), op.getLayout(), idx);
        assert(innerLayoutArg);
        mapping.map(innerLayoutArg, inLayout);
      }

      for (auto& innerOp : innerBlock.without_terminator()) {
        rewriter.clone(innerOp, mapping);
      }
      mapped.push_back(mapping.lookup(innerReturnVal));
    }
    rewriter.replaceOpWithNewOp<ZStruct::ArrayOp>(op, outType, mapped);
    return success();
  }
};

struct UnrollReduces : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op, PatternRewriter& rewriter) const final {
    Value inArray = op.getArray();
    auto inType = mlir::cast<ZStruct::ArrayType>(inArray.getType());

    Block& innerBlock = op.getBody().front();
    auto innerLhsArg = innerBlock.getArgument(0);
    auto innerRhsArg = innerBlock.getArgument(1);
    Value innerLayoutArg;
    if (innerBlock.getNumArguments() > 2)
      innerLayoutArg = innerBlock.getArgument(2);

    auto yieldOp = llvm::cast<ZStruct::YieldOp>(innerBlock.getTerminator());
    auto innerReturnVal = yieldOp.getValue();
    assert(innerReturnVal.getType() == innerLhsArg.getType());

    Value reduced = op.getInit();
    for (size_t i = 0; i < inType.getSize(); i++) {
      IRMapping mapping;
      mapping.map(innerLhsArg, reduced);

      Value idx = rewriter.create<Zll::ConstOp>(op.getLoc(), i);
      Value inVal = rewriter.create<ZStruct::SubscriptOp>(op.getLoc(), inArray, idx);
      mapping.map(innerRhsArg, inVal);

      if (op.getLayout()) {
        assert(innerLayoutArg);
        Value inLayout = rewriter.create<ZStruct::SubscriptOp>(op.getLoc(), op.getLayout(), idx);
        mapping.map(innerLayoutArg, inLayout);
      }

      for (auto& innerOp : innerBlock.without_terminator()) {
        rewriter.clone(innerOp, mapping);
      }
      reduced = mapping.lookup(innerReturnVal);
    }
    rewriter.replaceOp(op, reduced);
    return success();
  }
};

struct UnrollPass : public UnrollBase<UnrollPass> {
  void runOnOperation() override {
    auto* ctx = &getContext();
    auto op = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<UnrollMaps>(ctx);
    patterns.insert<UnrollReduces>(ctx);
    if (applyPatternsAndFoldGreedily(op, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // End namespace

std::unique_ptr<OperationPass<ModuleOp>> createUnrollPass() {
  return std::make_unique<UnrollPass>();
}

void getUnrollPatterns(RewritePatternSet& patterns, MLIRContext* ctx) {
  patterns.insert<UnrollMaps>(ctx);
  patterns.insert<UnrollReduces>(ctx);
}

} // namespace zirgen::ZStruct
