// Copyright 2025 RISC Zero, Inc.
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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/Analysis/MixPowerAnalysis.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen::Zll {

namespace {

ValType getExtValType(MLIRContext* ctx) {
  return ValType::getExtensionType(ctx);
}

struct AndEqzPattern : public OpRewritePattern<AndEqzOp> {
  AndEqzPattern(MLIRContext* ctx, MixPowAnalysis& mixPows, Value mixPowBuf)
      : OpRewritePattern(ctx), mixPows(mixPows), mixPowBuf(mixPowBuf) {}

  LogicalResult matchAndRewrite(AndEqzOp op, PatternRewriter& rewriter) const override {
    auto mixPow = rewriter.create<GetGlobalOp>(op.getLoc(), mixPowBuf, mixPows.getMixPowIndex(op));
    auto mulOp = rewriter.create<MulOp>(op.getLoc(), op.getVal(), mixPow);
    auto addOp =
        rewriter.create<AddOp>(op.getLoc(), getExtValType(getContext()), op.getIn(), mulOp);
    rewriter.replaceOp(op, addOp);
    return mlir::success();
  }

  MixPowAnalysis& mixPows;
  Value mixPowBuf;
};

struct AndCondPattern : public OpRewritePattern<AndCondOp> {
  AndCondPattern(MLIRContext* ctx, MixPowAnalysis& mixPows, Value mixPowBuf)
      : OpRewritePattern(ctx), mixPows(mixPows), mixPowBuf(mixPowBuf) {}

  LogicalResult matchAndRewrite(AndCondOp op, PatternRewriter& rewriter) const override {
    auto mixPow = rewriter.create<GetGlobalOp>(op.getLoc(), mixPowBuf, mixPows.getMixPowIndex(op));
    auto mulOp = rewriter.create<MulOp>(
        op.getLoc(), getExtValType(getContext()), op.getInner(), op.getCond());
    auto mulOp2 = rewriter.create<MulOp>(op.getLoc(), getExtValType(getContext()), mulOp, mixPow);
    rewriter.replaceOpWithNewOp<AddOp>(op, getExtValType(getContext()), op.getIn(), mulOp2);
    return mlir::success();
  }

  MixPowAnalysis& mixPows;
  Value mixPowBuf;
};

struct TruePattern : public OpRewritePattern<TrueOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(TrueOp op, PatternRewriter& rewriter) const override {
    auto constOp =
        rewriter.create<ConstOp>(op.getLoc(), PolynomialAttr::get(op.getContext(), {0, 0, 0, 0}));
    rewriter.replaceOp(op, constOp);
    return mlir::success();
  }
};

struct ExtractPolyMixPass : public ExtractPolyMixBase<ExtractPolyMixPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto& mixPows = getAnalysis<MixPowAnalysis>();

    auto extValType = ValType::getExtensionType(&getContext());
    OpBuilder builder(&getContext());
    Type mixPowBufType = builder.getType<BufferType>(
        extValType, mixPows.getPowersNeeded().size(), BufferKind::Global);
    auto argAttrs = builder.getDictionaryAttr(
        builder.getNamedAttr("zirgen.argName", builder.getStringAttr("poly_mix")));
    funcOp.insertArgument(funcOp.getArguments().size(), mixPowBufType, argAttrs, funcOp.getLoc());
    Value mixPowBuf = funcOp.getBody().front().getArguments().back();

    RewritePatternSet patterns(&getContext());
    patterns.insert<AndEqzPattern>(&getContext(), mixPows, mixPowBuf);
    patterns.insert<AndCondPattern>(&getContext(), mixPows, mixPowBuf);
    patterns.insert<TruePattern>(&getContext());

    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)).failed()) {
      return signalPassFailure();
    }
    AttrTypeReplacer replacer;
    replacer.addReplacement([&](ConstraintType ty) { return extValType; });
    replacer.recursivelyReplaceElementsIn(
        getOperation(), /*replaceAttrs=*/true, /*replaceLocs=*/false, /*replaceTypes=*/true);
  }
};

struct AnnotatePolyMixPass : public AnnotatePolyMixBase<AnnotatePolyMixPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    auto& mixPows = getAnalysis<MixPowAnalysis>();

    auto extValType = ValType::getExtensionType(&getContext());
    OpBuilder builder(&getContext());
    Type mixPowBufType = builder.getType<BufferType>(
        extValType, mixPows.getPowersNeeded().size(), BufferKind::Global);
    auto argAttrs = builder.getDictionaryAttr(
        builder.getNamedAttr("zirgen.argName", builder.getStringAttr("poly_mix")));
    funcOp.insertArgument(funcOp.getArguments().size(), mixPowBufType, argAttrs, funcOp.getLoc());

    getOperation()->walk([&](AndEqzOp op) {
      assert(!op->hasAttr("zirgen.mixPowerIndex"));
      op->setAttr("zirgen.mixPowerIndex", builder.getIndexAttr(mixPows.getMixPowIndex(op)));
    });
    getOperation()->walk([&](AndCondOp op) {
      assert(!op->hasAttr("zirgen.mixPowerIndex"));
      op->setAttr("zirgen.mixPowerIndex", builder.getIndexAttr(mixPows.getMixPowIndex(op)));
    });
  }
};

} // End namespace

std::unique_ptr<mlir::OperationPass<func::FuncOp>> createExtractPolyMixPass() {
  return std::make_unique<ExtractPolyMixPass>();
}

std::unique_ptr<mlir::OperationPass<func::FuncOp>> createAnnotatePolyMixPass() {
  return std::make_unique<AnnotatePolyMixPass>();
}

} // namespace zirgen::Zll
