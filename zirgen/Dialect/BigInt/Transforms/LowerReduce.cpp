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

struct ReplaceReduce : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ReduceOp op, PatternRewriter& rewriter) const override {
    auto quotOp = rewriter.create<NondetQuotOp>(op.getLoc(), op.getLhs(), op.getRhs());
    auto remOp = rewriter.create<NondetRemOp>(op.getLoc(), op.getLhs(), op.getRhs());
    auto remult = rewriter.create<MulOp>(op.getLoc(), quotOp, op.getRhs());
    auto readd = rewriter.create<AddOp>(op.getLoc(), remult, remOp);
    auto diff = rewriter.create<SubOp>(op.getLoc(), readd, op.getLhs());
    rewriter.create<EqualZeroOp>(op.getLoc(), diff);
    rewriter.replaceOp(op, remOp);
    return success();
  }
};

struct LowerReducePass : public LowerReduceBase<LowerReducePass> {
  void runOnOperation() override {
    auto ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceReduce>(ctx);
    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)).failed()) {
      return signalPassFailure();
    }
  }
};

} // End namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLowerReducePass() {
  return std::make_unique<LowerReducePass>();
}

} // namespace zirgen::BigInt
