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

struct ReplaceInv : public OpRewritePattern<InvOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(InvOp op, PatternRewriter& rewriter) const override {
    // Construct the constant 1
    mlir::Type oneType = rewriter.getIntegerType(1);    // a `1` is bitwidth 1
    auto oneAttr = rewriter.getIntegerAttr(oneType, 1); // value 1
    auto one = rewriter.create<ConstOp>(op.getLoc(), oneAttr);

    auto inv = rewriter.create<NondetInvOp>(op.getLoc(), op.getLhs(), op.getRhs());
    auto remult = rewriter.create<MulOp>(op.getLoc(), op.getLhs(), inv);
    auto reduced = rewriter.create<ReduceOp>(op.getLoc(), remult, op.getRhs());
    auto diff = rewriter.create<SubOp>(op.getLoc(), reduced, one);
    rewriter.create<EqualZeroOp>(op.getLoc(), diff);
    rewriter.replaceOp(op, inv);
    return success();
  }
};

struct LowerInvPass : public LowerInvBase<LowerInvPass> {
  void runOnOperation() override {
    auto ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceInv>(ctx);
    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)).failed()) {
      return signalPassFailure();
    }
  }
};

} // End namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createLowerInvPass() {
  return std::make_unique<LowerInvPass>();
}

} // namespace zirgen::BigInt
