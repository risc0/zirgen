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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen::Zll {

namespace {

struct RemoveIf : public OpRewritePattern<IfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(IfOp op, PatternRewriter& rewriter) const override {
    OpBuilder::InsertionGuard guard(rewriter);

    for (auto& bodyOp : llvm::make_early_inc_range(op.getInner().front().without_terminator())) {
      TypeSwitch<Operation&>(bodyOp)
          .Case<IfOp>([&](auto ifOp) {
            rewriter.setInsertionPoint(op);
            auto mulOp = rewriter.create<MulOp>(bodyOp.getLoc(), ifOp.getCond(), op.getCond());
            rewriter.modifyOpInPlace(ifOp, [&]() { ifOp.getCondMutable().set(mulOp); });
            rewriter.moveOpBefore(ifOp, op);
          })
          .Case<EqualZeroOp>([&](auto equalZeroOp) {
            rewriter.setInsertionPoint(op);
            auto mulOp = rewriter.create<MulOp>(bodyOp.getLoc(), equalZeroOp.getIn(), op.getCond());
            rewriter.modifyOpInPlace(equalZeroOp, [&]() { equalZeroOp.getInMutable().set(mulOp); });
          });
      rewriter.moveOpBefore(&bodyOp, op);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct IfToMultiplyPass : public IfToMultiplyBase<IfToMultiplyPass> {
  void runOnOperation() override {
    auto ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<RemoveIf>(ctx);
    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)).failed()) {
      return signalPassFailure();
    }
  }
};

} // End namespace

std::unique_ptr<Pass> createIfToMultiplyPass() {
  return std::make_unique<IfToMultiplyPass>();
}

} // namespace zirgen::Zll
