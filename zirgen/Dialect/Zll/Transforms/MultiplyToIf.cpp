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

struct MultiplyToIfPass : public MultiplyToIfBase<MultiplyToIfPass> {
  DenseMap</*factor=*/Value, /*count=*/SmallVector<EqualZeroOp>> factorConstraints;
  SmallVector</*factor=*/Value> factors;
  DenseMap<std::pair</*factor=*/Value, /* containing block */ Block*>, IfOp> ifOps;
  DenseSet<EqualZeroOp> needsReconstruction;

  void countFactors(EqualZeroOp eqzOp, Value factor) {
    auto mulOp = factor.getDefiningOp<MulOp>();
    if (mulOp) {
      countFactors(eqzOp, mulOp.getLhs());
      countFactors(eqzOp, mulOp.getRhs());
    } else {
      if (!factorConstraints.contains(factor))
        // Make sure we get a deterministic order that doesn't depend on comparing pointer addresses
        factors.push_back(factor);
      factorConstraints[factor].push_back(eqzOp);
    }
  }

  void sinkConstraint(EqualZeroOp op, Value factor) {
    if (needsReconstruction.contains(op)) {
      op.getInMutable().set(factor);
      needsReconstruction.erase(op);
      return;
    }

    Block* b = op->getBlock();

    auto& ifOp = ifOps[std::make_pair(factor, b)];
    if (!ifOp) {
      OpBuilder builder = OpBuilder::atBlockTerminator(b);
      ifOp = builder.create<IfOp>(factor.getLoc(), factor);
      builder.createBlock(&ifOp.getInner());
      builder.create<Zll::TerminateOp>(op.getLoc());
    }

    Block* targetBlock = &ifOp.getInner().front();
    op->moveBefore(targetBlock, targetBlock->without_terminator().end());
  }

  void runOnOperation() override {
    getOperation()->walk([&](EqualZeroOp eqzOp) {
      needsReconstruction.insert(eqzOp);
      countFactors(eqzOp, eqzOp.getIn());
    });

    // Highest priority is to make if statements for factors which are the most used, so put those
    // first.
    llvm::stable_sort(factors, [&](auto a, auto b) {
      return factorConstraints.at(a).size() > factorConstraints.at(b).size();
    });
    for (Value factor : factors) {
      //      llvm::errs() << "factor " << factor << " count " <<
      //      factorConstraints.at(factor).size()
      //                   << "\n";
      for (EqualZeroOp eqzOp : factorConstraints.at(factor))
        sinkConstraint(eqzOp, factor);
    }

    assert(needsReconstruction.empty());
  }
};

} // End namespace

std::unique_ptr<Pass> createMultiplyToIfPass() {
  return std::make_unique<MultiplyToIfPass>();
}

} // namespace zirgen::Zll
