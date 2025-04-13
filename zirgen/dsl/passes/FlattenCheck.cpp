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

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/BindLayouts.h"
#include "zirgen/Dialect/ZStruct/Analysis/BufferAnalysis.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/RewritePatterns.h"
#include "zirgen/dsl/passes/CommonRewrites.h"
#include "zirgen/dsl/passes/PassDetail.h"

using namespace mlir;
using namespace zirgen::ZStruct;
using namespace zirgen::Zll;

namespace zirgen {
namespace dsl {

namespace {

// Inline zhlt.constructs for use in "check" functions.
struct InlineCheckConstruct : public OpRewritePattern<Zhlt::ComposableCheckCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Zhlt::ComposableCheckCallOp op, PatternRewriter& rewriter) const final {
    Zhlt::ComposableCheckFuncOp callable =
        op->getParentOfType<ModuleOp>().lookupSymbol<Zhlt::ComposableCheckFuncOp>(op.getCallee());
    if (!callable)
      return rewriter.notifyMatchFailure(op, "failed to resolve symbol " + op.getCallee());

    IRMapping mapping;
    Region clonedBody;
    callable.getBody().cloneInto(&clonedBody, mapping);
    remapInlinedLocations(clonedBody.getBlocks(), op.getLoc());
    Block* block = &clonedBody.front();
    auto returnOp = cast<Zhlt::ReturnOp>(block->getTerminator());

    rewriter.inlineBlockBefore(block, op, op.getOperands());
    rewriter.replaceOp(op, returnOp->getOperands());
    rewriter.eraseOp(returnOp);
    return success();
  }
};

struct FlattenCheckPass : public FlattenCheckBase<FlattenCheckPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<InlineCheckConstruct>(ctx);
    patterns.insert<BackToCall>(ctx);
    patterns.insert<EraseOp<StoreOp>>(ctx);
    patterns.insert<EraseOp<VariadicPackOp>>(ctx);
    patterns.insert<EraseOp<ExternOp>>(ctx);
    patterns.insert<EraseOp<AliasLayoutOp>>(ctx);
    patterns.insert<EraseOp<Zhlt::MagicOp>>(ctx);
    patterns.insert<InlineCalls>(ctx);
    patterns.insert<SplitSwitchArms>(ctx);
    patterns.insert<ReplaceYieldWithTerminator>(ctx);
    ZStruct::SwitchOp::getCanonicalizationPatterns(patterns, ctx);
    ZStruct::getUnrollPatterns(patterns, ctx);
    Zll::EqualZeroOp::getCanonicalizationPatterns(patterns, ctx);

    // Only try these if nothing else work, since they cause a lot of duplication.
    patterns.insert<UnravelSwitchPackResult>(ctx, /*benefit=*/0);
    patterns.insert<UnravelSwitchArrayResult>(ctx, /*benefit=*/0);
    patterns.insert<UnravelSwitchValResult>(ctx, /*benefit=*/0);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    // Now, inline everything and get rid of everything that's not a constraint.
    GreedyRewriteConfig config;
    config.maxIterations = 100;
    CheckFuncOp checkFuncOp = getOperation();
    if (applyPatternsGreedily(checkFuncOp, frozenPatterns, config).failed()) {
      checkFuncOp->emitError("Could not generate check function");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<Zhlt::CheckFuncOp>> createFlattenCheckPass() {
  return std::make_unique<FlattenCheckPass>();
}

} // namespace dsl
} // namespace zirgen
