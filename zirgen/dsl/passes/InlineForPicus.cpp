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

#include "Passes.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Inliner.h"

#include "zirgen/dsl/passes/PassDetail.h"

using namespace mlir;

namespace zirgen {
namespace dsl {

namespace {

// Picus inductively proves the determinism of a circuit: that is, assuming all
// previous cycles are deterministic, that the next cycle is deterministic. This
// means backs with non-zero distance read deterministic values, whereas backs
// with a distance of zero are only deterministic if that is otherwise provable.
// This rewrite pattern converts any zero distance Zhlt::BackOps into
// Zhlt::BackCallOps for subsequent inlining in order to express this fact.
struct ZeroDistanceBacksToCalls : public OpRewritePattern<Zhlt::BackOp> {
  using OpType = Zhlt::BackOp;
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Zhlt::BackOp back, PatternRewriter& rewriter) const {
    if (back.getDistance().getZExtValue() > 0)
      return failure();

    auto distance =
        rewriter.create<mlir::arith::ConstantOp>(back->getLoc(), rewriter.getIndexAttr(0));
    auto callee =
        SymbolTable::lookupNearestSymbolFrom<Zhlt::ComponentOp>(back, back.getCalleeAttr());
    auto callOp = rewriter.create<Zhlt::BackCallOp>(
        back->getLoc(), callee.getSymName(), callee.getOutType(), distance, back.getLayout());
    rewriter.replaceOp(back, callOp);
    return success();
  }
};

struct InlineForPicusPass : public InlineForPicusBase<InlineForPicusPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    ModuleOp mod = getOperation();

    // Convert backs with distance zero for inlining
    RewritePatternSet patterns(ctx);
    patterns.insert<ZeroDistanceBacksToCalls>(ctx);
    if (applyPatternsAndFoldGreedily(mod, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    CallGraph& cg = getAnalysis<CallGraph>();

    auto profitabilityCb = [=](const Inliner::ResolvedCall& call) {
      // Inline any calls to components marked "picus_inline"
      if (call.targetNode->getCallableRegion()->getParentOp()->hasAttr("picus_inline")) {
        return true;
      }

      // All BackCallOps come from backs with distance 0 because of the rewrite
      // pattern, and these should all be inlined so we can do a more detailed
      // analysis of their determinism.
      if (isa<Zhlt::BackCallOp>(call.call)) {
        return true;
      }

      // Inline a specific set of constructors
      auto op = cast<Zhlt::ConstructOp>(call.call);
      auto callee = op.getCallee();
      return callee == "Add" || callee == "BitAnd" || callee == "Component" || callee == "Div" ||
             callee == "InRange" || callee == "Inv" || callee == "Mul" || callee == "NondetReg" ||
             callee == "Sub" || callee == "Val" || callee == "Reg";
    };

    // Get an instance of the inliner.
    InlinerConfig config;
    auto runPipelineHelper = [](Pass& pass, OpPassManager& pipeline, Operation* op) {
      return mlir::cast<InlineForPicusPass>(pass).runPipeline(pipeline, op);
    };
    Inliner inliner(
        mod, cg, *this, getAnalysisManager(), runPipelineHelper, config, profitabilityCb);

    // Run the inlining.
    if (failed(inliner.doInlining()))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createInlineForPicusPass() {
  return std::make_unique<InlineForPicusPass>();
}

} // namespace dsl
} // namespace zirgen
