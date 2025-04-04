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

#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen {
namespace Zhlt {

namespace {

struct InlineRequiredPass : public InlineRequiredBase<InlineRequiredPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    CallGraph& cg = getAnalysis<CallGraph>();

    auto profitabilityCb = [=](const Inliner::ResolvedCall& call) {
      // Don't inline things that aren't constructor calls
      auto op = dyn_cast<Zhlt::ConstructOp>(call.call);
      if (!op)
        return false;

      // Inline any calls to components marked "inline" or "extern"
      Operation* target = call.targetNode->getCallableRegion()->getParentOp();
      return target->hasAttr("inline") || target->hasAttr("extern");
    };

    // Get an instance of the inliner.
    InlinerConfig config;
    auto runPipelineHelper = [](Pass& pass, OpPassManager& pipeline, Operation* op) {
      return mlir::cast<InlineRequiredPass>(pass).runPipeline(pipeline, op);
    };
    Inliner inliner(
        mod, cg, *this, getAnalysisManager(), runPipelineHelper, config, profitabilityCb);

    // Run the inlining.
    if (failed(inliner.doInlining()))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createInlineRequiredPass() {
  return std::make_unique<InlineRequiredPass>();
}

} // namespace Zhlt
} // namespace zirgen
