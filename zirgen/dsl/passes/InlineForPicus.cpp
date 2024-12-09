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

#include "Passes.h"
#include "mlir/Analysis/CallGraph.h"
// #include "mlir/Transforms/Inliner.h"

#include "zirgen/dsl/passes/PassDetail.h"

using namespace mlir;

namespace zirgen {
namespace dsl {

namespace {

struct InlineForPicusPass : public InlineForPicusBase<InlineForPicusPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    CallGraph &cg = getAnalysis<CallGraph>();

    // By default, assume that any inlining is profitable.
    auto profitabilityCb = [=](const Inliner::ResolvedCall &call) {
      return isProfitableToInline(call, 0);
    };

    // Get an instance of the inliner.
    Inliner inliner(op, cg, *this, getAnalysisManager(), runPipelineHelper,
                    config, profitabilityCb);

  //   // Run the inlining.
  //   if (failed(inliner.doInlining()))
  //     signalPassFailure();
  //   return;
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createInlineForPicusPass() {
  return std::make_unique<InlineForPicusPass>();
}

} // namespace dsl
} // namespace zirgen
