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
#include "zirgen/Dialect/ZStruct/Transforms/PassDetail.h"
#include "zirgen/Dialect/ZStruct/Transforms/RewritePatterns.h"

using namespace mlir;

namespace zirgen::ZStruct {
namespace {

struct UnrollPass : public UnrollBase<UnrollPass> {
  void runOnOperation() override {
    auto* ctx = &getContext();
    auto op = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<UnrollMaps>(ctx);
    patterns.insert<UnrollReduces>(ctx);
    if (applyPatternsGreedily(op, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // End namespace

std::unique_ptr<Pass> createUnrollPass() {
  return std::make_unique<UnrollPass>();
}

} // namespace zirgen::ZStruct
