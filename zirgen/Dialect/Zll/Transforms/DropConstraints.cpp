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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/PassDetail.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"

using namespace mlir;

namespace zirgen::Zll {

namespace {

struct RemoveEqualZero : public OpRewritePattern<EqualZeroOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(EqualZeroOp op, PatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct DropConstraintsPass : public DropConstraintsBase<DropConstraintsPass> {
  void runOnOperation() override {
    auto ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<RemoveEqualZero>(ctx);
    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed()) {
      return signalPassFailure();
    }
  }
};

} // End namespace

std::unique_ptr<OperationPass<mlir::func::FuncOp>> createDropConstraintsPass() {
  return std::make_unique<DropConstraintsPass>();
}

} // namespace zirgen::Zll
