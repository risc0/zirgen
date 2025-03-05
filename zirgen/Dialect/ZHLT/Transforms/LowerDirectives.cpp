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

// #include "mlir/IR/BuiltinDialect.h"
// #include "mlir/IR/BuiltinOps.h"
// #include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"

using namespace mlir;
namespace zirgen::Zhlt {

namespace {

struct LowerDirectives : public OpRewritePattern<DirectiveOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DirectiveOp directive, PatternRewriter& rewriter) const final {
    StringRef name = directive.getName();
    if (name == "AssumeRange" || name == "AssertRange") {
      // AssumeRange!(low, x, high);
      // -->
      // Assert(1 - InRange(low, x, high), "value out of range!");
      Location loc = directive.getLoc();
      OperandRange args = directive.getArgs();
      Value message = rewriter.create<Zll::StringOp>(loc, "value out of range!");
      Value one = rewriter.create<Zll::ConstOp>(loc, 1);
      Value cond = rewriter.create<Zll::InRangeOp>(loc, args[0], args[1], args[2]);
      cond = rewriter.create<Zll::SubOp>(loc, one, cond);
      rewriter.replaceOpWithNewOp<Zll::ExternOp>(
          directive, TypeRange{}, ValueRange{cond, message}, "Assert", /*extra=*/"");
      return success();
    } else if (name == "PicusInput" || name == "PicusHintEq") {
      // This is a no-op unless compiling with `--emit=picus`. Since this pass
      // is after picus emission in the compiler pipeline, we are in some other
      // compilation mode. Just erase it.
      rewriter.eraseOp(directive);
      return success();
    } else {
      return failure();
    }
  }
};

struct LowerDirectivesPass : public LowerDirectivesBase<LowerDirectivesPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerDirectives>(&getContext());
    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLowerDirectivesPass() {
  return std::make_unique<LowerDirectivesPass>();
}

} // namespace zirgen::Zhlt
