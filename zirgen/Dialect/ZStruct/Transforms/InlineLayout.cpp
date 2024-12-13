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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "zirgen/Dialect/ZStruct/IR/Types.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/PassDetail.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"

#include <set>
#include <vector>

using namespace mlir;

namespace zirgen::ZStruct {
namespace {

// Map of names to GetBufferOps referencing those names
using BufferMapT = DenseMap<Block*, DenseMap</*bufName=*/StringAttr, GetBufferOp>>;

struct InlineLayout : public OpRewritePattern<BindLayoutOp> {
  InlineLayout(MLIRContext* ctx, ModuleOp mod) : OpRewritePattern(ctx), mod(mod) {}

  LogicalResult matchAndRewrite(BindLayoutOp op, PatternRewriter& rewriter) const final {
    AttrTypeReplacer replacer;

    replacer.addReplacement([&](SymbolRefAttr symRef) -> mlir::Attribute {
      auto globConst = ModuleOp(mod).lookupSymbol<GlobalConstOp>(symRef);
      if (!globConst) {
        llvm::errs() << "Can't find " << symRef << "\n";
        return globConst.getConstantAttr();
      }
      return globConst.getConstantAttr();
    });

    auto replaced = replacer.replace(op.getLayoutAttr());
    if (replaced != op.getLayoutAttr()) {
      rewriter.modifyOpInPlace(op, [&]() { op.setLayoutAttr(replaced); });
      return success();
    }

    return failure();
  }

  ModuleOp mod;
};

struct InlineLayoutPass : public InlineLayoutBase<InlineLayoutPass> {
  void runOnOperation() override {
    ModuleOp mod = llvm::dyn_cast<ModuleOp>(getOperation());
    if (!mod)
      mod = getOperation()->getParentOfType<ModuleOp>();
    RewritePatternSet patterns(&getContext());
    patterns.add<InlineLayout>(&getContext(), mod);
    LookupOp::getCanonicalizationPatterns(patterns, &getContext());
    SubscriptOp::getCanonicalizationPatterns(patterns, &getContext());
    LoadOp::getCanonicalizationPatterns(patterns, &getContext());
    StoreOp::getCanonicalizationPatterns(patterns, &getContext());
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (applyPatternsAndFoldGreedily(getOperation(), frozenPatterns).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> createInlineLayoutPass() {
  return std::make_unique<InlineLayoutPass>();
}

} // namespace zirgen::ZStruct
