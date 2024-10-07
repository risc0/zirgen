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

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "zirgen/Dialect/ZStruct/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen::ZStruct {

namespace {

struct GetBufferPattern : public OpRewritePattern<ZStruct::GetBufferOp> {
  GetBufferPattern(MLIRContext* ctx, StringRef bufName) : OpRewritePattern(ctx), bufName(bufName) {}

  LogicalResult matchAndRewrite(GetBufferOp getBufOp, PatternRewriter& rewriter) const final {
    if (getBufOp.getName() != bufName)
      return failure();

    auto funcOp = getBufOp->getParentOfType<FunctionOpInterface>();
    if (!funcOp)
      return failure();

    // Search any existing arguments for one with the same name as the buffer
    Value bufArg;
    for (auto [argIdx, arg] : llvm::enumerate(funcOp.getArguments())) {
      auto argNameAttr = funcOp.getArgAttrOfType<StringAttr>(argIdx, "zirgen.argName");
      if (!argNameAttr || argNameAttr != bufName)
        continue;
      assert(!bufArg && "Duplicate arg name?");
      bufArg = arg;
    }

    if (!bufArg) {
      // No argument existing; add one
      rewriter.modifyOpInPlace(funcOp, [&]() {
        auto argIdx = funcOp.getNumArguments();
        funcOp.insertArgument(argIdx,
                              getBufOp.getType(),
                              rewriter.getDictionaryAttr({rewriter.getNamedAttr(
                                  "zirgen.argName", rewriter.getStringAttr(bufName))}),
                              getBufOp.getLoc());
        bufArg = funcOp.getFunctionBody().getArgument(argIdx);
      });

      // Update all call sites adding a zstruct.get_buffer to get the relevant buffer to pass.
      auto uses = SymbolTable::getSymbolUses(funcOp, funcOp->getParentOp());
      for (auto symbolUse : *uses) {
        Operation* callOp = symbolUse.getUser();
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(callOp);
        auto callBufOp =
            rewriter.create<GetBufferOp>(callOp->getLoc(), getBufOp.getType(), bufName);
        rewriter.modifyOpInPlace(callOp, [&]() {
          callOp->insertOperands(callOp->getOperands().size(), ValueRange{callBufOp});
        });
      }
    }

    rewriter.replaceOp(getBufOp, bufArg);
    return success();
  }

  StringRef bufName;
};

struct BuffersToArgsPass : public BuffersToArgsBase<BuffersToArgsPass> {
  void runOnOperation() override {
    auto bufs = Zll::lookupModuleAttr<Zll::BuffersAttr>(getOperation());

    for (auto bufDesc : bufs.getBuffers()) {
      RewritePatternSet patterns(&getContext());
      patterns.add<GetBufferPattern>(&getContext(), bufDesc.getName());
      if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)).failed()) {
        getOperation()->emitError("Unable to apply buffers to args patterns");
        signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createBuffersToArgsPass() {
  return std::make_unique<BuffersToArgsPass>();
}

} // namespace zirgen::ZStruct
