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

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen::Zhlt {

namespace {

template <typename TargetFuncOpT>
struct RewriteFuncPattern : public OpInterfaceRewritePattern<FunctionOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(FunctionOpInterface funcOp, PatternRewriter& rewriter) const final {
    if (llvm::isa<TargetFuncOpT>(funcOp))
      return failure();

    auto newFunc = rewriter.create<TargetFuncOpT>(
        funcOp.getLoc(),
        SymbolTable::getSymbolName(funcOp),
        llvm::cast<FunctionType>(funcOp.getFunctionType()),
        funcOp->getAttrOfType<StringAttr>(SymbolTable::getVisibilityAttrName()),
        funcOp.getAllArgAttrs(),
        funcOp.getAllResultAttrs());

    rewriter.modifyOpInPlace(newFunc, [&]() {
      // Propagate argument names, if present.
      if (auto asmOp = llvm::dyn_cast<OpAsmOpInterface>(*funcOp)) {
        llvm::DenseMap<Value, /*index=*/size_t> argIndexes;
        for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
          argIndexes[arg] = idx;
        }

        asmOp.getAsmBlockArgumentNames(funcOp.getFunctionBody(), [&](Value val, StringRef name) {
          if (!argIndexes.contains(val))
            return;
          auto argNum = argIndexes.at(val);
          if (!newFunc.getArgAttr(argNum, "zirgen.argName")) {
            newFunc.setArgAttr(argNum, "zirgen.argName", rewriter.getStringAttr(name));
          }
        });
      }
    });

    newFunc.getBody().takeBody(funcOp.getFunctionBody());
    rewriter.eraseOp(funcOp);
    return success();
  }
};

template <typename TargetCallOpT>
struct RewriteCallPattern : public OpInterfaceRewritePattern<CallOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CallOpInterface callOp, PatternRewriter& rewriter) const final {
    if (llvm::isa<TargetCallOpT>(callOp))
      return failure();

    auto targetSym = llvm::dyn_cast_if_present<FlatSymbolRefAttr>(
        llvm::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee()));
    if (!targetSym)
      return failure();
    auto target = SymbolTable::lookupNearestSymbolFrom<FunctionOpInterface>(callOp, targetSym);
    if (!target)
      return failure();

    rewriter.replaceOpWithNewOp<TargetCallOpT>(callOp,
                                               callOp->getResultTypes(),
                                               targetSym,
                                               TypeAttr::get(target.getFunctionType()),
                                               callOp.getArgOperands(),
                                               ArrayAttr(),
                                               ArrayAttr());
    return success();
  }
};

struct LowerStepFuncsPass : public LowerStepFuncsBase<LowerStepFuncsPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RewriteFuncPattern<StepFuncOp>>(&getContext());
    patterns.add<RewriteCallPattern<StepCallOp>>(&getContext());
    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed()) {
      getOperation().emitError("Unable to apply function patterns");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLowerStepFuncsPass() {
  return std::make_unique<LowerStepFuncsPass>();
}

} // namespace zirgen::Zhlt
