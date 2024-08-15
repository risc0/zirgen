// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"
#include "zirgen/Dialect/ZStruct/Analysis/BufferAnalysis.h"

#include <set>
#include <vector>

using namespace mlir;

namespace zirgen::Zhlt {

namespace {

struct AttachGlobalLayoutPattern : public OpInterfaceRewritePattern<FunctionOpInterface> {
  AttachGlobalLayoutPattern(MLIRContext* ctx,
                            PatternBenefit benefit,
                            ZStruct::GlobalConstOp globalLayoutOp,
                            StringRef bufferName,
                            Type globalBufferType)
      : OpInterfaceRewritePattern(ctx, benefit)
      , globalLayoutOp(globalLayoutOp)
      , bufferName(bufferName)
      , globalBufferType(globalBufferType) {}

  LogicalResult matchAndRewrite(FunctionOpInterface funcOp, PatternRewriter& rewriter) const final {
    // Instantiate global layout only once per function.
    ZStruct::BindLayoutOp bindLayoutOp;
    funcOp.walk([&](Zhlt::GetGlobalLayoutOp getGlobalOp) {
      ZStruct::GlobalConstOp constOp = globalLayoutOp;

      if (getGlobalOp.getBuffer() != bufferName)
        return;

      if (!bindLayoutOp) {
        rewriter.setInsertionPointToStart(&funcOp.getFunctionBody().front());
        StringRef buffer = getGlobalOp.getBuffer();
        auto getBufferOp =
            rewriter.create<ZStruct::GetBufferOp>(getGlobalOp.getLoc(), globalBufferType, buffer);
        bindLayoutOp = rewriter.create<ZStruct::BindLayoutOp>(
            funcOp.getLoc(), constOp.getType(), constOp.getSymName(), getBufferOp);
      }

      rewriter.setInsertionPoint(getGlobalOp);
      rewriter.replaceOpWithNewOp<ZStruct::LookupOp>(
          getGlobalOp, bindLayoutOp, getGlobalOp.getName());
    });

    if (bindLayoutOp)
      return success();
    else
      return failure();
  }

  ZStruct::GlobalConstOp globalLayoutOp;
  StringRef bufferName;
  Type globalBufferType;
};

struct GenerateStepsPass : public GenerateStepsBase<GenerateStepsPass> {
  void runOnOperation() override {
    SmallVector<CheckFuncOp> checkFuncs;

    getOperation().walk([&](ComponentOp component) {
      llvm::StringRef baseName = component.getName();
      if (baseName.starts_with("test$") || baseName.ends_with("$accum") || baseName == "Top") {
        addStep(component);
      }
    });

    RewritePatternSet patterns(&getContext());
    addAttachGlobalLayoutPattern(patterns, "global");
    addAttachGlobalLayoutPattern(patterns, "mix");
    if (applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)).failed()) {
      getOperation().emitError("Unable to apply layout patterns");
      signalPassFailure();
    }

    // All GetGlobalLayoutOps should now be attached to a buffer
    if (getOperation()
            .walk([&](GetGlobalLayoutOp op) { return WalkResult::interrupt(); })
            .wasInterrupted()) {
      getOperation().emitError("Unable to attach all global layouts");
      signalPassFailure();
    }
  }

  void addAttachGlobalLayoutPattern(RewritePatternSet& patterns, StringRef name) {
    auto layoutSymName = ZStruct::getLayoutConstName(name);
    auto layoutOp = getOperation().lookupSymbol<ZStruct::GlobalConstOp>(layoutSymName);
    if (layoutOp) {
      auto& bufferAnalysis = getAnalysis<ZStruct::BufferAnalysis>();
      auto bufferType = bufferAnalysis.getBuffer(name).getType(&getContext());
      patterns.add<AttachGlobalLayoutPattern>(
          &getContext(), /*benefit=*/1, layoutOp, name, bufferType);
    }
  }

  void addStep(ComponentOp component) {
    Location loc = component.getLoc();
    auto funcOp = component.getAspect<ExecFuncOp>();
    if (!funcOp) {
      component.emitError() << "Unable to find an exec function for top-level step";
      return signalPassFailure();
    }

    OpBuilder builder(component);

    auto stepOp = builder.create<StepFuncOp>(loc, component.getName());

    builder.setInsertionPointToStart(stepOp.addEntryBlock());

    llvm::SmallVector<Value> args;
    auto bufferAnalysis = getAnalysis<ZStruct::BufferAnalysis>();
    auto contextArg = builder.getBlock()->getArgument(0);

    for (auto execArg : funcOp.getBody().front().getArguments()) {
      if (execArg.getType() == contextArg.getType()) {
        args.push_back(contextArg);
      } else {
        auto [constOp, bufferDesc] = bufferAnalysis.getLayoutAndBufferForArgument(execArg);
        if (!constOp) {
          funcOp.emitError() << "Unable to find a value for argument " << execArg
                             << " to top-level step for component " << component.getName();
          return signalPassFailure();
        }
        auto getBufferOp = builder.create<ZStruct::GetBufferOp>(
            funcOp.getLoc(), bufferDesc.getType(&getContext()), bufferDesc.name);
        args.push_back(builder.create<ZStruct::BindLayoutOp>(
            funcOp.getLoc(), constOp.getType(), constOp.getSymName(), getBufferOp));
      }
    }

    mlir::FunctionType funcType = funcOp.getFunctionType();
    builder.create<ExecCallOp>(funcOp.getLoc(),
                               builder.getAttr<FlatSymbolRefAttr>(funcOp.getSymName()),
                               funcType,
                               funcOp.getInputSegmentSizes(),
                               funcOp.getResultSegmentSizes(),
                               args);

    builder.create<Zhlt::ReturnOp>(loc);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createGenerateStepsPass() {
  return std::make_unique<GenerateStepsPass>();
}

} // namespace zirgen::Zhlt
