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
#include "llvm/ADT/DenseSet.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/BindLayouts.h"
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
        bindLayoutOp =
            rewriter.create<ZStruct::BindLayoutOp>(funcOp.getLoc(),
                                                   constOp.getType(),
                                                   SymbolRefAttr::get(constOp.getSymNameAttr()),
                                                   getBufferOp);
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

// Creates a StepFuncOp for each circuit entry point.
//
// These functions represent the actual user-accessible "entry point" to the
// circuit witness generation procedure(s). They are simple wrappers around a
// call to the ExecFuncOp for that entry point, but also resolve the required
// buffers (typically 'data' and/or 'accum'). It is possible to inline, unroll,
// and flatten a StepFuncOp such that all "structure" disappears and we're left
// with a fixed-size blob of buffers, offsets, loads, stores, and basic
// computations.
struct GenerateStepsPass : public GenerateStepsBase<GenerateStepsPass> {
  void runOnOperation() override {
    SmallVector<CheckFuncOp> checkFuncs;

    getOperation().walk([&](ComponentOp component) {
      if (Zhlt::isEntryPoint(component)) {
        addStep(component);
      }
    });

    RewritePatternSet patterns(&getContext());
    addAttachGlobalLayoutPattern(patterns, "global");
    addAttachGlobalLayoutPattern(patterns, "mix");
    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed()) {
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
      auto bufs = Zll::lookupModuleAttr<Zll::BuffersAttr>(layoutOp);
      patterns.add<AttachGlobalLayoutPattern>(
          &getContext(), /*benefit=*/1, layoutOp, name, bufs.getBuffer(name).getType());
    }
  }

  void addStep(ComponentOp component) {
    Location loc = component.getLoc();
    OpBuilder builder(component);
    auto bufferAnalysis = getAnalysis<ZStruct::BufferAnalysis>();

    auto stepOp = builder.create<StepFuncOp>(
        loc, ("step$" + component.getName()).str(), builder.getFunctionType({}, {}));

    builder.setInsertionPointToStart(stepOp.addEntryBlock());
    if (failed(bindLayoutsForEntryPoint<ExecCallOp>(component, builder, bufferAnalysis))) {
      signalPassFailure();
    }
    builder.create<Zhlt::ReturnOp>(loc);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createGenerateStepsPass() {
  return std::make_unique<GenerateStepsPass>();
}

} // namespace zirgen::Zhlt
