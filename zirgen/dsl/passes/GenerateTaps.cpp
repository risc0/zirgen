// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/Analysis/BufferAnalysis.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "zirgen/dsl/passes/PassDetail.h"

using namespace mlir;
using namespace zirgen::ZStruct;

namespace zirgen {
namespace dsl {

// The requirements for GenerateTapsPass are:
//  * Layouts must be generated
//
//  * The circuit-wide CheckFuncOp completely unrolled, in that both the "ref" and
//  "back" arguments to any ZStruct::Load must be evaluatable as constants.
struct GenerateTapsPass : public GenerateTapsBase<GenerateTapsPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto ctx = module.getContext();
    OpBuilder builder(ctx);
    Location loc = builder.getUnknownLoc();
    auto bufferAnalysis = getAnalysis<ZStruct::BufferAnalysis>();

    DenseMap<std::pair</*buffer=*/StringAttr, /*offset=*/size_t>, /*backs=*/DenseSet<size_t>>
        namedTaps;

    auto walkResult = module.walk([&](Zhlt::CheckFuncOp check) {
      Zll::Interpreter interp(ctx);

      auto res = check->walk([&](LoadOp op) {
        auto ref = interp.evaluateConstantOfType<BoundLayoutAttr>(op.getRef());
        if (!ref) {
          op->emitError() << "Ref must be a constant";
          return WalkResult::interrupt();
        }
        auto distanceAttr = interp.evaluateConstantOfType<IntegerAttr>(op.getDistance());
        if (!distanceAttr) {
          op->emitError() << "distance must be a constant";
          return WalkResult::interrupt();
        }
        size_t distance = getIndexVal(distanceAttr);
        namedTaps[std::make_pair(ref.getBuffer(), llvm::cast<RefAttr>(ref.getLayout()).getIndex())]
            .insert(distance);
        return WalkResult::advance();
      });
      if (res.wasInterrupted())
        return WalkResult::interrupt();
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }

    SmallVector<Attribute> taps;
    for (auto [regGroupId, tapBuf] : llvm::enumerate(bufferAnalysis.getTapBuffers())) {
      for (size_t offset = 0; offset != tapBuf.regCount; ++offset) {
        SmallVector<size_t> backs =
            llvm::to_vector(namedTaps.lookup(std::make_pair(tapBuf.name, offset)));
        if (backs.empty()) {
          // Fill in any holes so that the register list is contiguous
          // TODO: We shouldn't have to do this, either by verifying there aren't any holes,
          // or by not depending on there being no holes.
          backs.push_back(0);
        }
        llvm::sort(backs);
        for (size_t back : backs) {
          taps.push_back(builder.getAttr<Zll::TapAttr>(regGroupId, offset, back));
        }
      }
    }

    builder.setInsertionPointToStart(module.getBody());

    Type tapArrayType = builder.getType<ArrayType>(builder.getType<TapType>(), taps.size());

    builder.create<GlobalConstOp>(
        loc, Zhlt::getTapsConstName(), tapArrayType, ArrayAttr::get(ctx, taps));
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateTapsPass() {
  return std::make_unique<GenerateTapsPass>();
}

} // namespace dsl
} // namespace zirgen
