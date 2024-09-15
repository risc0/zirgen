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

#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"
#include "zirgen/Dialect/ZStruct/Analysis/BufferAnalysis.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"

#include <set>
#include <vector>

using namespace mlir;

namespace zirgen::Zhlt {

namespace {

#define GEN_PASS_DEF_MERGESTEPS
#include "zirgen/Dialect/ZHLT/Transforms/Passes.h.inc"

struct MergeStepsPass : public impl::MergeStepsBase<MergeStepsPass> {
  using MergeStepsBase<MergeStepsPass>::MergeStepsBase;

  void runOnOperation() override {
    OpBuilder builder = OpBuilder::atBlockEnd((getOperation().getBody()));
    auto loc = getOperation().getLoc();
    auto circuitDef = Zll::CircuitDefOp::lookupInModule(getOperation());

    SmallVector<Type> bufTypes;
    llvm::StringMap<size_t> bufArgIndex;
    for (auto [idx, bufDef] :
         llvm::enumerate(circuitDef.getBuffers().getAsRange<Zll::BufferDefAttr>())) {
      bufTypes.push_back(bufDef.getType());
      assert(!bufArgIndex.contains(bufDef.getName()));
      bufArgIndex[bufDef.getName()] = idx;
    }

    auto funcType = FunctionType::get(&getContext(), bufTypes, {builder.getType<Zll::ValType>()});
    auto funcOp = builder.create<func::FuncOp>(loc, "all_steps", funcType);
    builder.setInsertionPointToStart(funcOp.addEntryBlock());

    for (auto [idx, bufDef] :
         llvm::enumerate(circuitDef.getBuffers().getAsRange<Zll::BufferDefAttr>())) {
      funcOp.setArgAttr(idx, "zirgen.argName", builder.getStringAttr(bufDef.getName()));
    }

    // We're required to return a single value in many cases.
    // TODO: Formalize and document this requirement.
    auto zero = builder.create<Zll::ConstOp>(loc, 0);

    for (auto stepDef : circuitDef.getSteps().getAsRange<Zll::StepDefAttr>()) {
      SmallVector<Value> stepArgs;

      auto stepOp = getOperation().lookupSymbol<FunctionOpInterface>(stepDef.getName());
      if (!stepOp) {
        getOperation()->emitError()
            << "Could not find definition for step " << stepDef.getName() << "\n";
        signalPassFailure();
        return;
      }

      for (auto [stepArgIdx, stepArgType] : llvm::enumerate(stepOp.getArgumentTypes())) {
        auto name = stepOp.getArgAttrOfType<StringAttr>(stepArgIdx, "zirgen.argName");
        if (!name) {
          stepOp->emitError() << "Could not find name for argument " << stepArgIdx;
          signalPassFailure();
          return;
        }
        if (!bufArgIndex.contains(name)) {
          stepOp->emitError() << "References unknown buffer " << name << "\n";
          signalPassFailure();
          return;
        }
        stepArgs.push_back(funcOp.getArgument(bufArgIndex.at(name)));

        stepOp.setPrivate();
      }

      builder.create<mlir::func::CallOp>(loc, stepOp.getNameAttr(), stepOp.getResultTypes(), stepArgs);
    }

    builder.create<func::ReturnOp>(loc, ValueRange{zero});

    // Remove references to the individual step functions
    SmallVector<Attribute> newSteps;
    for (auto stepDef : circuitDef.getSteps().getAsRange<Zll::StepDefAttr>()) {
      newSteps.push_back(builder.getAttr<Zll::StepDefAttr>(stepDef.getName(), FlatSymbolRefAttr{}));
    }

    circuitDef.setStepsAttr(builder.getArrayAttr(newSteps));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createMergeStepsPass() {
  return std::make_unique<MergeStepsPass>();
}

} // namespace zirgen::Zhlt
