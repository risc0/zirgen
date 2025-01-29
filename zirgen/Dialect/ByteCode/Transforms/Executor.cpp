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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#include "zirgen/Dialect/ByteCode/Analysis/ArmAnalysis.h"
#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"
#include "zirgen/Dialect/ByteCode/Transforms/Passes.h"

#define DEBUG_TYPE "gen-executor"

using namespace mlir;

namespace zirgen::ByteCode {

#define GEN_PASS_DEF_GENEXECUTOR
#include "zirgen/Dialect/ByteCode/Transforms/Passes.h.inc"

namespace {

void buildArm(OpBuilder& builder,
              Location loc,
              Region& region,
              const ArmInfo* arm,
              const DenseMap<StringAttr, size_t>& funcArgs) {

  builder.createBlock(&region);

  SmallVector<Value> values;

  LLVM_DEBUG(llvm::dbgs() << "building arm: " << *arm << "\n");

  for (auto [argName, argType] : llvm::zip_equal(arm->funcArgNames, arm->getFuncArgTypes()))
    values.push_back(builder.create<GetArgumentOp>(loc, argType, argName));

  for (auto val : arm->getLoadVals()) {
    values.push_back(builder.create<LoadOp>(loc, val.getType()));
  }

  size_t offset = 0;
  Operation* newOp = nullptr;
  for (auto [op, numIntArgs] : llvm::zip_equal(arm->ops, arm->opIntArgs)) {
    SmallVector<Value> intArgs;
    for (auto i : llvm::seq(numIntArgs)) {
      (void)i;
      intArgs.push_back(builder.create<DecodeOp>(loc));
    }

    SmallVector<Value> operands;
    for (size_t valueOffset : ArrayRef(arm->valueOffsets).slice(offset, op->getNumOperands())) {
      operands.push_back(values[valueOffset]);
    }
    offset += op->getNumOperands();

    if (llvm::isa<func::ReturnOp>(op)) {
      builder.create<ExitOp>(loc, operands);
      return;
    } else if (numIntArgs) {
      newOp = builder.create<WrappedOp>(
          loc, op->getResultTypes(), op->getName().getIdentifier(), intArgs, operands);
    } else {
      OperationState state(loc, op->getName());
      state.addOperands(operands);
      state.addTypes(op->getResultTypes());
      newOp = builder.create(state);
    }

    llvm::append_range(values, newOp->getResults());
  }

  // Yield results from the last operation
  SmallVector<Value> operands;
  for (size_t valueOffset : ArrayRef(arm->valueOffsets).slice(offset)) {
    operands.push_back(values[valueOffset]);
  }
  builder.create<YieldOp>(loc, operands);
}

} // namespace

struct GenExecutorPass : public impl::GenExecutorBase<GenExecutorPass> {
  using GenExecutorBase::GenExecutorBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    ArmAnalysis& armAnalysis = getAnalysis<ArmAnalysis>();

    // Make sure we include all distinct operations so we can support anything that we want to
    // encode into bytecode.
    SmallVector<const ArmInfo*> arms;
    llvm::append_range(arms, llvm::make_pointer_range(armAnalysis.getDistinctOps()));

    // Take arms from multiOpArms based on total number of operations
    // among all instances until we fill up maxArms arms.
    SmallVector<std::pair<size_t, const ArmInfo*>> multiOpArms;
    llvm::append_range(multiOpArms,
                       llvm::map_range(armAnalysis.getMultiOpArms(), [&](const ArmInfo& armInfo) {
                         return std::make_pair(armInfo.ops.size(), &armInfo);
                       }));
    llvm::sort(multiOpArms, llvm::less_first());

    while (arms.size() < maxArms && !multiOpArms.empty()) {
      const ArmInfo* armInfo = multiOpArms.back().second;
      multiOpArms.pop_back();

      if (armInfo->count < minArmUse)
        continue;
      if (armInfo->numLoadVals > maxCapture)
        continue;
      if (armInfo->numYieldVals > maxYield)
        continue;

      /*      llvm::errs() << "Got arm " << *armInfo << " OK!\n";
            for (Operation* op : armInfo->ops) {
              llvm::errs() << "  " << *op << "\n";
              }*/
      arms.push_back(armInfo);
    }

    if (arms.empty()) {
      mod.emitError() << "Unable to find potential dispatch arms";
      signalPassFailure();
      return;
    }

    llvm::sort(multiOpArms, llvm::less_first());

    auto funcType = armAnalysis.getFunctionType();
    auto argNames = armAnalysis.getArgNames();

    DenseMap<StringAttr, /*index=*/size_t> argIndex;
    for (auto [idx, argName] : llvm::enumerate(argNames))
      argIndex[argName] = idx;

    OpBuilder builder = OpBuilder::atBlockBegin(mod.getBody());
    auto executeOp =
        builder.create<ExecutorOp>(builder.getUnknownLoc(),
                                   execSymbol,
                                   /*visibility=*/StringAttr(),
                                   funcType,
                                   builder.getArrayAttr(llvm::to_vector_of<Attribute>(argNames)),
                                   /*intKinds=*/ArrayAttr(),
                                   /*number of arms=*/arms.size());
    for (auto [idx, armInfo] : llvm::enumerate(arms)) {
      ++armCount;
      maxArmOps.updateMax(armInfo->ops.size());
      maxArmUses.updateMax(armInfo->count);
      maxArmReplaced.updateMax(armInfo->count * armInfo->ops.size());
      buildArm(builder, armInfo->loc, executeOp.getArms()[idx], armInfo, argIndex);
    }
  }
};

} // namespace zirgen::ByteCode
