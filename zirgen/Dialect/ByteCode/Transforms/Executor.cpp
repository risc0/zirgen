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
#include "zirgen/Dialect/Zll/IR/IR.h"

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
  for (auto [op, numIntArgs] : llvm::zip_equal(arm->getOps(), arm->opIntArgs)) {
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

struct BuildArmInfo {
  BuildArmInfo(const ArmInfo* armInfo) : armInfo(armInfo) { allOps = armInfo->allOps; }

  ssize_t getBenefit() const {
    if (true) {
      //      ssize_t benefit = 35;
      //      for (Operation* op : armInfo->getOps()) {
      //        if (llvm::isa<MulOp, AndEqzOp, AndCondOp>(op))
      //          benefit += 1;
      //      }
      //      ssize_t benefit = armInfo->values.size();
      //      ssize_t benefit = armInfo->getOps().size();
      ssize_t benefit = llvm::count_if(armInfo->getOps(), [&](auto op) {
        return !llvm::any_of(op->getResults(), [&](Value result) {
          auto valType = llvm::dyn_cast<Zll::ValType>(result.getType());
          if (valType && valType.getFieldK() > 1)
            return true;
          return llvm::isa<Zll::ConstraintType>(result.getType());
        });
      });

      // benefit -= armInfo->numLoadVals;
      benefit -= armInfo->numYieldVals;
      benefit *= armInfo->getCount() - 1;
      return benefit;
    } else {
      ssize_t benefit = armInfo->getOps().size() * armInfo->getOps().size();
      benefit -= armInfo->numLoadVals * armInfo->numYieldVals;
      // benefit *= armInfo->getCount() - 1;
      return benefit;
    }
  }

  bool operator<(const BuildArmInfo& rhs) const { return getBenefit() < rhs.getBenefit(); }
  const ArmInfo* armInfo;

  SmallVector<ArrayRef<Operation*>> allOps;
};

struct ArmFinder {
  ArmFinder(ArmAnalysis& armAnalysis) : armAnalysis(armAnalysis) {
    // Take arms from multiOpArms based on total number of operations
    // among all instances until we fill up maxArms arms.

    llvm::append_range(multiOpArms, llvm::make_pointer_range(armAnalysis.getMultiOpArms()));
    std::make_heap(multiOpArms.begin(), multiOpArms.end());
  }

  const ArmInfo* takeNextBest() {
    while (!multiOpArms.empty()) {
      std::pop_heap(multiOpArms.begin(), multiOpArms.end());
      BuildArmInfo& buildArmInfo = multiOpArms.back();

      auto it = llvm::remove_if(buildArmInfo.allOps, [&](ArrayRef<Operation*> ops) {
        return llvm::any_of(ops, [&](Operation* op) { return seen.contains(op); });
      });
      if (it != buildArmInfo.allOps.end()) {
        // Seen some of these already; throw it back in the queue
        buildArmInfo.allOps.erase(it, buildArmInfo.allOps.end());
        if (buildArmInfo.allOps.size() < 2)
          multiOpArms.pop_back();
        else
          std::push_heap(multiOpArms.begin(), multiOpArms.end());
        continue;
      }

      const ArmInfo* armInfo = buildArmInfo.armInfo;
      /*      if (true ) {
              if (armInfo->getCount() < minArmUse)
                continue;
              if (armInfo->numLoadVals > maxCapture)
                continue;
              if (armInfo->numYieldVals > maxYield)
                continue;
            }
      */
      for (ArrayRef<Operation*> ops : buildArmInfo.allOps) {
        seen.insert(ops.begin(), ops.end());
      }
      multiOpArms.pop_back();
      return armInfo;

      //      llvm::errs() << "benefit " << getBenefit(*armInfo) << " arm " << *armInfo << "\n";

      /*      llvm::errs() << "Got arm " << *armInfo << " OK!\n";
            for (Operation* op : armInfo->ops) {
              llvm::errs() << "  " << *op << "\n";
              }*/
      //      return armInfo;
    }
    assert(multiOpArms.empty());
    return nullptr;
  }

  ArmAnalysis& armAnalysis;
  SmallVector<BuildArmInfo> multiOpArms;
  DenseSet<Operation*> seen;
};

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

    ArmFinder armFinder(armAnalysis);

    while (arms.size() < maxArms) {
      const ArmInfo* armInfo = armFinder.takeNextBest();
      if (!armInfo)
        break;
      arms.push_back(armInfo);
    }
    if (arms.empty()) {
      mod.emitError() << "Unable to find potential dispatch arms";
      signalPassFailure();
      return;
    }

    auto funcType = armAnalysis.getFunctionType();
    auto argNames = armAnalysis.getArgNames();

    OpBuilder builder = OpBuilder::atBlockBegin(mod.getBody());
    DenseMap<StringAttr, /*index=*/size_t> argIndex;
    SmallVector<Attribute> argAttrs;
    for (auto [idx, argName] : llvm::enumerate(argNames)) {
      argIndex[argName] = idx;
      argAttrs.push_back(
          builder.getDictionaryAttr(builder.getNamedAttr("zirgen.argName", argName)));
    }

    auto executeOp = builder.create<ExecutorOp>(builder.getUnknownLoc(),
                                                execSymbol,
                                                funcType,
                                                /*visibility=*/StringAttr(),
                                                /*arg_attrs=*/builder.getArrayAttr(argAttrs),
                                                /*res_attrs=*/ArrayAttr(),
                                                /*bit width=*/IntegerAttr(),
                                                /*number of arms=*/arms.size());
    for (auto [idx, armInfo] : llvm::enumerate(arms)) {
      ++armCount;
      maxArmOps.updateMax(armInfo->getOps().size());
      maxArmUses.updateMax(armInfo->getCount());
      maxArmReplaced.updateMax(armInfo->getCount() * armInfo->getOps().size());
      buildArm(builder, armInfo->loc, executeOp.getArms()[idx], armInfo, argIndex);
    }
  }
};

} // namespace zirgen::ByteCode
