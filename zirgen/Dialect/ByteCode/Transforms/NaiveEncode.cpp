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
#include "zirgen/Dialect/ByteCode/Transforms/Encoder.h"

#define DEBUG_TYPE "naive-encode"

using namespace mlir;

namespace zirgen::ByteCode {

#define GEN_PASS_DEF_GENEXECUTOR
#define GEN_PASS_DEF_GENENCODING
#include "zirgen/Dialect/ByteCode/Transforms/Passes.h.inc"

namespace {

constexpr size_t kMaxNumArms = 256;

void buildArm(OpBuilder& builder,
              Location loc,
              Region& region,
              const ArmInfo* arm,
              const DenseMap<StringAttr, Value>& funcArgs) {
  builder.createBlock(&region);

  SmallVector<Value> values;

  llvm::errs() << "arm: " << arm << ", " << *arm << "\n";
  for (auto argName : arm->funcArgNames) {
    llvm::errs() << "argName: " << argName << "\n";

    if (!funcArgs.contains(argName)) {
      llvm::errs() << "Unable to find arg " << argName << " in arg names:\n";
      llvm::interleaveComma(llvm::make_first_range(funcArgs), llvm::errs());
      abort();
    }
    values.push_back(funcArgs.lookup(argName));
  }

  for (auto val : arm->getLoadVals()) {
    values.push_back(builder.create<LoadOp>(loc, val.getType()));
  }

  size_t offset = 0;
  Operation* newOp = nullptr;
  for (auto [op, numIntArgs] : llvm::zip_equal(arm->ops, arm->opIntArgs)) {
    llvm::errs() << "offset " << offset << "/" << arm->valueOffsets.size() << " op " << *op
                 << " numIntArgs " << numIntArgs << "\n";
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
  assert(newOp);
  builder.create<YieldOp>(loc, newOp->getResults());
}

} // namespace

struct GenExecutorPass : public impl::GenExecutorBase<GenExecutorPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    func::FuncOp funcOp = getOperation();
    assert(funcOp.getBody().hasOneBlock());
    auto loc = funcOp.getLoc();

    ArmAnalysis& armAnalysis = getAnalysis<ArmAnalysis>();
    Block* oldBody = &funcOp.getBody().front();
    OpBuilder builder(&getContext());

    // Gather function arguments

    DenseMap<StringAttr, Value> funcArgs;
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
      auto argName = funcOp.getArgAttrOfType<StringAttr>(idx, "zirgen.argName");
      if (!argName) {
        argName = builder.getStringAttr("arg" + std::to_string(idx));
      }
      funcArgs[argName] = arg;
    }

    // Make sure we include all distinct operations so we can support anything that we want to
    // encode into bytecode.
    SmallVector<const ArmInfo*> arms;
    llvm::append_range(arms, llvm::make_pointer_range(armAnalysis.getDistinctOps()));

    // Take arms from multiOpArms based on total number of operations
    // among all instances until we fill up kMaxNumArms arms.
    SmallVector<std::pair<size_t, const ArmInfo*>> multiOpArms;
    llvm::append_range(multiOpArms,
                       llvm::map_range(armAnalysis.getMultiOpArms(), [&](const ArmInfo& armInfo) {
                         return std::make_pair(armInfo.ops.size() * armInfo.count, &armInfo);
                       }));
    llvm::sort(multiOpArms, llvm::less_first());

    while (arms.size() < kMaxNumArms && !multiOpArms.empty()) {
      arms.push_back(multiOpArms.back().second);
      multiOpArms.pop_back();
    }

    builder.createBlock(&funcOp.getBody());

    auto executeOp = builder.create<ExecutorOp>(
        funcOp.getLoc(), funcOp.getResultTypes(), /*number of arms=*/arms.size());

    for (auto [idx, armInfo] : llvm::enumerate(arms)) {
      buildArm(builder, loc, executeOp.getArms()[idx], armInfo, funcArgs);
    }

    builder.setInsertionPointAfter(executeOp);
    builder.create<func::ReturnOp>(loc, executeOp->getResults());

    oldBody->erase();
  }
};

struct GenEncodingPass : public impl::GenEncodingBase<GenEncodingPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    assert(funcOp.getBody().hasOneBlock());

    // Generate an executor for this function.  We have to store it
    // inside our current operation so that runPipeline will agree to
    // run a pipeline on it.

    OpBuilder builder(&getContext());
    ModuleOp subMod = builder.create<ModuleOp>(getOperation()->getLoc());
    builder.createBlock(&subMod.getBodyRegion());
    func::FuncOp execFuncOp = llvm::cast<func::FuncOp>(builder.clone(*funcOp));
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    builder.insert(subMod);
    OpPassManager pm;
    pm.addPass(createGenExecutor());
    if (failed(runPipeline(pm, execFuncOp))) {
      execFuncOp.erase();
      signalPassFailure();
      return;
    }
    subMod->remove();

    ExecutorOp execOp = *execFuncOp.getBody().getOps<ExecutorOp>().begin();

    RewritePatternSet patterns(&getContext());
    addEncodePatterns(patterns, execOp);
    ConversionTarget target(getContext());
    target.addLegalOp<func::FuncOp, func::ReturnOp, EncodedOp>();
    if (applyFullConversion(getOperation(), target, std::move(patterns)).failed())
      signalPassFailure();
    subMod.erase();
  }
};

} // namespace zirgen::ByteCode
