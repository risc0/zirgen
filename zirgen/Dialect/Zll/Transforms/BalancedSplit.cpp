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

#include "zirgen/Dialect/Zll/Transforms/BalancedSplit.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Transforms/RegionUtils.h"
#include "risc0/core/util.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen::Zll {

namespace {

bool isInexpensive(Operation* op) {
  return llvm::isa<ConstOp, GetOp, GetGlobalOp>(op);
}

struct Splitter {
  Splitter(Block* origBlock, size_t nsplit);

  std::optional<size_t> getPartIndex(Operation* op) {
    for (const auto& [idx, part] : llvm::enumerate(partSets)) {
      if (part.contains(op))
        return idx;
    }
    return std::nullopt;
  }

  void addSpill(Value value);
  void calculateSpills();
  void splitPart(size_t partNumm, Block* dest);
  void addToPart(size_t partNum, Operation* op);

  Block* origBlock;
  size_t nsplit;

  SmallVector<DenseSet<Operation*>> partSets;
  SmallVector<SmallVector<Operation*>> partOps;

  // values to spill to a temporary buffer, by type.
  llvm::MapVector<Type, DenseMap<Value, /*offset=*/size_t>> spilled;
  DenseMap<Type, Value> spillBufs;

  OpBuilder builder;
};

Splitter::Splitter(Block* origBlock, size_t nsplit)
    : origBlock(origBlock)
    , nsplit(nsplit)
    , partSets(nsplit)
    , partOps(nsplit)
    , builder(origBlock->getParentOp()->getContext()) {
  SmallVector<Operation*> ops;
  for (auto& op : origBlock->without_terminator()) {
    if (!isInexpensive(&op)) {
      ops.push_back(&op);
    }
  }

  for (size_t i : llvm::seq(nsplit)) {
    size_t startIdx = ops.size() * i / nsplit;
    size_t endIdx = ops.size() * (i + 1) / nsplit;

    llvm::append_range(partOps[i], ArrayRef(ops).slice(startIdx, endIdx - startIdx));
    partSets[i].insert(partOps[i].begin(), partOps[i].end());
  }
}

void Splitter::addToPart(size_t i, Operation* op) {
  partOps[i].push_back(op);
  partSets[i].insert(op);
}

void Splitter::addSpill(Value value) {
  auto& typeSpilled = spilled[value.getType()];
  if (typeSpilled.contains(value))
    return;
  size_t index = typeSpilled.size();
  typeSpilled[value] = index;
  assert(typeSpilled.size() == index + 1);
}

void Splitter::calculateSpills() {
  for (size_t i : llvm::seq(nsplit)) {
    for (Operation* op : partOps[i]) {
      for (Value arg : op->getOperands()) {
        auto operand = llvm::dyn_cast<OpResult>(arg);
        if (!operand)
          continue;

        if (!llvm::isa<ValType>(operand.getType()))
          continue;

        auto ownerIndex = getPartIndex(operand.getDefiningOp());
        if (ownerIndex && ownerIndex != i)
          // Crosses spill boundaries
          addSpill(operand);
      }
    }
  }

  builder.setInsertionPoint(&origBlock->front());
  for (auto& [type, vals] : spilled) {
    auto bufType =
        builder.getType<BufferType>(llvm::cast<ValType>(type), vals.size(), BufferKind::Temporary);
    spillBufs[type] = builder.create<MakeTemporaryBufferOp>(builder.getUnknownLoc(), bufType);
  }
}

void Splitter::splitPart(size_t partNum, Block* dest) {
  DenseMap<Value, Value> locals;

  auto mapVal = [&](Value orig) -> Value {
    Value mapped = locals.lookup(orig);
    if (!mapped) {
      // Default to capture original value, if we don't spill it into a buffer
      mapped = orig;

      if (spilled.contains(orig.getType())) {
        auto& typeSpilled = spilled[orig.getType()];
        if (typeSpilled.contains(orig)) {
          mapped = builder.create<GetGlobalOp>(orig.getLoc(),
                                               spillBufs.at(orig.getType()),
                                               /*offset=*/typeSpilled.at(orig));
        }
      }

      locals[orig] = mapped;
    }
    return mapped;
  };

  for (Operation* op : partOps[partNum]) {
    op->moveBefore(dest, dest->end());
    builder.setInsertionPoint(op);
    for (OpOperand& opArg : op->getOpOperands()) {
      opArg.set(mapVal(opArg.get()));
    }
    for (Value result : op->getResults()) {
      // Use anything generated locally without having to reload it from spill
      locals[result] = result;

      if (spilled.contains(result.getType())) {
        auto& typeSpilled = spilled[result.getType()];
        if (typeSpilled.contains(result)) {
          builder.setInsertionPointToEnd(dest);
          builder.create<SetGlobalOp>(result.getLoc(),
                                      spillBufs.at(result.getType()),
                                      /*offset=*/typeSpilled.at(result),
                                      result);
        }
      }
    }
  }
}

std::string getSymbolBase(Operation* op) {
  while (op) {
    if (auto funcOp = llvm::dyn_cast<FunctionOpInterface>(op)) {
      return funcOp.getName().str();
    }
    op = op->getParentOp();
  }

  return "split";
}

} // namespace

void balancedSplitBlock(Block* block, size_t nsplit) {
  assert(nsplit > 0);

  if (nsplit == 1)
    return;

  Location loc = block->getParentOp()->getLoc();
  ModuleOp mod = block->getParentOp()->getParentOfType<ModuleOp>();
  unsigned int uniqueIndex = 0;
  std::string symbolBase = getSymbolBase(block->getParentOp());

  Operation* origTerminator = block->getTerminator();
  ValueRange returnVals = origTerminator->getOperands();

  Splitter splitter(block, nsplit);
  splitter.calculateSpills();

  // makeRegionIsolatedFromAbove requires IRRewriter, not just OpBuilder.
  IRRewriter builder(mod.getContext());

  // Innermost split returns the values, the rest just pass it along.
  builder.setInsertionPointToEnd(block);
  auto firstReturnOp = builder.create<func::ReturnOp>(loc, returnVals);
  splitter.addToPart(nsplit - 1, firstReturnOp);

  for (size_t i : llvm::reverse(llvm::seq(size_t(1), nsplit))) {
    auto newName = SymbolTable::generateSymbolName<64>(
        symbolBase, [&](auto sym) -> bool { return mod.lookupSymbol(sym); }, uniqueIndex);

    // Apparently makeRegionIsolatedFromAbove only correctly detects captures when run on a
    // descendant region, so start the new function out inside the old one.
    builder.setInsertionPointToEnd(block);
    auto newFuncOp = builder.create<func::FuncOp>(
        loc, newName, builder.getFunctionType(/*inputs=*/{}, /*results=*/returnVals.getTypes()));
    SymbolTable::setSymbolVisibility(newFuncOp, SymbolTable::Visibility::Private);
    Block* splitBlock = &newFuncOp.getBody().emplaceBlock();
    splitter.splitPart(i, splitBlock);
    builder.setInsertionPointToEnd(splitBlock);

    auto passValues = makeRegionIsolatedFromAbove(
        builder,
        newFuncOp.getBody(),
        /*clone into region?=*/[&](Operation* op) { return isInexpensive(op); });

    newFuncOp.setFunctionType(builder.getFunctionType(
        /*inputs=*/newFuncOp.getBody().front().getArgumentTypes(),
        /*outputs=*/returnVals.getTypes()));

    builder.setInsertionPointToEnd(block);
    auto callOp = builder.create<func::CallOp>(loc, newFuncOp, passValues);
    splitter.addToPart(i - 1, callOp);
    if (i == 1) {
      // Change original terminator to use results from the call
      origTerminator->setOperands(callOp.getResults());
      splitter.addToPart(0, origTerminator);
    } else {
      // Make a new return op
      auto returnOp = builder.create<func::ReturnOp>(loc, callOp.getResults());
      splitter.addToPart(i - 1, returnOp);
    }

    newFuncOp->moveBefore(mod.getBody(), mod.getBody()->begin());
    newFuncOp.getBody().front().invalidateOpOrder();
  }

  // Process the first part in place
  splitter.splitPart(0, block);
  block->invalidateOpOrder();
}

struct BalancedSplitPass : public BalancedSplitBase<BalancedSplitPass> {
  BalancedSplitPass(size_t maxOps) : maxOps(maxOps) {}

  void runOnOperation() override {
    getOperation()->walk([&](Block* block) {
      size_t numOps = block->getOperations().size();
      if (numOps > maxOps) {
        balancedSplitBlock(block, risc0::ceilDiv(numOps, maxOps));
      }
    });
  }

  size_t maxOps;
};

std::unique_ptr<Pass> createBalancedSplitPass(size_t maxOps) {
  return std::make_unique<BalancedSplitPass>(maxOps);
}

} // namespace zirgen::Zll
