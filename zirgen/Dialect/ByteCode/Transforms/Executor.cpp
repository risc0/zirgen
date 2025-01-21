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

#include "zirgen/Dialect/ByteCode/Transforms/Executor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace zirgen::ByteCode {

namespace {

size_t getBitsToHoldInclusive(size_t maxVal) {
  if (maxVal < 1)
    return 0;
  size_t bitsToHold = 0;
  while (maxVal) {
    bitsToHold += 8;
    maxVal >>= 8;
  }
  return bitsToHold;
}

size_t getBitsToHoldUpTo(size_t size) {
  if (size < 2)
    return 0;
  return getBitsToHoldInclusive(size - 1);
}

} // namespace

ExecuteOp
buildExecutor(Location loc, Region* region, Value encodedInput, BufferizeInterface& bufferize) {
  assert(region->hasOneBlock());
  func::ReturnOp returnOp = llvm::cast<func::ReturnOp>(region->front().getTerminator());
  TypeRange resultTypes = returnOp.getOperands().getTypes();

  size_t numArms = 0;
  DenseMap</*dispatch key=*/DispatchKeyAttr, /*arm number=*/size_t> armByKey;

  OpBuilder builder(region->getContext());
  for (Operation* op : llvm::make_pointer_range(region->front())) {
    DispatchKeyAttr key = getDispatchKey(op);
    if (armByKey.contains(key)) {
      // Don't need to make a new arm for this key
      continue;
    }

    armByKey[key] = numArms++;
  }

  ExecuteOp execOp = builder.create<ExecuteOp>(
      loc, resultTypes, encodedInput, /*intKindInfo=*/ArrayAttr(), numArms);

  // Sizes required for any temporary buffers.
  llvm::MapVector</*intKind=*/Attribute, /*bufSize=*/size_t> bufSizes;

  // Maximum values used to decode any op-specific integers
  llvm::MapVector</*intKind=*/Attribute, /*maxVal=*/size_t> decodeMax;

  for (Operation* op : llvm::make_pointer_range(region->front())) {
    DispatchKeyAttr key = getDispatchKey(op);
    size_t armIdx = armByKey.lookup(key);
    Region& armRegion = execOp.getRegion(armIdx);

    // Update maximums of int values used.
    SmallVector<size_t> intArgs;
    if (auto bcInterface = dyn_cast<ByteCodeOpInterface>(op)) {
      bcInterface.getByteCodeIntArgs(intArgs);
      for (auto [intKind, val] : llvm::zip_equal(key.getIntKinds(), intArgs)) {
        auto& maxVal = decodeMax[intKind];
        maxVal = std::max<size_t>(maxVal, val);
      }
    } else
      assert(key.getIntKinds().empty());

    // Update maximums of buffer references
    for (Value operand : op->getOperands()) {
      auto definer = operand.getDefiningOp();
      if (definer && region->getParentOp()->isProperAncestor(definer)) {
        // This value comes from within the region; decode it from the bytecode and update our
        // required buffer sizes.
        auto [intKind, bufIndex] = bufferize.getKindAndIndex(operand);
        auto& bufSize = bufSizes[intKind];
        bufSize = std::max<size_t>(bufSize, bufIndex + 1);
      }
    }

    SmallVector<Attribute> resultIntKinds;
    for (Value result : op->getResults()) {
      auto [intKind, bufIndex] = bufferize.getKindAndIndex(result);
      resultIntKinds.push_back(intKind);
      auto& bufSize = bufSizes[intKind];
      bufSize = std::max<size_t>(bufSize, bufIndex + 1);
    }

    if (!armRegion.empty()) {
      // Already filled this region; no need to fill again
      continue;
    }

    builder.createBlock(&armRegion);

    SmallVector<Value> args;

    for (Value operand : op->getOperands()) {
      auto definer = operand.getDefiningOp();
      if (definer && region->getParentOp()->isProperAncestor(definer)) {
        // This value comes from within the region; decode it from the bytecode and update our
        // required buffer sizes.
        auto [intKind, bufIndex] = bufferize.getKindAndIndex(operand);
        args.push_back(builder.create<LoadTemporaryOp>(op->getLoc(), operand.getType(), intKind));
      } else {
        // Capture it from outside the region
        args.push_back(operand);
      }
    }

    for (auto [intKind, val] : llvm::zip_equal(key.getIntKinds(), intArgs)) {
      args.push_back(builder.create<DecodeOp>(op->getLoc(), intKind));
    }

    if (auto returnOp = llvm::dyn_cast<func::ReturnOp>(op)) {
      assert(resultIntKinds.empty());
      builder.create<ExitOp>(op->getLoc(), key, args);
      continue;
    } else if (!key.getIntKinds().empty()) {
      // Wrap in a Operation to decode the extra args
      auto opOp = builder.create<OperationOp>(
          op->getLoc(), op->getResultTypes(), op->getName().getIdentifier(), args);
      builder.create<YieldOp>(
          op->getLoc(), key, opOp->getResults(), builder.getArrayAttr(resultIntKinds));
    } else {
      IRMapping mapper;
      mapper.map(op->getOperands(), args);
      Operation* cloned = builder.clone(*op, mapper);
      builder.create<YieldOp>(
          op->getLoc(), key, cloned->getResults(), builder.getArrayAttr(resultIntKinds));
    }
  }

  SmallVector<Attribute> intKindInfos;
  intKindInfos.push_back(builder.getAttr<IntKindInfoAttr>(
      getDispatchKeyIntKind(builder.getContext()), getBitsToHoldUpTo(numArms)));

  for (auto& [intKind, bufSize] : bufSizes) {
    intKindInfos.push_back(builder.getAttr<IntKindInfoAttr>(intKind, getBitsToHoldUpTo(bufSize)));
  }
  for (auto& [intKind, maxVal] : decodeMax) {
    intKindInfos.push_back(
        builder.getAttr<IntKindInfoAttr>(intKind, getBitsToHoldInclusive(maxVal)));
  }

  execOp.setIntKindsAttr(builder.getArrayAttr(intKindInfos));
  return execOp;
}

} // namespace zirgen::ByteCode
