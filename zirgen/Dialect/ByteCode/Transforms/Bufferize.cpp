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

#include "zirgen/Dialect/ByteCode/Transforms/Bufferize.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace zirgen::ByteCode {

std::pair</*kind=*/mlir::StringAttr, /*size=*/size_t>
NaiveBufferize::getKindAndSize(mlir::Value value) {
  return std::make_pair(kind, 1);
}

namespace {

struct BufInfo {
  size_t allocate(Value operand, size_t numRegs) {
    assert(numRegs > 0);

    // Try to reuse a free list item
    for (size_t tryRegs = numRegs; hasFree(tryRegs); tryRegs *= 2) {
      auto& tryFree = getFree(tryRegs);
      if (!tryFree.empty()) {
        size_t offset = tryFree.pop_back_val();
        while (tryRegs > numRegs) {
          tryRegs /= 2;
          getFree(tryRegs).insert(offset);
          offset += tryRegs;
        }
        return offset;
      }
    }

    // Allocate new
    size_t offset = maxOffset;
    maxOffset += numRegs;
    return offset;
  }

  void consume(Operation* op, Value operand, size_t offset, size_t numRegs) {
    if (llvm::any_of(operand.getUsers(), [&](Operation* user) {
          return user->getBlock() == op->getBlock() && op->isBeforeInBlock(user);
        })) {
      return;
    }

    release(offset, numRegs);
  }

  void release(size_t offset, size_t numRegs) {
    while (hasFree(numRegs * 2)) {
      auto& tryFree = getFree(numRegs);
      assert(!tryFree.contains(offset));
      if (offset >= numRegs && tryFree.contains(offset - numRegs)) {
        // Merge
        tryFree.remove(offset - numRegs);
        offset -= numRegs;
        numRegs *= 2;
        continue;
      }
      if (tryFree.contains(offset + numRegs)) {
        tryFree.remove(offset + numRegs);
        numRegs *= 2;
        continue;
      }
      break;
    }

    getFree(numRegs).insert(offset);
  }

  SmallVector<SetVector</*offset=*/size_t>> sizedFree;
  size_t maxOffset = 0;
  bool hasFree(size_t numRegs) {
    assert(numRegs > 0);
    assert(!(numRegs & (numRegs - 1)) && "Only power of 2 reg counts are supported currently");
    size_t idx = llvm::Log2_64(numRegs);
    return sizedFree.size() > idx;
  }
  SetVector</*offset=*/size_t>& getFree(size_t numRegs) {
    assert(numRegs > 0);
    assert(!(numRegs & (numRegs - 1)) && "Only power of 2 reg counts are supported currently");
    size_t idx = llvm::Log2_64(numRegs);
    if (sizedFree.size() <= idx)
      sizedFree.resize(idx + 1);
    return sizedFree[idx];
  }
};

struct ActiveInfo {
  size_t offset;
  size_t numRegs;
  StringAttr bufKind;
};

struct BufferizeImpl {
  BufferizeImpl(EncodedBlockOp encodedOp, BufferizeInterface& bufferize)
      : encodedOp(encodedOp), bufferize(bufferize) {}

  void processBlock(Block* block) {
    for (EncodedOp op : block->getOps<EncodedOp>()) {
      processOperation(op);
    }

    // Execute all replacements starting from the end to avoid removing values that still have uses.
    IRRewriter rewriter(block->begin()->getContext());
    while (!replacements.empty()) {
      auto [op, encoded] = replacements.pop_back_val();
      rewriter.setInsertionPoint(op);
      rewriter.create<EncodedOp>(op->getLoc(), encoded);
      rewriter.eraseOp(op);
    }
  }

  void processOperation(EncodedOp op) {
    DenseMap<Value, size_t> encodings;

    for (Value operand : op->getOperands()) {
      if (encodings.contains(operand))
        // If an argument is duplciated, don't consume it more than once.
        continue;

      if (active.contains(operand)) {
        const auto& act = active.at(operand);
        encodings[operand] = act.offset;
        bufs[act.bufKind].consume(op, operand, act.offset, act.numRegs);
      }
    }

    for (Value result : op->getResults()) {
      assert(!encodings.contains(result));

      encodings[result] = allocateOperand(op, result);
    }

    SmallVector<BuildEncodedElement> newEncoded;
    for (size_t i : llvm::seq(op.size())) {
      const auto& elemVar = op.getElement(i);
      if (const size_t* elem = std::get_if<size_t>(&elemVar)) {
        newEncoded.push_back(size_t(*elem));
      } else if (const Value* elem = std::get_if<Value>(&elemVar)) {
        assert(encodings.contains(*elem));
        newEncoded.push_back(size_t(encodings.at(*elem)));
      } else if (const OpResult* elem = std::get_if<OpResult>(&elemVar)) {
        assert(encodings.contains(*elem));
        newEncoded.push_back(size_t(encodings.at(*elem)));
      } else
        llvm_unreachable("Unknown encoded element variant");
    }

    replacements.emplace_back(op, std::move(newEncoded));
  }

  size_t allocateOperand(Operation* op, Value operand) {
    auto [kind, numRegs] = bufferize.getKindAndSize(operand);

    size_t offset = bufs[kind].allocate(operand, numRegs);
    active[operand] = ActiveInfo{.offset = offset, .numRegs = numRegs, .bufKind = kind};
    return offset;
  }

private:
  EncodedOp encodedOp;
  BufferizeInterface& bufferize;

  DenseMap</*buffer kind=*/StringAttr, BufInfo> bufs;
  DenseMap<Value, /*encoded temp buffer offset=*/ActiveInfo> active;
  SmallVector<std::pair<Operation*, SmallVector<BuildEncodedElement>>> replacements;
};
} // namespace

LogicalResult bufferize(EncodedBlockOp encodedOp, BufferizeInterface& bufferize) {
  BufferizeImpl impl(encodedOp, bufferize);

  encodedOp.walk([&](Block* block) { impl.processBlock(block); });

  //  impl.saveMetadata(encodedOp);
  return success();

} // namespace

} // namespace zirgen::ByteCode
