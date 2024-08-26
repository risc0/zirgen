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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "zirgen/Dialect/ZStruct/IR/Types.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/PassDetail.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"

#include <set>
#include <vector>

using namespace mlir;

namespace zirgen::ZStruct {
namespace {

// Map of names to GetBufferOps referencing those names
using BufferMapT = DenseMap<Block*, DenseMap</*bufName=*/StringAttr, GetBufferOp>>;

GetBufferOp getBuffer(BufferMapT& bufs, Block* block, StringAttr name) {
  auto parentOp = block->getParentOp();
  if (parentOp && parentOp->getBlock() && !parentOp->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    auto parentBuf = getBuffer(bufs, parentOp->getBlock(), name);
    if (parentBuf)
      return parentBuf;
  }

  auto [it, didInsert] = bufs.try_emplace(block);
  auto& localBufs = it->second;
  if (didInsert) {
    // Initial population of GetBufferOps.  Traverse in reverse order
    // so we get the earliest one in the block as the final entry.
    for (auto op : llvm::reverse(block->getOps<GetBufferOp>())) {
      localBufs[op.getNameAttr()] = op;
    }
  }

  return localBufs.lookup(name);
}

BoundLayoutAttr evalLayout(Zll::Interpreter& interp, Value ref) {
  auto layout = interp.evaluateConstantOfType<BoundLayoutAttr>(ref);
  if (!layout)
    return {};
  return layout;
}

struct ResolveLoad : public OpRewritePattern<LoadOp> {
  ResolveLoad(MLIRContext* ctx, Zll::Interpreter& interp, BufferMapT& bufs)
      : OpRewritePattern(ctx), interp(interp), bufs(bufs) {}

  LogicalResult matchAndRewrite(LoadOp op, PatternRewriter& rewriter) const final {
    BoundLayoutAttr layout = evalLayout(interp, op.getRef());
    if (!layout)
      return failure();

    IntegerAttr distance;

    auto buf = getBuffer(bufs, op->getBlock(), layout.getBuffer());
    if (!buf)
      return failure();

    if (buf.getType().getKind() != Zll ::BufferKind::Global) {
      // Ignore interpreter errors
      ScopedDiagnosticHandler scopedHandler(getContext(),
                                            [&](Diagnostic& diag) { return success(); });

      distance = interp.evaluateConstantOfType<IntegerAttr>(op.getDistance());
      if (!distance)
        return failure();
    }

    auto ref = llvm::dyn_cast<RefAttr>(layout.getLayout());
    if (!ref)
      return failure();

    // Read however many field elements we need from the buffer,
    // shifting and multiplying by {0, 1, 0, 0} to fill in all of the
    // extended element result.
    RefType refType = ref.getType();
    auto baseType =
        rewriter.getType<Zll::ValType>(refType.getElement().getField(), /*extended=*/UnitAttr{});

    Value elem;
    Value shift;
    for (auto offset : llvm::reverse(llvm::seq(0u, refType.getElement().getFieldK()))) {
      Value newElem;
      if (buf.getType().getKind() == Zll::BufferKind::Global) {
        newElem =
            rewriter.create<Zll::GetGlobalOp>(op.getLoc(), baseType, buf, ref.getIndex() + offset);
      } else {
        newElem = rewriter.create<Zll::GetOp>(op.getLoc(),
                                              baseType,
                                              buf,
                                              ref.getIndex() + offset,
                                              distance.getValue().getZExtValue(),
                                              /*tap=*/IntegerAttr{});
      }
      if (elem) {
        if (!shift) {
          SmallVector<uint64_t> shiftElems;
          shiftElems.resize(refType.getElement().getFieldK(), 0);
          shiftElems[1] = 1;
          shift = rewriter.create<Zll::ConstOp>(
              op.getLoc(), refType.getElement(), rewriter.getAttr<PolynomialAttr>(shiftElems));
        }
        elem = rewriter.create<Zll::MulOp>(op.getLoc(), elem, shift);
        elem = rewriter.create<Zll::AddOp>(op.getLoc(), elem, newElem);
      } else {
        elem = newElem;
      }
    }
    assert(elem && "Empty field elements?");

    rewriter.replaceOp(op, elem);
    return success();
  }

  Zll::Interpreter& interp;
  BufferMapT& bufs;
};

struct ResolveStore : public OpRewritePattern<StoreOp> {
  ResolveStore(MLIRContext* ctx, Zll::Interpreter& interp, BufferMapT& bufs)
      : OpRewritePattern(ctx), interp(interp), bufs(bufs) {}

  LogicalResult matchAndRewrite(StoreOp op, PatternRewriter& rewriter) const final {
    BoundLayoutAttr layout = evalLayout(interp, op.getRef());
    if (!layout)
      return failure();

    auto ref = llvm::dyn_cast<RefAttr>(layout.getLayout());
    if (!ref)
      return failure();
    auto buf = getBuffer(bufs, op->getBlock(), layout.getBuffer());
    if (!buf)
      return failure();

    if (buf.getType().getKind() == Zll::BufferKind::Global) {
      if (op.getVal().getType().getFieldK() > 1)
        return failure();
      rewriter.replaceOpWithNewOp<Zll::SetGlobalOp>(op, buf, ref.getIndex(), op.getVal());
    } else {
      rewriter.replaceOpWithNewOp<Zll::SetOp>(op, buf, ref.getIndex(), op.getVal());
    }
    return success();
  }

  Zll::Interpreter& interp;
  BufferMapT& bufs;
};

struct InlineLayoutPass : public InlineLayoutBase<InlineLayoutPass> {
  void runOnOperation() override {
    BufferMapT bufs;
    Zll::Interpreter interp(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<ResolveLoad, ResolveStore>(&getContext(), interp, bufs);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    if (applyPatternsAndFoldGreedily(getOperation(), frozenPatterns).failed()) {
      signalPassFailure();
    }

    if (false && !bufs.empty()) {
      // applyPatternsAndFoldGreedily may have moved around our GetBufferOps
      // GetBufferOps, so propagate them all to the top level.
      Block* topBlock = &getOperation()->getRegion(0).front();
      assert(topBlock && "Expecting inline layout pass to run on a region");
      for (auto blockBufs : bufs) {
        for (auto bufOps : /*map<name, op>=*/blockBufs.second) {
          Operation* op = bufOps.second;
          op->moveBefore(topBlock, topBlock->begin());
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createInlineLayoutPass() {
  return std::make_unique<InlineLayoutPass>();
}

} // namespace zirgen::ZStruct
