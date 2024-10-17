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

#include "zirgen/dsl/passes/CommonRewrites.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"

namespace zirgen {

/// Remap locations from the inlined blocks with CallSiteLoc locations with the
/// provided caller location.  Copied from mlir's InliningUtils.
void remapInlinedLocations(iterator_range<Region::iterator> inlinedBlocks, Location callerLoc) {
  DenseMap<Location, Location> mappedLocations;
  auto remapOpLoc = [&](Operation* op) {
    auto it = mappedLocations.find(op->getLoc());
    if (it == mappedLocations.end()) {
      auto newLoc = CallSiteLoc::get(op->getLoc(), callerLoc);
      it = mappedLocations.try_emplace(op->getLoc(), newLoc).first;
    }
    op->setLoc(it->second);
  };
  for (auto& block : inlinedBlocks)
    block.walk(remapOpLoc);
}

LogicalResult InlineCalls::matchAndRewrite(CallOpInterface callOp,
                                           PatternRewriter& rewriter) const {
  auto callable = callOp.getCallableForCallee();
  if (!callable)
    return rewriter.notifyMatchFailure(callOp, "Not a callable");
  if (!llvm::isa<SymbolRefAttr>(callable))
    return rewriter.notifyMatchFailure(callOp, "Callable isn't a symbol");
  auto callee = SymbolTable::lookupNearestSymbolFrom<CallableOpInterface>(
      callOp, llvm::cast<SymbolRefAttr>(callable));
  if (!callee)
    return rewriter.notifyMatchFailure(callOp, "Can't find callee with Callable interface");

  if (callOp.getArgOperands().getTypes() != callee.getArgumentTypes()) {
    return rewriter.notifyMatchFailure(callOp, "Argument type mismatch");
  }

  IRMapping mapping;
  Region clonedBody;
  callee.getCallableRegion()->cloneInto(&clonedBody, mapping);
  remapInlinedLocations(clonedBody.getBlocks(), callOp.getLoc());
  Block* block = &clonedBody.front();
  auto returnOp = block->getTerminator();
  if (!returnOp->hasTrait<OpTrait::ReturnLike>())
    return rewriter.notifyMatchFailure(callOp, "Callee missing return operation");

  rewriter.inlineBlockBefore(block, callOp, callOp.getArgOperands());
  rewriter.replaceOp(callOp, returnOp->getOperands());
  rewriter.eraseOp(returnOp);

  return success();
}

LogicalResult BackToCall::matchAndRewrite(Zhlt::BackOp op, PatternRewriter& rewriter) const {
  // TODO: unify distance types and just call op.getDistanceAttr().
  auto distance = rewriter.create<arith::ConstantOp>(
      op->getLoc(), rewriter.getIndexAttr(op.getDistance().getZExtValue()));
  auto callee = SymbolTable::lookupNearestSymbolFrom<Zhlt::ComponentOp>(op, op.getCalleeAttr());
  auto callOp = rewriter.create<Zhlt::BackCallOp>(op->getLoc(),
                                                  callee.getSymName(),
                                                  callee.getOutType(),
                                                  distance,
                                                  op.getLayout());
  rewriter.replaceOp(op, callOp);
  return success();
}

static bool isIdempotent(Operation* op) {
  return op->hasTrait<OpTrait::ConstantLike>() || isPure(op) ||
         llvm::isa<PolyOp, EqualZeroOp, YieldOp, IfOp, TerminateOp, LoadOp>(op);
}

LogicalResult UnravelSwitchPackResult::matchAndRewrite(SwitchOp op, PatternRewriter& rewriter) const {
  // Don't bother unravelling if we don't need these results
  if (op->use_empty())
    return failure();

  StructType ty = dyn_cast<StructType>(op.getType());
  if (!ty)
    return failure();

  // Make sure all operations are ones we expect
  for (auto& region : op->getRegions()) {
    for (auto& nestedOp : region.getOps()) {
      if (!isIdempotent(&nestedOp))
        return failure();
    }
  }

  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> splitFields;
  for (auto field : ty.getFields()) {
    auto fieldSplitOp =
        rewriter.create<SwitchOp>(op.getLoc(), field.type, op.getSelector(), op.getArms().size());
    for (size_t i = 0; i != op.getArms().size(); ++i) {
      auto& origArm = op.getArms()[i];
      OpBuilder::InsertionGuard insertionGuard(rewriter);
      rewriter.createBlock(&fieldSplitOp.getArms()[i]);

      IRMapping mapper;
      for (auto& origOp : origArm.front()) {
        TypeSwitch<Operation*>(&origOp)
            .Case<EqualZeroOp>([&](auto origOp) {
              // Don't add constraints to any of the copies.
            })
            .Case<Zll::IfOp>([&](auto origOp) {
              // If ops can't contribute to the result, so skip them.
            })
            .Case<YieldOp>([&](auto origOp) {
              auto lookupOp = rewriter.createOrFold<LookupOp>(
                  origOp.getLoc(), mapper.lookupOrDefault(origOp.getOperand()), field.name);
              rewriter.create<YieldOp>(origOp.getLoc(), lookupOp);
            })
            .Default([&](auto origOp) { rewriter.clone(*origOp, mapper); });
      }
    }
    fieldSplitOp->getOpResult(0).setType(field.type);
    splitFields.push_back(fieldSplitOp);
  }

  // TODO: this currently discards the layout of the member, which should be
  // the common super layout of the mux.
  auto packOp = rewriter.create<PackOp>(op.getLoc(), op.getType(), Value(), splitFields);
  rewriter.replaceAllUsesWith(op, packOp);
  return success();
}

LogicalResult UnravelSwitchArrayResult::matchAndRewrite(SwitchOp op, PatternRewriter& rewriter) const {
  // Don't bother unravelling if we don't need these results
  if (op->use_empty())
    return failure();

  ArrayType ty = dyn_cast<ArrayType>(op.getType());
  if (!ty)
    return failure();

  // Make sure all operations are ones we expect
  for (auto& region : op->getRegions()) {
    for (auto& nestedOp : region.getOps()) {
      if (!isIdempotent(&nestedOp))
        return failure();
    }
  }

  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> splitElements;
  for (size_t i = 0; i < ty.getSize(); i++) {
    Value index = rewriter.create<arith::ConstantOp>(op->getLoc(), rewriter.getIndexAttr(i));
    auto elementSplitOp = rewriter.create<SwitchOp>(
        op.getLoc(), ty.getElement(), op.getSelector(), op.getArms().size());
    for (size_t j = 0; j < op.getArms().size(); j++) {
      OpBuilder::InsertionGuard insertionGuard(rewriter);
      rewriter.createBlock(&elementSplitOp.getArms()[j]);

      IRMapping mapper;
      for (auto& origOp : op.getArms()[j].front()) {
        TypeSwitch<Operation*>(&origOp)
            .Case<EqualZeroOp>([&](auto origOp) {
              // Don't add constraints to any of the copies.
            })
            .Case<IfOp>([&](auto origOp) {
              // If ops can't contribute to the result, so skip them.
            })
            .Case<YieldOp>([&](auto origOp) {
              auto subscriptOp = rewriter.createOrFold<SubscriptOp>(
                  origOp.getLoc(), mapper.lookupOrDefault(origOp.getOperand()), index);
              rewriter.create<YieldOp>(origOp.getLoc(), subscriptOp);
            })
            .Default([&](auto origOp) { rewriter.clone(*origOp, mapper); });
      }
    }
    splitElements.push_back(elementSplitOp);
  }

  auto arrayOp = rewriter.create<ArrayOp>(op.getLoc(), op.getType(), splitElements);
  rewriter.replaceAllUsesWith(op, arrayOp);
  return success();
}



LogicalResult UnravelSwitchValResult::matchAndRewrite(SwitchOp op, PatternRewriter& rewriter) const {
  // Don't bother if we don't need these results
  if (op->use_empty())
    return failure();

  ValType ty = dyn_cast<ValType>(op.getType());
  if (!ty)
    return failure();

  // If there's anything better ot be done like inlining or inner switch operations, deal with
  // those first.
  for (auto& region : op->getRegions()) {
    for (auto& block : region) {
      for (auto& blockOp : block) {
        if (!llvm::isa<PolyOp, EqualZeroOp, YieldOp, IfOp, TerminateOp, LoadOp>(blockOp) &&
            !blockOp.hasTrait<OpTrait::ConstantLike>() && !isPure(&blockOp))
          return failure();
      }
    }
  }

  rewriter.setInsertionPointAfter(op);
  SmallVector<Value> elems;
  for (auto [arm, sel] : llvm::zip_equal(op.getArms(), op.getSelector())) {
    Value selValue = sel;
    IRMapping mapper;
    Value mulOp;
    for (auto& origOp : arm.front()) {
      TypeSwitch<Operation*>(&origOp)
          .Case<EqualZeroOp>([&](auto origOp) {
            // Don't copy constraints; they will stay in the original switch operation.
          })
          .Case<IfOp>([&](auto origOp) {
            // "If" operations don't return anything, so they can't contribute to the result.
          })
          .Case<YieldOp>([&](auto origOp) {
            mulOp = rewriter.createOrFold<Zll::MulOp>(
                op.getLoc(), mapper.lookupOrDefault(origOp.getOperand()), selValue);
          })
          .Case<LoadOp>([&](auto origOp) {
            auto loadOp = rewriter.clone(*origOp, mapper);
            rewriter.modifyOpInPlace(
                loadOp, [&]() { loadOp->setAttr("unchecked", rewriter.getAttr<UnitAttr>()); });
          })
          .Default([&](auto origOp) { rewriter.clone(*origOp, mapper); });
    }
    assert(mulOp && "Undable to find yield op in arm");
    elems.push_back(mulOp);
  }
  Value sum;
  for (auto elem : elems) {
    if (sum)
      sum = rewriter.createOrFold<Zll::AddOp>(op.getLoc(), sum, elem);
    else
      sum = elem;
  }
  assert(sum && "Unable to find any arms to sum up");
  rewriter.replaceAllUsesWith(op, sum);
  return success();
}

} // namespace zirgen
