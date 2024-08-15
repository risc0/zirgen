// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/dsl/passes/CommonRewrites.h"

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

} // namespace zirgen
