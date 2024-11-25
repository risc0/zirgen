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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"

using namespace mlir;
using namespace zirgen::ZStruct;

namespace zirgen {

void remapInlinedLocations(iterator_range<Region::iterator> inlinedBlocks, Location callerLoc);

template <typename OpTy> struct EraseOp : public OpRewritePattern<OpTy> {
  using OpType = OpTy;
  using OpRewritePattern<OpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpType op, PatternRewriter& rewriter) const final {
    if (op.use_empty()) {
      op.erase();
    } else if (!isa<Zhlt::MagicOp>(op)) {
      // Replace uses with MagicOp.  This should ensure that they get
      // optimized away.
      SmallVector<Value> results =
          llvm::to_vector(llvm::map_range(op->getResultTypes(), [&](auto ty) -> Value {
            return rewriter.create<Zhlt::MagicOp>(op->getLoc(), ty);
          }));
      rewriter.replaceOp(op, results);
    } else {
      return failure();
    }

    return success();
  }
};

struct InlineCalls : public OpInterfaceRewritePattern<CallOpInterface> {
  using OpType = CallOpInterface;
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(CallOpInterface callOp, PatternRewriter& rewriter) const;
};

template <typename WrappedPattern> struct RewritePatternSuper {
  using OpType = typename WrappedPattern::OpType;

  // using Type = OpRewritePattern<OpType>;
  using Type =
      typename std::conditional<std::is_base_of<OpRewritePattern<OpType>, WrappedPattern>::value,
                                OpRewritePattern<OpType>,
                                OpInterfaceRewritePattern<OpType>>::type;

  static_assert(std::is_base_of<OpRewritePattern<OpType>, WrappedPattern>::value ||
                    std::is_base_of<OpInterfaceRewritePattern<OpType>, WrappedPattern>::value,
                "BaseClass must be either an OpRewritePattern or OpInterfaceRewritePattern");
};

// For any WrappedPattern, OnlyUnmuxed<WrappedPattern> works the same way as
// WrappedPattern except that it only applies to operations that do not occur
// lexically inside any muxes.
template <typename WrappedPattern>
struct OnlyUnmuxed : public RewritePatternSuper<WrappedPattern>::Type {
  using OpType = typename WrappedPattern::OpType;

  template <typename... Args>
  OnlyUnmuxed(MLIRContext* ctx, Args&&... args)
      : RewritePatternSuper<WrappedPattern>::Type(ctx), wrapped(ctx, std::forward<Args>(args)...) {}

  LogicalResult matchAndRewrite(OpType op, PatternRewriter& rewriter) const {
    if (op->template getParentOfType<SwitchOp>())
      return rewriter.notifyMatchFailure(op, "occurs inside of a mux");
    // auto typedOp = cast<typename WrappedPattern::OpType>(op);
    return wrapped.matchAndRewrite(op, rewriter);
  }

private:
  WrappedPattern wrapped;
};

// Replace a "Back" inside an execution function to call the back
// function.  Assumes it's within an "exec" or "check" function.
struct BackToCall : public OpRewritePattern<Zhlt::BackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Zhlt::BackOp op, PatternRewriter& rewriter) const final {
    // TODO: unify distance types and just call op.getDistanceAttr().
    auto distance = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(), rewriter.getIndexAttr(op.getDistance().getZExtValue()));
    auto callee = SymbolTable::lookupNearestSymbolFrom<Zhlt::ComponentOp>(op, op.getCalleeAttr());
    auto callOp = rewriter.create<Zhlt::BackCallOp>(
        op->getLoc(), callee.getSymName(), callee.getOutType(), distance, op.getLayout());
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

} // namespace zirgen
