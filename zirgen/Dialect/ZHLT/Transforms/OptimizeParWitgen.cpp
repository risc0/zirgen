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
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"
#include "zirgen/Dialect/ZStruct/Transforms/RewritePatterns.h"
#include "llvm/Support/Debug.h"

#include <set>
#include <vector>

using namespace mlir;
using namespace zirgen::Zll;
using namespace zirgen::ZStruct;

namespace zirgen::Zhlt {

namespace {

// Attempts to to unravel the use of the result of a switch operation
// returning a StructType.  We generate an additional switch operation
// with the same condition for each member of the structure, having
// the switch operation only return that member.  Then, we add a pack
// operation to reconstruct the structure from the individual switch
// operations.
//
// Constraints are left in the original switch operation for
// processing by SplitSwitchArms, but all uses of the result value are
// changed to use the repacked value.
//
// This requires all operations inside are at the least idempotent if
// not completely pure, since they may be duplicated between struct
// members.  As such, we verify that all the operations are ones are
// allow before we attempt unravelling.

struct UnravelSwitchPackResult : public OpRewritePattern<SwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SwitchOp op, PatternRewriter& rewriter) const final {
    // Don't bother unravelling if we don't need these results
    if (op->use_empty())
      return rewriter.notifyMatchFailure(op, "Unused");

    StructType ty = dyn_cast<StructType>(op.getType());
    if (!ty)
      return rewriter.notifyMatchFailure(op, "Not a struct return");

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

    auto packOp = rewriter.create<PackOp>(op.getLoc(), op.getType(), splitFields);
    rewriter.replaceAllUsesWith(op, packOp);
    return success();
  }
};

// Attempts to to unravel the use of the result of a switch operation returning
// an ArrayType. We generate an additional switch operation with the same
// condition for each element of the array, having the switch operation only
// return that member. Then, we add an array operation to reconstruct the array
// from the individual switch operations.
//
// Constraints are left in the original switch operation for processing by
// SplitSwitchArms, but all uses of the result value are changed to use the
// repacked value.
//
// This requires all operations inside are idempotent, since they may be
// duplicated between array elements.  As such, we verify that all the
// operations are ones are allow before we attempt unravelling.
struct UnravelSwitchArrayResult : public OpRewritePattern<SwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SwitchOp op, PatternRewriter& rewriter) const final {
    // Don't bother unravelling if we don't need these results
    if (op->use_empty())
      return failure();

    ArrayType ty = dyn_cast<ArrayType>(op.getType());
    if (!ty)
      return failure();

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
};

// Attempts to unravel a use of the result of a switch operations returning a val
// by multiplying by each of the selectors and summing the result.
//
// This assumes there's exactly one selector that has a value of 1 and the
// rest are zero.
struct UnravelSwitchValResult : public OpRewritePattern<SwitchOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SwitchOp op, PatternRewriter& rewriter) const final {
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
};

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

struct EraseUselessExtern : public OpRewritePattern<Zll::ExternOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Zll::ExternOp op, PatternRewriter& rewriter) const final {
    if (!op.use_empty())
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

struct ReplaceYieldWithTerminator : public OpRewritePattern<YieldOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(YieldOp op, PatternRewriter& rewriter) const final {
    if (llvm::isa<IfOp>(op->getParentOp())) {
      rewriter.replaceOpWithNewOp<TerminateOp>(op);
      return success();
    }

    return failure();
  }
};

struct OptimizeParWitgenPass : public OptimizeParWitgenBase<OptimizeParWitgenPass> {
  void optWitnessFunc(StepFuncOp funcOp, OpBuilder& builder) {
    auto* ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<EraseOp<EqualZeroOp>>(ctx);
    patterns.insert<EraseUselessExtern>(ctx);
    ZStruct::getUnrollPatterns(patterns, ctx);
    for (auto* dialect : ctx->getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : ctx->getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, ctx);

    patterns.insert<SplitSwitchArms>(ctx);

    // Lower benefit so we do these only when we can't do anything else:
    patterns.insert<UnravelSwitchPackResult>(ctx, /*benefit=*/0);
    patterns.insert<UnravelSwitchArrayResult>(ctx, /*benefit=*/0);
    patterns.insert<UnravelSwitchValResult>(ctx, /*benefit=*/0);

    if (applyPatternsAndFoldGreedily(funcOp, std::move(patterns)).failed()) {
      auto diag = getOperation()->emitError("unable to strip for witgen");
      signalPassFailure();
    }
  }

  void runOnOperation() override {
    auto funcs = llvm::to_vector(getOperation().getBody()->getOps<StepFuncOp>());

    for (auto func : funcs) {
      auto builder = OpBuilder::atBlockEnd(getOperation().getBody());
      optWitnessFunc(func, builder);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createOptimizeParWitgenPass() {
  return std::make_unique<OptimizeParWitgenPass>();
}

} // namespace zirgen::Zhlt
