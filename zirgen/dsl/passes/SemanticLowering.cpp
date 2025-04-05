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

#include <functional>

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/RewritePatterns.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "zirgen/dsl/passes/CommonRewrites.h"
#include "zirgen/dsl/passes/PassDetail.h"

using namespace mlir;
using namespace zirgen::Zll;
using namespace zirgen::ZStruct;

namespace cl = llvm::cl;

static cl::opt<size_t> circuitNdebug(
    "circuit-ndebug",
    cl::desc("Don't check constraints when generating proofs.  This can make debugging more "
             "difficult, since problems won't be detected until verification."),
    cl::init(false));

namespace zirgen {
namespace dsl {

// Convert zhlt.construct to func.call on the component's "exec" function.
struct ConstructToCall : public OpRewritePattern<Zhlt::ConstructOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Zhlt::ConstructOp op, PatternRewriter& rewriter) const final {
    auto callOp = rewriter.create<Zhlt::ExecCallOp>(op->getLoc(),
                                                    op.getCallee(),
                                                    op.getType(),
                                                    op.getConstructParam(),
                                                    /*layout=*/op.getLayout());
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

// Inline zhlt.constructs for use in "check" functions.
struct InlineCheckConstruct : public OpRewritePattern<Zhlt::ConstructOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Zhlt::ConstructOp op, PatternRewriter& rewriter) const final {
    Zhlt::ComponentOp callable =
        op->getParentOfType<ModuleOp>().lookupSymbol<Zhlt::ComponentOp>(op.getCallee());
    if (!callable)
      return rewriter.notifyMatchFailure(op, "failed to resolve symbol " + op.getCallee());

    IRMapping mapping;
    Region clonedBody;
    callable.getBody().cloneInto(&clonedBody, mapping);
    remapInlinedLocations(clonedBody.getBlocks(), op.getLoc());
    Block* block = &clonedBody.front();
    auto returnOp = cast<Zhlt::ReturnOp>(block->getTerminator());

    rewriter.inlineBlockBefore(block, op, op.getOperands());
    rewriter.replaceOp(op, returnOp->getOperands());
    rewriter.eraseOp(returnOp);
    return success();
  }
};

// Convert maps over argument arrays to maps over a range
struct ArrayMapToRangeMap : public OpRewritePattern<MapOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MapOp op, PatternRewriter& rewriter) const final {
    TypedValue<ArrayLikeTypeInterface> array = op.getArray();

    if (array.getDefiningOp() && !isa<Zhlt::MagicOp>(array.getDefiningOp()))
      return rewriter.notifyMatchFailure(op, "array is not an argument");

    // Create a range to replace the argument array with
    SmallVector<Value> content;
    for (uint64_t i = 0; i < array.getType().getSize(); i++) {
      content.push_back(rewriter.create<Zll::ConstOp>(op.getLoc(), i));
    }
    Value range = rewriter.create<ZStruct::ArrayOp>(op.getLoc(), content);

    // before: for elem : arr { ... }
    //  after: for i : 0..N { elem := arr[i]; ... }
    auto mapOp = rewriter.create<ZStruct::MapOp>(op.getLoc(), op.getType(), range, op.getLayout());
    mapOp.getBody().takeBody(op.getBody());
    rewriter.setInsertionPointToStart(&mapOp.getBody().front());
    BlockArgument idx = mapOp.getBody().getArgument(0);
    idx.setType(getValType(op.getContext()));
    Value replacementArg = rewriter.create<SubscriptOp>(op.getLoc(), array, idx);
    rewriter.replaceAllUsesWith(idx, replacementArg);

    rewriter.replaceOp(op, mapOp);
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

// Convert zhlt.construct to func.call on the component's "back"
// function.  Assumes it's within a "back" fnuction.
struct ConstructToBack : public OpRewritePattern<Zhlt::ConstructOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Zhlt::ConstructOp op, PatternRewriter& rewriter) const final {
    auto parent = op->getParentOfType<Zhlt::BackFuncOp>();
    if (!parent) {
      return failure();
    }

    if (op.use_empty()) {
      // TODO: Find a better way to indicate that back functions have no effects.
      op.erase();
      return success();
    }

    auto distance = parent.getDistance();
    auto callee = SymbolTable::lookupNearestSymbolFrom<Zhlt::ComponentOp>(op, op.getCalleeAttr());
    auto callOp = rewriter.create<Zhlt::BackCallOp>(
        op->getLoc(), callee.getSymName(), callee.getOutType(), distance, op.getLayout());
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

// Replace a "back" inside a back function to add its distance to the
// parent's.  Assumes it's within a "back$" function.
struct BackBackToCall : public OpRewritePattern<Zhlt::BackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Zhlt::BackOp op, PatternRewriter& rewriter) const final {
    auto parent = op->getParentOfType<Zhlt::BackFuncOp>();
    if (!parent) {
      return failure();
    }

    auto oldDistance = parent.getDistance();
    // TODO: unify distance types and just call op.getDistanceAttr().
    auto backOpDistance = rewriter.create<arith::ConstantOp>(
        op->getLoc(), rewriter.getIndexAttr(op.getDistance().getZExtValue()));
    auto callee = SymbolTable::lookupNearestSymbolFrom<Zhlt::ComponentOp>(op, op.getCalleeAttr());
    auto distance = rewriter.create<arith::AddIOp>(op->getLoc(), oldDistance, backOpDistance);
    auto callOp = rewriter.create<Zhlt::BackCallOp>(
        op->getLoc(), callee.getSymName(), callee.getOutType(), distance, op.getLayout());
    rewriter.replaceOp(op, callOp);
    return success();
  }
};

// Replace a "load" inside a back function to get the distance from the enclosing function.
// Assumes it's within a "back$" function.
struct AddLoadDistance : public OpRewritePattern<LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LoadOp op, PatternRewriter& rewriter) const final {
    if (llvm::isa<BlockArgument>(op.getDistance())) {
      // Already attached to parent BackFuncOp.
      return failure();
    }
    auto parent = op->getParentOfType<Zhlt::BackFuncOp>();
    if (!parent) {
      return failure();
    }
    assert(parent.getSymName().starts_with("back$"));
    auto distanceArg = parent.getDistance();
    rewriter.startOpModification(op);
    op.getDistanceMutable().assign(distanceArg);
    rewriter.finalizeOpModification(op);
    return success();
  }
};

/// Analyse all components to see which ones are accessed using
/// zhlt.back, both directly and transitively.
class BacksNeededAnalysis {
public:
  BacksNeededAnalysis(ModuleOp module) {
    SmallVector<Zhlt::ComponentOp> added;

    // Find which components call zhlt.back directly
    module.walk([&](Zhlt::BackOp backOp) {
      auto symName = backOp.getCalleeAttr().getAttr();
      auto [it, didInsert] = backsNeeded.try_emplace(symName);
      it->second.insert(backOp);
      if (didInsert) {
        added.push_back(module.lookupSymbol<Zhlt::ComponentOp>(symName));
      }
    });

    // Any components which were called by zhlt.back also need to have
    // back functions generated for all their subcomponents.
    while (!added.empty()) {
      SmallVector<Zhlt::ComponentOp> newAdded;

      for (auto component : added) {
        component.walk([&](Zhlt::ConstructOp construct) {
          auto symName = construct.getCalleeAttr().getAttr();
          auto [it, didInsert] = backsNeeded.try_emplace(symName);
          it->second.insert(construct);
          if (didInsert) {
            newAdded.push_back(module.lookupSymbol<Zhlt::ComponentOp>(symName));
          }
        });
      }

      added = std::move(newAdded);
    }
  }

  bool backNeeded(Zhlt::ComponentOp component) {
    if (!component.getLayoutType() && !component.getConstructParamTypes().empty()) {
      // No need to generate back functions for components without registers.
      return false;
    }
    return backsNeeded.contains(component.getSymNameAttr());
  }

  const DenseSet<Operation*>& getUses(Zhlt::ComponentOp component) {
    return backsNeeded[component.getSymNameAttr()];
  }

private:
  DenseMap<StringAttr, DenseSet<Operation*>> backsNeeded;
};

struct GenerateBackPass : public GenerateBackBase<GenerateBackPass> {
  void runOnOperation() override {
    auto* ctx = &getContext();
    auto& backsNeeded = getAnalysis<BacksNeededAnalysis>();

    RewritePatternSet patterns(ctx);
    patterns.insert<EraseOp<StoreOp>>(ctx);
    patterns.insert<EraseOp<ExternOp>>(ctx);
    patterns.insert<EraseOp<Zhlt::MagicOp>>(ctx);
    patterns.insert<EraseOp<ZStruct::AliasLayoutOp>>(ctx);
    patterns.insert<EraseOp<EqualZeroOp>>(ctx);
    patterns.insert<ConstructToBack>(ctx);
    patterns.insert<BackBackToCall>(ctx);
    patterns.insert<AddLoadDistance>(ctx);
    patterns.insert<ArrayMapToRangeMap>(ctx);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    OpBuilder builder(ctx);
    // Generate "back" functions
    getOperation()->walk([&](Zhlt::ComponentOp op) {
      if (!backsNeeded.backNeeded(op)) {
        return;
      }

      builder.setInsertionPoint(op);

      auto layoutType = op.getLayoutType();
      auto valType = op.getResultType();
      auto distanceType = builder.getIndexType();

      // Start out with the old function signature and copy in the block
      auto func =
          builder.create<Zhlt::BackFuncOp>(op->getLoc(), op.getSymName(), valType, layoutType);

      IRMapping mapping;
      op.getBody().cloneInto(&func.getBody(), mapping);

      // Now strip it down to just the distance and layout arguments.
      // Replace other arguments with MagicOps, which should go away
      // at the end of the lowering.
      Block* block = &func.getBody().front();
      // arg0 = exec context, arg1 = back distance, arg2 = optional layout
      size_t numValArgs = op.getConstructParam().size();
      builder.setInsertionPointToStart(block);
      for (auto constructArg : op.getConstructParam()) {
        auto valArg = mapping.lookup(constructArg);
        auto magic = builder.create<Zhlt::MagicOp>(op.getLoc(), valArg.getType());
        valArg.replaceAllUsesWith(magic);
      }
      block->eraseArguments(0u, numValArgs);
      block->insertArgument(0u, distanceType, op->getLoc());
      // arg1 = back distance, arg2 = optional layout
      assert(block->getNumArguments() == 1 || block->getNumArguments() == 2);

      if (applyPatternsGreedily(func, frozenPatterns).failed()) {
        auto diag = func->emitError()
                    << "Unable to generate `back' function; required by the following locations:";
        for (auto usedBy : backsNeeded.getUses(op)) {
          diag.attachNote(usedBy->getLoc()) << "here";
        }

        signalPassFailure();
      }
    });
  }
};

struct GenerateExecPass : public GenerateExecBase<GenerateExecPass> {
  void runOnOperation() override {
    auto* ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<ConstructToCall>(ctx);
    patterns.insert<BackToCall>(ctx);
    if (circuitNdebug)
      patterns.insert<EraseOp<EqualZeroOp>>(ctx);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    OpBuilder builder(ctx);
    // Transform ComponentOps into exec functions.
    getOperation().walk([&](Zhlt::ComponentOp op) {
      builder.setInsertionPoint(op);

      auto func = builder.create<Zhlt::ExecFuncOp>(op->getLoc(),
                                                   op.getName(),
                                                   op.getOutType(),
                                                   op.getConstructParamTypes(),
                                                   op.getLayoutType());
      IRMapping mapping;
      op.getBody().cloneInto(&func.getBody(), mapping);

      if (applyPatternsGreedily(func, frozenPatterns).failed()) {
        func->emitError("Could not generate back function");
        signalPassFailure();
      }
    });
  }
};

// Transform ComponentOps into constraint-checking functions.
struct GenerateCheckPass : public GenerateCheckBase<GenerateCheckPass> {
  void
  generateCheckFunc(OpBuilder& builder, StringRef checkFuncName, ArrayRef<StringAttr> callees) {
    MLIRContext* ctx = builder.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<InlineCheckConstruct>(ctx);
    patterns.insert<BackToCall>(ctx);
    patterns.insert<EraseOp<StoreOp>>(ctx);
    patterns.insert<EraseOp<VariadicPackOp>>(ctx);
    patterns.insert<EraseOp<ExternOp>>(ctx);
    patterns.insert<EraseOp<AliasLayoutOp>>(ctx);
    patterns.insert<EraseOp<Zhlt::MagicOp>>(ctx);
    patterns.insert<InlineCalls>(ctx);
    patterns.insert<SplitSwitchArms>(ctx);
    patterns.insert<ReplaceYieldWithTerminator>(ctx);
    ZStruct::SwitchOp::getCanonicalizationPatterns(patterns, ctx);
    ZStruct::getUnrollPatterns(patterns, ctx);
    Zll::EqualZeroOp::getCanonicalizationPatterns(patterns, ctx);

    // Only try these if nothing else work, since they cause a lot of duplication.
    patterns.insert<UnravelSwitchPackResult>(ctx, /*benefit=*/0);
    patterns.insert<UnravelSwitchArrayResult>(ctx, /*benefit=*/0);
    patterns.insert<UnravelSwitchValResult>(ctx, /*benefit=*/0);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    OpBuilder::InsertionGuard guard(builder);
    auto loc = NameLoc::get(builder.getStringAttr("All Constraints"));
    auto checkFuncOp = builder.create<Zhlt::CheckFuncOp>(loc, checkFuncName);
    builder.setInsertionPointToStart(checkFuncOp.addEntryBlock());

    for (auto callee : callees) {
      builder.create<func::CallOp>(loc, callee, /*results=*/TypeRange{});
    }

    // Now, inline everything and get rid of everything that's not a constraint.
    builder.create<Zhlt::ReturnOp>(loc);
    GreedyRewriteConfig config;
    config.maxIterations = 100;
    if (applyPatternsGreedily(checkFuncOp, frozenPatterns, config).failed()) {
      checkFuncOp->emitError("Could not generate check function");
      signalPassFailure();
    }
  }

  void runOnOperation() override {
    auto* ctx = &getContext();

    OpBuilder builder(ctx);
    mlir::ModuleOp mod = getOperation();
    builder.setInsertionPointToEnd(mod.getBody());

    std::map</*checkFuncName=*/std::string, /*callees=*/SmallVector<StringAttr>> checkFuncs;

    mod.walk([&](Zhlt::StepFuncOp op) {
      // Generate a circuit-wide constraint checker, plus one for teach t3est.
      StringRef stepName = op.getName();
      std::string checkName;
      if (stepName.consume_front("step$test$")) {
        stepName.consume_back("$accum");
        checkName = ("test$" + stepName).str();
      }

      checkFuncs[checkName].push_back(op.getSymNameAttr());
    });

    for (const auto& [name, callees] : checkFuncs) {
      generateCheckFunc(builder, name, callees);
    }
  }
};

struct GenerateValidityRegsPass : public GenerateValidityRegsBase<GenerateValidityRegsPass> {
  Value runOnBlock(Location loc,
                   Block& block,
                   OpBuilder& builder,
                   IRMapping& mapper,
                   Value state,
                   Value polyMixArg) {
    for (Operation& origOp : block.without_terminator()) {
      Location opLoc = origOp.getLoc();
      if (opLoc != loc)
        opLoc = CallSiteLoc::get(loc, origOp.getLoc());

      TypeSwitch<Operation*>(&origOp)
          .Case<EqualZeroOp>([&](EqualZeroOp op) {
            auto oldIn = op.getIn();
            if (!mapper.contains(oldIn)) {
              // A previous error may have resulted in this invalid state
              op.emitError("Invalid state for MakePolynomial");
              signalPassFailure();
              return;
            }
            auto newIn = mapper.lookup(oldIn);
            state = builder.create<AndEqzOp>(opLoc, state, newIn);
          })
          .Case<IfOp>([&](IfOp op) {
            Value innerState = builder.create<TrueOp>(opLoc);
            innerState =
                runOnBlock(opLoc, op.getInner().front(), builder, mapper, innerState, polyMixArg);
            state = builder.create<AndCondOp>(opLoc, state, mapper.lookup(op.getCond()), innerState)
                        .getResult();
          })
          // Polynomial and structural ops just get passed through
          .Case<Zhlt::BackOp,
                GetBufferOp,
                SwitchOp,
                PolyOp,
                LookupOp,
                SubscriptOp,
                LoadOp,
                PackOp,
                ArrayOp,
                BindLayoutOp,
                arith::ConstantOp,
                arith::AddIOp>([&](auto op) {
            OpBuilder::InsertionGuard guard(builder);
            if (llvm::isa<LoadOp, LookupOp, SubscriptOp>(op.getOperation()))
              builder.setInsertionPointToStart(builder.getBlock());
            builder.clone(origOp, mapper);
          })
          .Default([&](Operation* op) {
            llvm::errs() << *op;
            op->emitError("Invalid op for MakePolynomial");
            signalPassFailure();
          });
    }
    return state;
  }

  void runOnOperation() override {
    auto module = getOperation();

    module.walk([&](Zhlt::CheckFuncOp checkFunc) {
      OpBuilder builder(checkFunc);
      if (checkFunc.getSymName() != "check$")
        return;
      auto func = builder.create<Zhlt::ValidityRegsFuncOp>(
          checkFunc.getLoc(),
          "validity_regs",
          builder.getFunctionType({builder.getType<PolyMixType>()},
                                  {builder.getType<ConstraintType>()}),
          /*argNames=*/ArrayRef<StringRef>({"polyMix"}));
      Block* block = func.addEntryBlock();
      Value polyMix = block->getArgument(0);
      IRMapping mapper;
      builder.setInsertionPointToStart(block);
      Value mixState = builder.create<TrueOp>(func.getLoc());
      mixState = runOnBlock(
          checkFunc.getLoc(), checkFunc.getBody().front(), builder, mapper, mixState, polyMix);
      builder.create<Zhlt::ReturnOp>(func.getLoc(), mixState);
      sortTopologically(builder.getBlock());
    });
  }
};

using NamedTap = std::tuple</*bufferName=*/StringRef, /*offset=*/size_t, /*back=*/size_t>;

struct TapifyLoadOp : public OpRewritePattern<LoadOp> {
  TapifyLoadOp(MLIRContext* ctx, Interpreter& interp, DenseMap<NamedTap, Value>& tapIndex)
      : OpRewritePattern(ctx), interp(interp), tapIndex(tapIndex) {}

  LogicalResult matchAndRewrite(LoadOp op, PatternRewriter& rewriter) const final {
    BoundLayoutAttr ref = interp.evaluateConstantOfType<BoundLayoutAttr>(op.getRef());
    if (!ref) {
      op->emitError("couldn't evaluate reference");
      return failure();
    }

    auto distAttr = interp.evaluateConstantOfType<IntegerAttr>(op.getDistance());
    if (!distAttr) {
      op->emitError("couldn't evaluate distance");
      return failure();
    }
    size_t distance = getIndexVal(distAttr);

    auto namedTap =
        NamedTap{ref.getBuffer(), llvm::cast<RefAttr>(ref.getLayout()).getIndex(), distance};
    if (tapIndex.contains(namedTap)) {
      rewriter.replaceOp(op, tapIndex.lookup(namedTap));
      return success();
    } else {
      return failure();
    }
  }

private:
  Interpreter& interp;
  DenseMap<NamedTap, Value>& tapIndex;
};

struct GenerateValidityTapsPass : public GenerateValidityTapsBase<GenerateValidityTapsPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto ctx = module.getContext();
    OpBuilder builder(ctx);
    auto bufs = Zll::lookupModuleAttr<Zll::BuffersAttr>(module);
    auto tapsOp = module.lookupSymbol<GlobalConstOp>(Zhlt::getTapsConstName());
    if (!tapsOp) {
      return;
    }
    ArrayAttr taps = cast<ArrayAttr>(tapsOp.getConstant());
    auto groupNames =
        llvm::map_to_vector(bufs.getTapBuffers(), [&](auto bufDesc) { return bufDesc.getName(); });

    module.walk([&](Zhlt::ValidityRegsFuncOp regsFunc) {
      builder.setInsertionPoint(regsFunc);
      auto func = builder.create<Zhlt::ValidityTapsFuncOp>(
          regsFunc.getLoc(),
          "validity_taps",
          builder.getFunctionType({/*taps=*/builder.getType<BufferType>(Zhlt::getExtValType(ctx),
                                                                        taps.size(),
                                                                        Zll::BufferKind::Constant),
                                   builder.getType<PolyMixType>()},
                                  {builder.getType<ConstraintType>()}),
          ArrayRef<StringRef>({"taps", "polyMix"}));
      builder.setInsertionPointToStart(func.addEntryBlock());
      auto tapsArg = func.getArgument(0);
      auto polyMixArg = func.getArgument(1);

      // This AttrTypeReplacer extends all field elements to extension
      // field elements, whether they be in types or attributes.
      size_t extSize = Zhlt::getExtValType(ctx).getFieldK();
      AttrTypeReplacer extendFieldTypes;
      // Leave layout types alone; they will be discarded in the final
      // output after all the tap locations have been evaluated.
      extendFieldTypes.addReplacement(
          [&](LayoutType t) { return std::make_pair(t, WalkResult::skip()); });
      extendFieldTypes.addReplacement(
          [&](RefType t) { return std::make_pair(t, WalkResult::skip()); });
      extendFieldTypes.addReplacement(
          [&](BufferType t) { return std::make_pair(t, WalkResult::skip()); });
      extendFieldTypes.addReplacement([&](ValType t) { return Zhlt::getExtValType(ctx); });
      extendFieldTypes.addReplacement([&](PolynomialAttr attr) {
        SmallVector<uint64_t> elems = llvm::to_vector(attr.asArrayRef());
        elems.resize(extSize, 0);
        return PolynomialAttr::get(ctx, elems);
      });

      // Store which refs are in which tap indexes.
      DenseMap<NamedTap, Value> tapIndex;
      size_t ntaps = 0;
      for (auto tap : taps) {
        auto tapRef = cast<TapAttr>(tap);
        auto namedTap =
            NamedTap{groupNames[tapRef.getRegGroupId()], tapRef.getOffset(), tapRef.getBack()};
        assert(!tapIndex.contains(namedTap));
        tapIndex[namedTap] = builder.create<Zll::GetOp>(regsFunc.getLoc(),
                                                        tapsArg,
                                                        /*offset=*/ntaps++,
                                                        /*back=*/0,
                                                        /*tap=*/mlir::IntegerAttr{});
      }
      assert(ntaps == taps.size());

      IRMapping mapper;
      mapper.map(/*polyMix=*/regsFunc.getArgument(0), polyMixArg);
      for (auto& op : regsFunc.getBody().front())
        builder.clone(op, mapper);

      Interpreter interp(ctx);
      RewritePatternSet patterns(ctx);
      patterns.insert<TapifyLoadOp>(ctx, interp, tapIndex);
      FrozenRewritePatternSet frozenPatterns(std::move(patterns));
      if (applyPatternsGreedily(func, frozenPatterns).failed()) {
        auto diag = func->emitError("Unable to generate `verify taps' function");
        signalPassFailure();
        return;
      }

      // Convert field elements and NondetRegs to extension field elements in all types.
      extendFieldTypes.recursivelyReplaceElementsIn(
          func, /*replaceAttrs=*/true, /*replaceLocs=*/false, /*replaceTypes=*/true);

      // Re-infer types on any load-ops, since we may have changed their return types
      func.walk([&](ZStruct::LoadOp loadOp) { reinferReturnType(loadOp); });

      // Elminate dead code referring to old layout.
      IRRewriter rewriter(builder);
      DominanceInfo domInfo;
      bool changed = true;
      while (changed) {
        eliminateCommonSubExpressions(rewriter, domInfo, func, &changed);
      }
    });
  }
};

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateBackPass() {
  return std::make_unique<GenerateBackPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateExecPass() {
  return std::make_unique<GenerateExecPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateCheckPass() {
  return std::make_unique<GenerateCheckPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateValidityRegsPass() {
  return std::make_unique<GenerateValidityRegsPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createGenerateValidityTapsPass() {
  return std::make_unique<GenerateValidityTapsPass>();
}

} // namespace dsl
} // namespace zirgen
