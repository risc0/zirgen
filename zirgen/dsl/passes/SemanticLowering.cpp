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

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/BindLayouts.h"
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
template <typename ConstructLike>
struct ConstructToCall : public OpRewritePattern<Zhlt::ConstructOp> {
  using OpType = Zhlt::ConstructOp;
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Zhlt::ConstructOp op, PatternRewriter& rewriter) const final {
    auto callOp = rewriter.create<ConstructLike>(op->getLoc(),
                                                 op.getCallee(),
                                                 op.getType(),
                                                 op.getConstructParam(),
                                                 /*layout=*/op.getLayout());
    rewriter.replaceOp(op, callOp->getResults());
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
    patterns.insert<ConstructToCall<Zhlt::ExecCallOp>>(ctx);
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

using PoisonFunc = void (*)(Operation* op, DenseSet<Operation*>& poisoned);

// Propagate poison along chains of uses
static void poisonUses(Operation* op, DenseSet<Operation*>& poisoned) {
  if (poisoned.contains(op))
    return;

  poisoned.insert(op);
  for (auto user : op->getUsers()) {
    poisonUses(user, poisoned);
  }
  if (isa<ZStruct::YieldOp>(op)) {
    poisonUses(op->getParentOp(), poisoned);
  }
}

// Propagate poison along chains of defs
static void poisonDefs(Operation* op, DenseSet<Operation*>& poisoned) {
  if (poisoned.contains(op))
    return;

  poisoned.insert(op);
  for (auto operand : op->getOperands()) {
    if (auto operandOp = operand.getDefiningOp()) {
      poisonDefs(operandOp, poisoned);
    }
  }
}

// Transform ComponentOps into constraint-checking functions.
struct GenerateCheckPass : public GenerateCheckBase<GenerateCheckPass> {
  // Create a check function that calls the composable check functions for all
  // the relevant entry points. These will be flattened later.
  void
  generateCheckFuncs() {
    mlir::ModuleOp mod = getOperation();

    // Collect entry points for the whole circuit as well as each test.
    std::map</*checkFuncName=*/std::string, /*callees=*/SmallVector<Zhlt::ComponentOp>> checkFuncs;
    mod.walk([&](Zhlt::ComponentOp op) {
      if (Zhlt::isEntryPoint(op)) {
        StringRef name = op.getName();
        name.consume_back("$accum");
        checkFuncs[name.str()].push_back(op);
      }
    });

    OpBuilder builder(&getContext());
    BufferAnalysis& bufferAnalysis = getAnalysis<BufferAnalysis>();
    auto loc = NameLoc::get(builder.getStringAttr("All Constraints"));

    for (const auto& [name, callees] : checkFuncs) {
      builder.setInsertionPointToEnd(mod.getBody());
      auto checkFuncOp = builder.create<Zhlt::CheckFuncOp>(loc, name);
      builder.setInsertionPointToStart(checkFuncOp.addEntryBlock());

      // Create calls to the composable check functions for each entry point.
      for (auto callee : callees) {
        if (failed(bindLayoutsForEntryPoint<ComposableCheckCallOp>(callee, builder, bufferAnalysis))) {
          signalPassFailure();
        }
      }

      builder.create<ReturnOp>(loc);
    }
  }

  void generateComposableCheckFuncs() {
    MLIRContext* ctx = &getContext();
    OpBuilder builder(ctx);

    // Start by cloning each component into a composable check function. These
    // are not yet legal, but they will be legalized momentarily.
    getOperation().walk([&](Zhlt::ComponentOp op) {
      builder.setInsertionPoint(op);

      auto func = builder.create<ComposableCheckFuncOp>(op->getLoc(),
                                                        op.getName(),
                                                        op.getOutType(),
                                                        op.getConstructParamTypes(),
                                                        op.getLayoutType());
      if (!isEntryPoint(op))
        func.setPrivate();
      IRMapping mapping;
      op.getBody().cloneInto(&func.getBody(), mapping);
      checkFunctions.push_back(func);
    });

    // Prune all non-constraint effects from the new check functions
    RewritePatternSet patterns(ctx);
    patterns.insert<BackToCall>(ctx);
    patterns.insert<EraseOp<StoreOp>>(ctx);
    patterns.insert<EraseOp<VariadicPackOp>>(ctx);
    patterns.insert<EraseOp<ExternOp>>(ctx);
    patterns.insert<EraseOp<AliasLayoutOp>>(ctx);
    patterns.insert<ConstructToCall<ComposableCheckCallOp>>(ctx);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    for (auto checkFunc : checkFunctions) {
      if (applyPatternsGreedily(checkFunc, frozenPatterns).failed()) {
        getOperation()->emitError("Could not remove side effects from composable check function");
        signalPassFailure();
      }
    }
  }

  void inlineForLegality(ComposableCheckFuncOp check, PoisonFunc poison) {
    MLIRContext* ctx = &getContext();

    // Use the 'poison' function to compute the set of operations in 'check'
    // that are involved in nondeterministic computations and should be
    // considered for inlining.
    llvm::DenseSet<Operation *> poisoned;
    check->walk([&](Operation* op) {
      if (op != check && !ComposableCheckFuncOp::isLegalNestedOp(op)) {
        poison(op, poisoned);
      }
    });

    // If no operations are poisoned, there's no work to do.
    if (poisoned.empty()) {
      return;
    }

    // We've already pruned unused outputs, which means the return value is
    // definitely used by the caller. If the return value is poisoned, we need
    // to inline so that all illegal operations can be folded away.
    InlinerInterface inlinerInterface(ctx);
    auto ret = cast<ReturnOp>(check.getBody().back().getTerminator());
    if (poisoned.contains(ret)) {
      auto uses = check.getSymbolUses(getOperation());
      for (SymbolTable::SymbolUse use : *uses) {
        auto call = cast<CallOpInterface>(use.getUser());
        LogicalResult result = inlineCall(inlinerInterface,
                                          call,
                                          check,
                                          &check.getBody());
        if (result.failed()) {
          signalPassFailure();
        }
        call->erase();
      }
      check.erase();
      return;
    }

    // Now inline any poisoned calls, so that we can hopefully fold away the
    // illegal operations. Because we legalize in a topological order, the
    // callees are already legalized. Thus, we can legalize in a single scan
    // without iterating to a fixed point.
    check->walk([&](ComposableCheckCallOp call) {
      if (poisoned.contains(call)) {
        auto callee = getOperation().lookupSymbol<ComposableCheckFuncOp>(call.getCallee());
        LogicalResult result = inlineCall(inlinerInterface,
                                          cast<CallOpInterface>(call.getOperation()),
                                          cast<CallableOpInterface>(callee.getOperation()),
                                          callee.getCallableRegion());
        if (result.failed()) {
          signalPassFailure();
        }
        call->erase();
      }
    });

    // Finally, fold.
    GreedyRewriteConfig config;
    config.fold = true;
    if (applyPatternsGreedily(check, {}, config).failed()) {
      signalPassFailure();
    }
  }

  void legalizeComposableCheckFuncs() {
    // Flatten and prune so folding is more effective
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<RemoveUnusedResults>(ctx);
    patterns.insert<RemoveUnusedArguments>(ctx);
    // patterns.insert<SplitSwitchArms>(ctx);
    getUnrollPatterns(patterns, ctx);

    // Only try these if nothing else work, since they cause a lot of duplication.
    patterns.insert<UnravelSwitchPackResult>(ctx, /*benefit=*/0);
    patterns.insert<UnravelSwitchArrayResult>(ctx, /*benefit=*/0);
    patterns.insert<UnravelSwitchValResult>(ctx, /*benefit=*/0);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    for (auto checkFunc : checkFunctions) {
      // If a check function is private and dead, just erase it now.
      if (checkFunc.isPrivate() && checkFunc.symbolKnownUseEmpty(getOperation())) {
        checkFunc->erase();
        continue;
      }

      if (applyPatternsGreedily(checkFunc, frozenPatterns).failed()) {
        getOperation()->emitError("Could not remove side effects from composable check function");
        signalPassFailure();
      }
    }

    // At this point, there may still be nondeterministic operations hanging
    // around. Since constraints can only operate on polynomials, these must
    // fold away with sufficient inlining. Here, we apply just enough inlining
    // to get to that point.
    const CallGraph& cg = getAnalysis<CallGraph>();
    auto it = llvm::scc_begin(&cg);
    while (!it.isAtEnd()) {
      auto scc = *it;
      assert(scc.size() == 1 && "without recursion, SCCs should always have size 1");
      CallGraphNode* node = scc.at(0);

      // Skip over the unused node representing calls external to the module.
      if (node->isExternal()) {
        ++it;
        continue;
      }

      auto callable = dyn_cast<ComposableCheckFuncOp>(node->getCallableRegion()->getParentOp());
      if (callable) {
        // Nondeterministic computations are useful for witgen but cannot be
        // used in constraints. All witgen side effects have already been
        // pruned, which renders almost all nondet operations dead with enough
        // inlining. Thus, inlining calls that use nondeterminism is almost
        // sufficient.
        inlineForLegality(callable, poisonUses);

        // However, sometimes constants that occur in constraints are computed
        // with nondeterministic operations on other constants. In that case, we
        // may also need to inline calls in the def chain.
        inlineForLegality(callable, poisonDefs);
      }
      ++it;
    }
  }

  void runOnOperation() override {
    generateComposableCheckFuncs();
    generateCheckFuncs();
    legalizeComposableCheckFuncs();
  }

  SmallVector<ComposableCheckFuncOp> checkFunctions;
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
      if (checkFunc.getSymName() != "check$Top")
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
