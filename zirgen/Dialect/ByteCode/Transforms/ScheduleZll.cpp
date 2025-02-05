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

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"
#include "zirgen/Dialect/ByteCode/Transforms/Bufferize.h"
#include "zirgen/Dialect/ByteCode/Transforms/Passes.h"
#include "zirgen/Dialect/ByteCode/Transforms/Schedule.h"
#include "zirgen/Dialect/Zll/IR/IR.h"

using namespace mlir;
using namespace zirgen::Zll;

#define DEBUG_TYPE "schedule-zll"

namespace zirgen::ByteCode {

#define GEN_PASS_DEF_SCHEDULEZLL
#define GEN_PASS_DEF_CLONEACTIVEZLL
#define GEN_PASS_DEF_CLONESIMPLEZLL
#define GEN_PASS_DEF_BUFFERIZEZLL
#include "zirgen/Dialect/ByteCode/Transforms/Passes.h.inc"

void addZllToByteCodeToPipeline(OpPassManager& pm) {
  auto& funcNest = pm.nest<mlir::func::FuncOp>();
  funcNest.addPass(createCompositeFixedPointPass(
      "optimize-for-zll-bytecode",
      [&](OpPassManager& fixedPM) {
        fixedPM.addPass(createScheduleZll());
        fixedPM.addPass(createCloneActiveZll());
      },
      /*maxIterations=*/30));
  /*    pm.addPass(createCloneSimpleZll());
pm.addPass(createLocationSnapshotPass({}, "/tmp/before-first-schedule.ir"));
pm.addPass(createScheduleZll());
pm.addPass(createLocationSnapshotPass({}, "/tmp/before-clone.ir"));
pm.addPass(createCloneActiveZll());
pm.addPass(createLocationSnapshotPass({}, "/tmp/before-second-schedule.ir"));
pm.addPass(createScheduleZll());
pm.addPass(createLocationSnapshotPass({}, "/tmp/before-bytecoding.ir"));*/
  pm.addPass(createScheduleZll());
  pm.addPass(ByteCode::createGenExecutor());
  pm.addPass(ByteCode::createEncode());
  pm.nest<ByteCode::EncodedBlockOp>().addPass(ByteCode::createBufferizeZll());
  //  pm.addPass(ByteCode::createCalcBitWidths());
}

static PassPipelineRegistration<>
    zllPipeline("zll-to-bytecode",
                "A pipeline which converts Zll functions to functions which interpret bytecode, "
                "and the associated bytecode.",
                addZllToByteCodeToPipeline);

struct ZllSchedule : public ScheduleInterface {
  size_t getValueRegs(mlir::Value value) override {
    return TypeSwitch<Type, size_t>(value.getType())
        .Case<ValType>([&](auto valType) { return valType.getFieldK(); })
        .Case<ConstraintType>([&](auto constraintType) { return 4; })
        .Default([&](auto) { return 0; });
  }

  bool isPure(mlir::Operation* op) override {
    if (llvm::isa<GetOp, GetGlobalOp, ConstOp>(op))
      return true;
    return mlir::isPure(op);
  }
};

struct ScheduleZllPass : public impl::ScheduleZllBase<ScheduleZllPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    assert(funcOp.getBody().hasOneBlock());

    Block* block = &funcOp.getBody().front();
    ZllSchedule schedule;
    scheduleBlock(block, schedule);
  }
};

namespace {

template <typename OpT> struct ClonePerUserPattern : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op, PatternRewriter& rewriter) const final {
    if (llvm::hasNItemsOrLess(op->getUsers(), 1))
      return failure();

    // Clone and pick one off
    Operation* cloned = rewriter.clone(*op);

    for (auto [oldResult, newResult] : llvm::zip_equal(op->getResults(), cloned->getResults())) {
      if (!oldResult.use_empty())
        oldResult.getUses().begin()->set(newResult);
    }
    return success();
  }
};

} // namespace

struct CloneSimpleZllPass : public impl::CloneSimpleZllBase<CloneSimpleZllPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    assert(funcOp.getBody().hasOneBlock());
    auto* ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<ClonePerUserPattern<ConstOp>>(ctx);
    patterns.insert<ClonePerUserPattern<GetOp>>(ctx);
    patterns.insert<ClonePerUserPattern<GetGlobalOp>>(ctx);
    patterns.insert<ClonePerUserPattern<TrueOp>>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

struct ZllBufferize : public BufferizeInterface {
  ZllBufferize(MLIRContext* ctx) : kind(StringAttr::get(ctx, "fpBuffer")) {}

  std::pair</*intKind=*/mlir::StringAttr, /*index=*/size_t>
  getKindAndSize(mlir::Value value) override {
    return std::make_pair(
        kind,
        TypeSwitch<Type, size_t>(value.getType())
            .Case<ValType>([&](auto valType) { return valType.getFieldK(); })
            .Case<ConstraintType>([&](auto) { return 4; })
            .Default([&](auto) -> size_t { assert(0 && "Unknown type in zll bufferize"); }));
  }
  StringAttr kind;
};

struct BufferizeZllPass : public impl::BufferizeZllBase<BufferizeZllPass> {
  void runOnOperation() override {
    EncodedBlockOp blockOp = getOperation();

    ZllBufferize zllBufferize(&getContext());
    if (failed(bufferize(blockOp, zllBufferize))) {
      blockOp->emitError("Unable to bufferize");
      signalPassFailure();
      return;
    }

    auto tempBufs = blockOp.getTempBufs();
    for (auto tempBuf : tempBufs->getAsRange<TempBufAttr>())
      maxWidth.updateMax(tempBuf.getSize());
  }
};

namespace {

constexpr bool kShowStats = true;

// State of active values
enum class ActiveState {
  Uninitialized,
  // Non-clonable value that is immediately active
  DoNotClone,
  // Non-clonable value that is no longer considered active and
  // shouldn't be accessed again until we really need it.  This is
  // similar to the concept of spilling a register, but it is not
  // moved to separate storage.
  SpilledDoNotClone,
  // Value that has been defined but not yet used.
  Defined,
  // Value that has been defined and used relatively recently and may be used again freely.
  Used,
  // Value that should be cloned upon next use.
  CloneNext
};

static llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ActiveState state) {
  switch (state) {
  case ActiveState::Uninitialized:
    return os << "Uninitialized";
  case ActiveState::DoNotClone:
    return os << "DoNotClone";
  case ActiveState::SpilledDoNotClone:
    return os << "SpilledDoNotClone";
  case ActiveState::Defined:
    return os << "Defined";
  case ActiveState::Used:
    return os << "Used";
  case ActiveState::CloneNext:
    return os << "CloneNext";
  }
  llvm_unreachable("Unhandled active state");
}

// Returns true if this state is counted as an active register.
bool isCountedState(ActiveState state) {
  switch (state) {
  case ActiveState::DoNotClone:
  case ActiveState::Defined:
  case ActiveState::Used:
    return true;
  case ActiveState::CloneNext:
  case ActiveState::SpilledDoNotClone:
    return false;
  default:
    llvm_unreachable("invalid state");
  }
}

struct ActiveRegs {
  ActiveRegs() = default;
  ActiveRegs(const ActiveRegs&) = delete;

  ZllSchedule schedule;
  size_t activeRegs = 0;
  llvm::MapVector<Value, ActiveState> active;

  using iterator = llvm::MapVector<Value, ActiveState>::iterator;
  iterator begin() { return active.begin(); }
  iterator end() { return active.end(); }

  bool contains(Value val) { return active.contains(val); }
  ActiveState get(Value val) {
    assert(contains(val));
    return active[val];
  }
  void erase(Value val) {
    assert(contains(val));
    size_t valRegs = schedule.getValueRegs(val);
    if (isCountedState(get(val))) {
      assert(valRegs <= activeRegs);
      activeRegs -= valRegs;
    }
    active.erase(val);
  }
  void set(Value val, ActiveState newState) {
    if (contains(val))
      erase(val);
    assert(!contains(val));
    if (isCountedState(newState))
      activeRegs += schedule.getValueRegs(val);
    active[val] = newState;
  }
  void update(iterator it, ActiveState newState) {
    size_t valRegs = schedule.getValueRegs(it->first);
    if (isCountedState(it->second)) {
      assert(valRegs <= activeRegs);
      activeRegs -= valRegs;
    }
    it->second = newState;
    if (isCountedState(it->second)) {
      activeRegs += valRegs;
    }
  }
};

// Set to true to verify the active register counts on every operation when running
// CloneActiveZllPass.
constexpr bool kVerifyActiveCounts = false;

} // namespace

struct CloneActiveZllPass : public impl::CloneActiveZllBase<CloneActiveZllPass> {
  using CloneActiveZllBase::CloneActiveZllBase;
  DenseSet<Operation*> cachedDoClone, cachedDontClone;

  bool shouldClone(Operation* op) {
    if (cachedDontClone.contains(op))
      return false;
    if (cachedDoClone.contains(op))
      return true;
    if (calcShouldClone(op)) {
      assert(cachedDoClone.contains(op));
      return true;
    } else {
      assert(cachedDontClone.contains(op));
      return false;
    }
  }

  bool calcShouldClone(Operation* op) {
    SmallVector<Operation*> workList;
    workList.push_back(op);
    DenseSet<Operation*> seen;

    size_t getCount = 0;
    size_t totCount = 0;

    size_t pos = 0;
    while (pos < workList.size()) {
      Operation* work = workList[pos];

      if (cachedDontClone.contains(work)) {
        cachedDontClone.insert(op);
        return false;
      }

      if (!llvm::isa<PolyOp, AndEqzOp, AndCondOp>(work)) {
        cachedDontClone.insert(work);
        cachedDontClone.insert(op);
        return false;
      }

      if (llvm::isa<GetOp>(work)) {
        ++getCount;
        if (getCount > maxCloneGets) {
          cachedDontClone.insert(op);
          return false;
        }
      }
      ++totCount;
      if (totCount > maxCloneOps) {
        cachedDontClone.insert(op);
        return false;
      }

      for (Value operand : work->getOperands()) {
        if (auto definer = operand.getDefiningOp()) {
          if (!seen.insert(definer).second)
            continue;
          workList.push_back(definer);
        }
      }

      ++pos;
    }

    cachedDoClone.insert(workList.begin(), workList.end());

    return true;
  }

  Operation* cloneBefore(Operation* op, Value val, AsmState& asmState) {
    Operation* definer = val.getDefiningOp();
    assert(definer);

    OpBuilder builder(op);
    SmallVector<Operation*> workList;
    workList.push_back(definer);

    IRMapping cloneMapper;
    while (!workList.empty()) {
      Operation* work = workList.back();
      if (cloneMapper.contains(work)) {
        workList.pop_back();
        continue;
      }
      bool needOperands = false;
      for (Value operand : work->getOperands()) {
        Operation* operandDefiner = operand.getDefiningOp();
        if (!operandDefiner)
          continue;
        if (!cloneMapper.contains(operandDefiner)) {
          workList.push_back(operandDefiner);
          needOperands = true;
        }
      }
      if (needOperands) {
        // Need to clone operands first.
        continue;
      }
      builder.clone(*work, cloneMapper);
      ++opClones;
      workList.pop_back();
    }

    auto& domInfo = getAnalysis<DominanceInfo>();
    for (auto oldVal : definer->getResults()) {
      Value newVal = cloneMapper.lookup(oldVal);
      bool replacedAny = false;
      oldVal.replaceUsesWithIf(newVal, [&](OpOperand& user) {
        bool doReplace = domInfo.dominates(op, user.getOwner());
        if (doReplace) {
          LLVM_DEBUG({
            llvm::dbgs() << "Replacing on ";
            user.getOwner()->print(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });
          replacedAny = true;
        }
        return doReplace;
      });
      assert(replacedAny);
    }
    return cloneMapper.lookup(definer);
  }

  void runOnOperation() override {
    cachedDoClone.clear();
    cachedDontClone.clear();
    AsmState asmState(getOperation());
    func::FuncOp funcOp = getOperation();
    assert(funcOp.getBody().hasOneBlock());
    Block* block = &funcOp.getBody().front();

    auto& liveness = getAnalysis<Liveness>();
    ZllSchedule schedule;

    ActiveRegs active;
    for (auto liveIn : liveness.getLiveIn(block))
      active.set(liveIn, ActiveState::DoNotClone);
    assert(liveness.getLiveOut(block).empty());

    maxBlockOps.updateMax(llvm::range_size(*block));

    auto addActiveResults = [&](ValueRange results) {
      for (Value res : results) {
        assert(!active.contains(res));
        assert(!res.use_empty());
        Operation* definer = res.getDefiningOp();
        if (definer && shouldClone(definer))
          active.set(res, ActiveState::Defined);
        else
          active.set(res, ActiveState::DoNotClone);

        LLVM_DEBUG({ llvm::dbgs() << "Adding result " << res << ": " << active.get(res) << "\n"; });

        maxRegsActive.updateMax(active.activeRegs);
      }
    };

    size_t opPos = 0;
    size_t blockSize = llvm::range_size(*block);
    for (Operation* op : llvm::make_pointer_range(*block)) {
      LLVM_DEBUG({
        llvm::dbgs() << "\nBefore ";
        op->print(llvm::dbgs(), asmState);
        llvm::dbgs() << *op << ", live regs " << active.activeRegs << ":\n";
        for (auto [val, state] : active) {
          llvm::dbgs() << "  ";
          val.print(llvm::dbgs(), asmState);
          llvm::dbgs() << ": " << state << "\n";
        }
      });

      if (kVerifyActiveCounts) {
        size_t verifyActiveRegs = 0;
        for (auto [liveVal, state] : active) {
          if (state != ActiveState::CloneNext && state != ActiveState::SpilledDoNotClone)
            verifyActiveRegs += schedule.getValueRegs(liveVal);
        }
        if (verifyActiveRegs != active.activeRegs) {
          llvm::errs() << "Expecting " << verifyActiveRegs << " active but have "
                       << active.activeRegs << "\n";

          llvm::errs() << "\nBefore " << *op << ":\n";
          for (auto [val, state] : active) {
            llvm::errs() << "  " << val << ": " << state << "\n";
          }
          abort();
        }
      }

      if (active.activeRegs <= targetRegs)
        ++withinTarget;
      else
        ++exceedTarget;

      for (auto it = active.begin(); it != active.end(); ++it) {
        if (active.activeRegs <= targetRegs)
          break;

        auto [activeVal, state] = *it;

        if (state == ActiveState::DoNotClone) {
          LLVM_DEBUG({
            llvm::dbgs() << "Spilling ";
            activeVal.print(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });
          llvm::errs() << "Spilling ";
          activeVal.print(llvm::errs(), asmState);
          llvm::errs() << "\n";
          active.update(it, ActiveState::SpilledDoNotClone);
          ++opSpillVals;
        } else if (state == ActiveState::Used) {
          LLVM_DEBUG({
            llvm::dbgs() << "Requesting to clone on next use: ";
            activeVal.print(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });
          active.update(it, ActiveState::CloneNext);
          ++opCloneVals;
        }
      }

      LLVM_DEBUG(
          { llvm::dbgs() << "Active: " << active.activeRegs << " / " << targetRegs << "\n"; });

      // Only process each operand once.
      for (Value operand : op->getOperands()) {
        if (active.contains(operand) && active.get(operand) == ActiveState::CloneNext) {
          active.erase(operand);

          LLVM_DEBUG({
            llvm::dbgs() << "Cloning ";
            operand.print(llvm::dbgs(), asmState);
            llvm::dbgs() << " before ";
            op->print(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });

          Operation* cloned = cloneBefore(op, operand, asmState);
          addActiveResults(cloned->getResults());
        }
      }

      addActiveResults(op->getResults());

      for (Value operand : op->getOperands()) {
        if (!active.contains(operand))
          continue;
        auto state = active.get(operand);
        if (state == ActiveState::Defined) {
          active.set(operand, ActiveState::Used);
          LLVM_DEBUG({
            llvm::dbgs() << "First use: ";
            operand.print(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });
        } else if (state == ActiveState::Used) {
          // Bump it to most recently used so it doesn't get spilled right away
          active.set(operand, ActiveState::Used);
        } else if (state == ActiveState::DoNotClone) {
          // Bump it to most recently used so it doesn't get spilled right away
          active.set(operand, ActiveState::DoNotClone);
        } else if (state == ActiveState::SpilledDoNotClone) {
          active.set(operand, ActiveState::DoNotClone);
          LLVM_DEBUG({
            llvm::dbgs() << "Resurrect ";
            operand.print(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
          });
        }

        if (liveness.isDeadAfter(operand, op)) {
          LLVM_DEBUG({
            llvm::dbgs() << "Last use: ";
            operand.print(llvm::dbgs(), asmState);
            llvm::dbgs() << " state " << active.get(operand) << "\n";
          });
          if (active.contains(operand))
            active.erase(operand);
        }
      }
      ++opPos;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "At end, live regs " << active.activeRegs << ":\n";
      for (auto [val, state] : active) {
        llvm::dbgs() << "  " << val << ": " << state << "\n";
      }
    });
    assert(active.activeRegs == 0);

    if (kShowStats) {
      // TODO: Figure out why --mlir-print-op-stats doesn't show our statistics. :(
      llvm::outs() << "\nCloneZLL pass finished\n";
      for (auto stat : getStatistics()) {
        llvm::outs() << stat->getName() << ": " << stat->getValue() << "\n";
        stat->Value = 0;
      }
      llvm::outs() << "\n";
    }
  }
};

} // namespace zirgen::ByteCode
