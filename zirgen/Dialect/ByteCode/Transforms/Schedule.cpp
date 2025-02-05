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

#include "zirgen/Dialect/ByteCode/Transforms/Schedule.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "schedule"

using namespace mlir;

// Prioritize by "Sethi-Ulmann number".
//
// Loosely based on llvm's "Bottum Up Register Reduction" scheduling in ScheduleDAGGRRList.cpp

//
// While there exists an unscheduled operation without out edges (e.g. `set` or `return`):
//   MakeScheduleProgresess(unscheduledOp)
//
//
// MakeScheduleProgress(op):
//   For each unscheduled in-edge,
//     Recursively calculate maximum live registers required to calculate this operand
//
//    For the in-edge with the highest register use,
//      MakeScheduleProgress(in-edge definer)
//
//   If no unscheduled in-edges left, schedule op.

namespace zirgen::ByteCode {

namespace {

struct ValueUserInfo {
  // Total number of users of this value.
  size_t numUsers = 0;

  // Users of this value that we've seen so far; once this reaches numUses, we don't have to keep it
  // around anymore.
  bool seenAll = false;
  llvm::SmallDenseSet<Operation*> seen;

  bool isDone() const { return seenAll || numUsers == seen.size(); }
};

struct OpInfo {
  // Width of registers needed to gather up all the inputs to the operation.
  //
  // This ia a generalization of the Sethi-Ulmann number for DAGs that are not trees.
  // This does not include result values
  size_t inRegWidth = 0;

  // Width of registers needed to store the direct output of this
  // value, in addition to any results of any transitive dependencies
  // of this operation that have not yet been consumed.
  size_t outRegWidth = 0;
  //
  // This total includes registers to contain any user of this value
  // that has not yet been consumed.

  ssize_t queuePriority() const { return ssize_t(inRegWidth) - ssize_t(outRegWidth); }

  // Users of the result values of transitive dependencies.
  DenseMap<Value, ValueUserInfo> users;
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const OpInfo& info) {

  os << "[width=" << info.inRegWidth << " in, " << info.outRegWidth << " out, " << info.users.size()
     << " pending uses, "
     << llvm::count_if(info.users, [&](const auto& userInfo) { return userInfo.second.isDone(); })
     << " done]";
  return os;
}

class Scheduler {
public:
  struct QueueOrder {
    QueueOrder(Scheduler* scheduler) : scheduler(scheduler) {}

    bool operator()(Operation* lhs, Operation* rhs) const {
      if (rhs->getBlock() != scheduler->orig)
        return false;
      if (lhs->getBlock() != scheduler->orig)
        return true;
      const auto& lhsInfo = scheduler->opInfoCache.at(lhs);
      const auto& rhsInfo = scheduler->opInfoCache.at(rhs);
      if (lhsInfo.queuePriority() != rhsInfo.queuePriority())
        return lhsInfo.queuePriority() < rhsInfo.queuePriority();

      if (lhsInfo.inRegWidth != rhsInfo.inRegWidth)
        return lhsInfo.inRegWidth < rhsInfo.inRegWidth;

      if (lhsInfo.users.size() != rhsInfo.users.size())
        return lhsInfo.users.size() < rhsInfo.users.size();

      /*    auto& os = llvm::errs();

      os << "Roughly equal:\n";
      lhs->print(os, scheduler->asmState);
      os << ": " << lhsInfo << " \n";
      ;
      rhs->print(os, scheduler->asmState);
      os << ": " << rhsInfo << " \n";
      ;
*/

      // Among equal priority items, preserve stability by prioritizing things earlier in the block.
      return rhs->isBeforeInBlock(lhs);
      //
    };
    Scheduler* scheduler;
  };
  Scheduler(Block* orig, ScheduleInterface& scheduleInterface)
      : orig(orig)
      , scheduled(std::make_unique<Block>())
      , asmState(orig->getParentOp(), OpPrintingFlags().assumeVerified())
      , scheduleInterface(scheduleInterface) {
    LLVM_DEBUG({ orig->print(llvm::dbgs(), asmState); });
  }

  void scheduleAll();

  void finalize() {
    assert(orig->empty());
    orig->getOperations().splice(orig->getOperations().end(), scheduled->getOperations());
    LLVM_DEBUG({ orig->print(llvm::dbgs(), asmState); });
  }

private:
  void scheduleOp(Operation* op) {
    assert(op);
    assert(op->getBlock() == orig);

    if (op->getNumResults() == 1) {
      Builder builder(op->getContext());
      const auto& vi = getOpInfo(op);
      op->setAttr("inRegWidth", builder.getIndexAttr(vi.inRegWidth));
      op->setAttr("outRegWidth", builder.getIndexAttr(vi.outRegWidth));
      op->setAttr("pending", builder.getIndexAttr(llvm::count_if(vi.users, [&](auto& elem) {
        return !elem.second.isDone();
      })));
      static size_t lastMax = 10;
      if (false && vi.users.size() > lastMax) {
        auto& os = llvm::errs();
        os << "When scheduling ";
        op->print(os, asmState);
        os << ", " << vi << " has the following pending uses:\n";
        for (auto& [pend, uses] : vi.users) {

          os << "Val: ";
          pend.print(os, asmState);
          if (uses.isDone()) {
            os << ", done\n";
          } else {
            os << ", " << uses.seen.size() << " / " << uses.numUsers << ":\n";
            for (auto user : uses.seen) {
              os << "    Seen   : ";
              user->print(os, asmState);
              os << "\n";
            }
          }
        }
        lastMax *= 2;
      }
    }
    op->moveBefore(scheduled.get(), scheduled->end());

    LLVM_DEBUG({
      llvm::dbgs() << "Scheduling ";
      op->print(llvm::dbgs(), asmState);
      llvm::dbgs() << " (" << llvm::range_size(*orig) << " left)\n";
    });
  }

  void makeScheduleProgress();
  const OpInfo& getOpInfo(Operation* val);
  size_t getValueRegs(Value val);

  DenseMap<Value, size_t> valRegsCache;

  Block* orig;
  std::unique_ptr<Block> scheduled;

  DenseMap<Operation*, OpInfo> opInfoCache;
  AsmState asmState;
  ScheduleInterface& scheduleInterface;

  SmallVector<Operation*> stack;
  DenseMap<Operation*, SmallVector<Operation*>> backwardSlices;
};

void Scheduler::scheduleAll() {
  while (!orig->empty()) {
    for (Operation* op : llvm::make_pointer_range(*orig)) {
      getOpInfo(op);
      if (op->hasTrait<OpTrait::IsTerminator>() || !scheduleInterface.isPure(op)) {
        while (op->getBlock() == orig) {
          if (stack.empty())
            stack.push_back(op);
          makeScheduleProgress();
        }
        break;
      }
    }
  }
}

void Scheduler::makeScheduleProgress() {
  while (!stack.empty()) {
    Operation* op = stack.back();
    LLVM_DEBUG({
      llvm::dbgs() << "Considering ";
      op->print(llvm::dbgs(), asmState);
      llvm::dbgs() << ": " << getOpInfo(op) << "\n";
    });

    bool haveAllOperands = true;
    for (Value operand : op->getOperands()) {
      Operation* definer = operand.getDefiningOp();
      if (!definer || definer->getBlock() != orig)
        continue;

      haveAllOperands = false;
    }
    if (haveAllOperands) {
      stack.pop_back();
      scheduleOp(op);
      backwardSlices.erase(op);
      continue;
    }

    if (!backwardSlices.contains(op)) {
      BackwardSliceOptions opts;
      opts.omitBlockArguments = true;
      opts.omitUsesFromAbove = true;
      llvm::SetVector<Operation*> slice;
      getBackwardSlice(op, &slice, opts);
      assert(!slice.empty());
      backwardSlices.try_emplace(op, slice.takeVector());
    }

    const auto& slice = backwardSlices.at(op);
    auto it = llvm::max_element(slice, QueueOrder(this));

    assert(it != slice.end());
    assert((*it)->getBlock() == orig);

    stack.push_back(*it);
  }
}

size_t Scheduler::getValueRegs(Value val) {
  assert(val);
  auto it = valRegsCache.find(val);
  if (it != valRegsCache.end()) {
    return it->second;
  } else {
    size_t regs = scheduleInterface.getValueRegs(val);
    valRegsCache.try_emplace(val, regs);
    return regs;
  }
}

const OpInfo& Scheduler::getOpInfo(Operation* topOp) {
  if (opInfoCache.contains(topOp))
    return opInfoCache.at(topOp);

  SmallVector<Operation*> workQueue;
  workQueue.push_back(topOp);
  SmallVector<std::pair<Value, const OpInfo*>> operandRegs;

  while (!workQueue.empty()) {
    Operation* op = workQueue.back();
    assert(op);

    if (opInfoCache.contains(op)) {
      workQueue.pop_back();
      continue;
    }

    OpInfo info;
    assert(op->getBlock() == orig);

    bool operandsDone = true;
    operandRegs.clear();

    for (Value operand : op->getOperands()) {
      Operation* definer = operand.getDefiningOp();
      if (!definer)
        continue;
      if (!opInfoCache.contains(definer)) {
        workQueue.push_back(definer);
        operandsDone = false;
        continue;
      }
      operandRegs.emplace_back(operand, &opInfoCache.at(definer));
    }

    if (!operandsDone) {
      assert(workQueue.back() != op);
      continue;
    }

    // Sort by most registers needed first
    llvm::stable_sort(operandRegs, [&](const auto& lhs, const auto& rhs) {
      return lhs.second->queuePriority() < rhs.second->queuePriority();
    });

    // Registers that have been saved so far by operands making use of them.
    ssize_t adjustedRegs = 0;

    size_t maxLive = 0;
    for (auto& [operand, operandInfo] : operandRegs) {
      for (auto& [val, rhsUseInfo] : operandInfo->users) {
        if (info.users.contains(val)) {
          auto& useInfo = info.users[val];
          assert(useInfo.numUsers == rhsUseInfo.numUsers);

          bool wasDone = useInfo.isDone();
          if (!useInfo.seenAll)
            useInfo.seen.insert(rhsUseInfo.seen.begin(), rhsUseInfo.seen.end());
          assert(useInfo.seen.size() <= useInfo.numUsers);

          if (useInfo.isDone() && !wasDone) {
            adjustedRegs -= getValueRegs(val);

            useInfo.seenAll = true;
            useInfo.seen.clear();
          }
        } else {
          info.users[val] = rhsUseInfo;
          adjustedRegs += getValueRegs(val);
        }
      }

      assert(adjustedRegs >= 0);
      size_t opRegs = std::max<size_t>(operandInfo->inRegWidth, adjustedRegs);
      if (opRegs > maxLive)
        maxLive = opRegs;
    }

    info.inRegWidth = maxLive;

    for (Value operand : op->getOperands()) {
      if (operand.getDefiningOp()) {
        assert(info.users.contains(operand));
        assert(!info.users[operand].seenAll &&
               "We think we've seen all of them, but we haven't seen this one yet!");
        info.users[operand].seen.insert(op);
      }
    }
    for (Value result : op->getResults()) {
      info.users[result].numUsers = llvm::range_size(result.getUsers());
    }

    info.outRegWidth = 0;
    for (auto& [val, useInfo] : info.users) {
      if (!useInfo.isDone()) {
        info.outRegWidth += getValueRegs(val);
      }
    }

    //    assert(info.totSaved >= outputRegs);
    //    info.totSaved -= outputRegs;
    /*
        LLVM_DEBUG({
            llvm::dbgs() << maxLive << " regs needed for ";
            val.print(llvm::dbgs(), asmState);
            llvm::dbgs() << "\n";
            });*/
    opInfoCache[op] = info;
  }

  return opInfoCache.at(topOp);
}

} // namespace

void scheduleBlock(Block* block, ScheduleInterface& scheduleInterface) {
  Scheduler sched(block, scheduleInterface);
  sched.scheduleAll();
  sched.finalize();
}

} // namespace zirgen::ByteCode

