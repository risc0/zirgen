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

class Scheduler {
public:
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
    op->moveBefore(scheduled.get(), scheduled->end());

    LLVM_DEBUG({
      llvm::dbgs() << "Scheduling ";
      op->print(llvm::dbgs(), asmState);
      llvm::dbgs() << " (" << llvm::range_size(*orig) << " left)\n";
    });
    if (!op->hasOneUse())
      invalidateRegsNeeded(op);
  }

  void invalidateRegsNeeded(Operation* topOp) {
    SmallVector<Operation*> workQueue;
    workQueue.push_back(topOp);

    while (!workQueue.empty()) {
      Operation* op = workQueue.pop_back_val();

      for (Value result : op->getResults()) {
        if (!regsNeededCache.erase(result))
          continue;
        llvm::append_range(workQueue, result.getUsers());
      }
    }
  }

  void makeScheduleProgress(Operation* op);
  size_t getRegsNeeded(Value val);

  Block* orig;
  std::unique_ptr<Block> scheduled;

  DenseMap<Value, size_t> regsNeededCache;
  AsmState asmState;
  ScheduleInterface& scheduleInterface;
};

void Scheduler::scheduleAll() {
  while (!orig->empty()) {
    bool madeProgress = false;
    for (Operation* op : llvm::make_pointer_range(*orig)) {
      if (scheduleInterface.isPure(op) && !op->hasTrait<OpTrait::IsTerminator>())
        continue;

      while (op->getBlock() == orig) {
        makeScheduleProgress(op);
      }
      madeProgress = true;
      break;
    }
    if (!madeProgress) {
      if (!orig->empty()) {
        llvm::errs() << "Didn't expected unused operations:\n";
        orig->print(llvm::errs(), asmState);
        orig->clear();
        exit(1);
      }
      return;
    }
  }
}

void Scheduler::makeScheduleProgress(Operation* op) {
  assert(op);

  for (;;) {
    LLVM_DEBUG({
      llvm::dbgs() << "Considering ";
      op->print(llvm::dbgs(), asmState);
      if (!op->getOperands().empty()) {
        llvm::dbgs() << "(needed by operands:";
        for (Value operand : op->getOperands()) {
          size_t needed = getRegsNeeded(operand);
          Operation* definer = operand.getDefiningOp();
          if (!definer || definer->getBlock() != orig)
            llvm::dbgs() << " (" << needed << ")";
          else
            llvm::dbgs() << " " << needed;
        }
        llvm::dbgs() << ")";
      }
      llvm::dbgs() << "\n";
    });

    SmallVector<std::pair</*regs needed=*/ssize_t, Value>> operands;

    Operation* predOp = nullptr;
    size_t predRegsNeeded = 0;

    for (Value operand : op->getOperands()) {
      Operation* definer = operand.getDefiningOp();
      if (!definer || definer->getBlock() != orig)
        continue;

      size_t needed = getRegsNeeded(operand);
      if (!predOp || needed > predRegsNeeded) {
        predOp = definer;
        predRegsNeeded = needed;
      }
    }

    if (predOp) {
      // Operand needs to be scheduled first.
      op = predOp;
      continue;
    }

    scheduleOp(op);
    return;
  }
}

size_t Scheduler::getRegsNeeded(Value topVal) {
  if (regsNeededCache.contains(topVal))
    return regsNeededCache.lookup(topVal);

  SmallVector<Value> workQueue;
  workQueue.push_back(topVal);
  SmallVector<std::pair<Value, size_t>> operandRegs;

  while (!workQueue.empty()) {
    Value val = workQueue.back();

    if (regsNeededCache.contains(val)) {
      workQueue.pop_back();
      continue;
    }

    size_t valRegs = scheduleInterface.getValueRegs(val);

    Operation* definer = val.getDefiningOp();
    if (!definer || definer->getBlock() != orig) {
      // This operation is not in the scheduling queue; it's either already been scheduled or is
      // outside our scope.
      // Increase its priority so we use it sooner.
      regsNeededCache[val] = 2 * valRegs;
      workQueue.pop_back();
      continue;
    }

    bool operandsDone = true;
    operandRegs.clear();

    for (Value operand : definer->getOperands()) {
      if (!regsNeededCache.contains(operand)) {
        workQueue.push_back(operand);
        operandsDone = false;
      }
      operandRegs.emplace_back(val, regsNeededCache.lookup(operand));
    }

    if (!operandsDone) {
      assert(workQueue.back() != val);
      continue;
    }

    assert(operandRegs.size() == definer->getNumOperands());

    // Sort by most registers needed first
    llvm::stable_sort(operandRegs,
                      [&](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });

    size_t prevOperandRegs = 0;
    size_t maxLive = 0;
    for (auto& [operand, operandRegCount] : operandRegs) {
      prevOperandRegs += scheduleInterface.getValueRegs(operand);
      size_t operandMaxRegs = operandRegCount + prevOperandRegs;
      if (operandMaxRegs > maxLive)
        maxLive = operandMaxRegs;
    }

    size_t outputRegs = 0;
    for (Value result : definer->getResults()) {
      outputRegs += scheduleInterface.getValueRegs(result);
    }
    if (outputRegs > maxLive)
      maxLive = outputRegs;

    /*    LLVM_DEBUG({
          llvm::dbgs() << maxLive << " regs needed for ";
          val.print(llvm::dbgs(), asmState);
          llvm::dbgs() << "\n";
          });*/
    regsNeededCache[val] = maxLive;
  }

  return regsNeededCache.lookup(topVal);
}

} // namespace

void scheduleBlock(Block* block, ScheduleInterface& scheduleInterface) {
  Scheduler sched(block, scheduleInterface);
  sched.scheduleAll();
  sched.finalize();
}

} // namespace zirgen::ByteCode
