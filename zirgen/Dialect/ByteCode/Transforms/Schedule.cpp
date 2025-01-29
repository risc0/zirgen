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

struct CaptureCompare {
  using CaptureVal = std::pair<Value, size_t>;

  bool operator()(const CaptureVal& lhs, const CaptureVal& rhs) {
    return lhs.first.getImpl() < rhs.first.getImpl();
  }
};

// Registers needed to calculate a particular value, including its transitive dependencies.
struct RegsNeededInfo {
  // Allow moves but not copies.
  RegsNeededInfo(const RegsNeededInfo&) = delete;
  RegsNeededInfo(RegsNeededInfo&&) = default;

  // Previously scheduled or captured values that are needed by this
  // operation.
  llvm::SmallVector<std::pair<Value, size_t>> captures;
  size_t captureRegs = 0;

  // Maximum width of yet-to-be scheduled scheduled operations, including all the captures.
  size_t maxWidth = 0;
};

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const RegsNeededInfo& info) {
  os << "[capture regs=" << info.captureRegs << "(" << info.captures.size()
     << "), max=" << info.maxWidth << "]";
  return os;
}

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
      llvm::dbgs() << " (" << llvm::range_size(*orig) << " left):";
      for (auto result : op->getResults()) {
        if (regsNeededCache.contains(result))
          llvm::dbgs() << " " << regsNeededCache.at(result);
      }
      llvm::dbgs() << "\n";
    });
    invalidateWorkQueue.push_back(op);
    if (!op->hasOneUse())
      invalidateRegsNeeded();
  }

  void invalidateRegsNeeded() {
    while (!invalidateWorkQueue.empty()) {
      Operation* op = invalidateWorkQueue.pop_back_val();

      for (Value result : op->getResults()) {
        if (!regsNeededCache.erase(result))
          continue;
        llvm::append_range(invalidateWorkQueue, result.getUsers());
      }
    }
  }

  void makeScheduleProgress(Operation* op);
  const RegsNeededInfo& getRegsNeeded(Value val);

  Block* orig;
  std::unique_ptr<Block> scheduled;

  DenseMap<Value, RegsNeededInfo> regsNeededCache;
  AsmState asmState;
  ScheduleInterface& scheduleInterface;
  SmallVector<Operation*> invalidateWorkQueue;
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
          const RegsNeededInfo& needed = getRegsNeeded(operand);
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

      const auto& needed = getRegsNeeded(operand);
      if (!predOp || needed.maxWidth > predRegsNeeded) {
        predOp = definer;
        predRegsNeeded = needed.maxWidth;
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

const RegsNeededInfo& Scheduler::getRegsNeeded(Value topVal) {
  auto it = regsNeededCache.find(topVal);
  if (it != regsNeededCache.end()) {
    return it->second;
  }

  SmallVector<Value> workQueue;
  workQueue.push_back(topVal);
  SmallVector<std::pair<Value, const RegsNeededInfo*>> operandRegs;

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
      //
      // Track as a capture.
      regsNeededCache.try_emplace(val,
                                  RegsNeededInfo{.captures = {{val, valRegs}},
                                                 .captureRegs = valRegs,
                                                 .maxWidth = valRegs});
      workQueue.pop_back();
      continue;
    }

    bool operandsDone = true;
    operandRegs.clear();

    size_t captureCapacity = 0;
    for (Value operand : definer->getOperands()) {
      if (regsNeededCache.contains(operand)) {
        auto* needed = &regsNeededCache.at(operand);
        captureCapacity += needed->captures.size();
        operandRegs.emplace_back(val, needed);
      } else {
        workQueue.push_back(operand);
        operandsDone = false;
      }
    }

    if (!operandsDone) {
      assert(workQueue.back() != val);
      continue;
    }

    assert(operandRegs.size() == definer->getNumOperands());

    // Sort by most registers needed first
    llvm::stable_sort(operandRegs, [&](const auto& lhs, const auto& rhs) {
      return lhs.second->maxWidth > rhs.second->maxWidth;
    });

    size_t prevOperandRegs = 0;
    size_t maxLive = 0;
    llvm::SmallVector<std::pair<Value, size_t>> captures;
    llvm::SmallVector<std::pair<Value, size_t>> capturesTmp;

    for (auto& [operand, operandRegCount] : operandRegs) {
      prevOperandRegs += scheduleInterface.getValueRegs(operand);
      size_t operandMaxRegs = operandRegCount->maxWidth + prevOperandRegs;
      if (operandMaxRegs > maxLive)
        maxLive = operandMaxRegs;

      if (captures.empty()) {
        llvm::append_range(captures, operandRegCount->captures);
      } else if (!operandRegCount->captures.empty()) {
        capturesTmp.clear();
        std::set_union(captures.begin(),
                       captures.end(),
                       operandRegCount->captures.begin(),
                       operandRegCount->captures.end(),
                       std::back_inserter(capturesTmp),
                       CaptureCompare());
        std::swap(capturesTmp, captures);
      }
    }

    size_t captureRegs = 0;
    for (auto [captured, regs] : captures) {
      captureRegs += regs;
    }
    if (captureRegs > maxLive)
      maxLive = captureRegs;
    RegsNeededInfo info{
        .captures = std::move(captures), .captureRegs = captureRegs, .maxWidth = maxLive};
    /*    LLVM_DEBUG({
          llvm::dbgs() << info << " regs needed for ";
          val.print(llvm::dbgs(), asmState);
          llvm::dbgs() << "\n";
          });*/
    auto [it, didInsert] = regsNeededCache.try_emplace(val, std::move(info));
    assert(didInsert);
  }

  return regsNeededCache.at(topVal);
}

} // namespace

void scheduleBlock(Block* block, ScheduleInterface& scheduleInterface) {
  Scheduler sched(block, scheduleInterface);
  sched.scheduleAll();
  sched.finalize();
}

} // namespace zirgen::ByteCode
