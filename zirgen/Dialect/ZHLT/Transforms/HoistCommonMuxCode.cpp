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

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mux-hoisting"

using namespace mlir;
using namespace zirgen::ZStruct;

namespace zirgen::Zhlt {

namespace {

bool isHoistable(Operation* op) {
  // Check if all of the operation's operands are defined outside of the mux. If
  // they are, then we can probably hoist -- but make sure not to reorder side
  // effects, and never hoist any ops with regions, AliasLayoutOps or block
  // terminators.
  return op->getNumRegions() == 0 &&
         !isa<LoadOp>(op) && // could be more precise by checking for writes within the block
         !isa<AliasLayoutOp>(op) && !op->hasTrait<OpTrait::IsTerminator>() &&
         llvm::all_of(op->getOperands(), [=](Value value) {
           return value.getParentRegion() != op->getParentRegion();
         });
}

bool compare(Operation* op1, Operation* op2) {
  if (op1->getName() != op2->getName() || op1->getNumOperands() != op2->getNumOperands() ||
      op1->getAttrs().size() != op2->getAttrs().size())
    return false;
  for (auto [opn1, opn2] : llvm::zip(op1->getOperands(), op2->getOperands())) {
    if (opn1 != opn2)
      return false;
  }
  for (auto [attr1, attr2] : llvm::zip(op1->getAttrs(), op2->getAttrs())) {
    if (attr1 != attr2)
      return false;
  }
  return true;
}

} // namespace

// Look for code shared between mux arms, and hoist it. If code is common to at
// least two arms, hoisting it reduces code size. However, this hurts code speed
// because that code will be unnecessarily executed for any mux arms where it
// didn't previously occur. That being said, this is always a beneficial
// optimization if the code occurs on *every* mux arm.
struct HoistCommonMuxCodePass : public HoistCommonMuxCodeBase<HoistCommonMuxCodePass> {
  HoistCommonMuxCodePass() = default;
  HoistCommonMuxCodePass(bool eager) { this->eager = eager; }
  HoistCommonMuxCodePass(const HoistCommonMuxCodePass& pass) {}

  void runOnOperation() override {
    getOperation().walk<WalkOrder::PostOrder>([&](SwitchOp mux) {
      // If code in the mux is shared by multiple but not all mux arms, then
      // hoisting it reduces code size but increases execution cost. We at worst
      // break even on execution if it's shared by all mux arms.
      if (eager) {
        hoistForSize(mux);
      } else {
        hoistForSpeed(mux);
      }
    });
  }

  void hoistForSpeed(SwitchOp mux) {
    // We're looking for operations shared in all mux arms, and all such
    // operations must also be in the first mux arm. Since we move operations
    // in region 0 before the mux in `doHoist`, we use this iterator-based
    // loop with a post-increment to keep track of which operation we're
    // considering hoisting.
    auto it = mux.getRegion(0).op_begin();
    while (it != mux->getRegion(0).op_end()) {
      Operation& op = *(it++);
      if (isHoistable(&op)) {
        SmallVector<Operation*> toHoist;
        toHoist.reserve(mux.getArms().size());
        toHoist.push_back(&op);
        if (shouldHoistForSpeed(mux, op, toHoist)) {
          doHoist(mux, toHoist);
        }
      }
    }
  }

  // Return true if hoisting `op` out of `mux` doesn't add redundant execution
  bool shouldHoistForSpeed(SwitchOp mux, Operation& op, SmallVector<Operation*>& toHoist) {
    // Check if `op` occurs in all arms of `mux`. We already know it occurs in
    // the first arm, and since it's hoistable in the first arm it must be
    // hoistable in all other arms, so skip these checks.
    return llvm::all_of(mux.getRegions(), [&](Region* region) {
      if (region->getRegionNumber() == 0)
        return true;

      return llvm::any_of(region->getOps(), [&](Operation& op2) {
        if (compare(&op, &op2)) {
          toHoist.push_back(&op2);
          return true;
        }
        return false;
      });
    });
  }

  void hoistForSize(SwitchOp mux) {
    // We're looking for operations shared in two or more mux arms, which means
    // we need to consider operations from any pair of mux arms.
    for (size_t i = 0; i < mux->getNumRegions(); ++i) {
      Region& region = mux.getRegion(i);
      auto it = region.op_begin();
      while (it != region.op_end()) {
        Operation& op = *(it++);
        if (isHoistable(&op)) {
          SmallVector<Operation*> toHoist;
          toHoist.reserve(mux.getArms().size());
          toHoist.push_back(&op);
          if (shouldHoistForSize(mux, op, i, toHoist)) {
            doHoist(mux, toHoist);
          }
        }
      }
    }
  }

  // Return true if hoisting `op` out of `mux` reduces code size
  bool shouldHoistForSize(SwitchOp mux, Operation& op, size_t i, SmallVector<Operation*>& toHoist) {
    // Check if op occurs in multiple arms of `mux`. We already know it occurs
    // in the first arm, so check if any of the other arms have it as well.
    // Since it's hoistable in the first arm, it's also hoistable in all other
    // arms, so we don't need to check again.
    bool shouldHoist = false;
    for (size_t j = i + 1; j < mux->getNumRegions(); ++j) {
      shouldHoist |= llvm::any_of(mux.getRegion(j).getOps(), [&](Operation& op2) {
        if (compare(&op, &op2)) {
          toHoist.push_back(&op2);
          return true;
        }
        return false;
      });
    }
    return shouldHoist;
  }

  void doHoist(SwitchOp mux, ArrayRef<Operation*> toHoist) {
    LLVM_DEBUG(llvm::dbgs() << "hoist: " << *toHoist[0] << "\n");
    toHoist[0]->moveBefore(mux);
    for (size_t i = 1; i < toHoist.size(); ++i) {
      toHoist[i]->replaceAllUsesWith(toHoist[0]->getResults());
      toHoist[i]->erase();
      ++opsDeleted;
    }
  }

  Statistic opsDeleted{this, "opsDeleted", "number of operations saved by mux hoisting"};
};

std::unique_ptr<OperationPass<ModuleOp>> createHoistCommonMuxCodePass() {
  return std::make_unique<HoistCommonMuxCodePass>(false);
}

std::unique_ptr<OperationPass<ModuleOp>> createHoistCommonMuxCodePass(bool eager) {
  return std::make_unique<HoistCommonMuxCodePass>(eager);
}

} // namespace zirgen::Zhlt
