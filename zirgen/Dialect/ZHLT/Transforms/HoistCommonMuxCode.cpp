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

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"

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
      !isa<AliasLayoutOp>(op) &&
      !op->hasTrait<OpTrait::IsTerminator>() &&
      llvm::all_of(op->getOperands(), [=](Value value) {
        return value.getParentRegion() != op->getParentRegion();
      });
}

bool compare(Operation* op1, Operation* op2) {
  if (op1->getName() != op2->getName() ||
      op1->getNumOperands() != op2->getNumOperands() ||
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

// For structure-like components, if two members are equal in the PackOp at the
// end of the constructor, those members will ultimately be equal in all other
// situations, such as when reconstructing an instance from a back and when
// zero-initializing (trivially, since both members are zeroed).
struct HoistCommonMuxCodePass : public HoistCommonMuxCodeBase<HoistCommonMuxCodePass> {
  HoistCommonMuxCodePass() = default;
  HoistCommonMuxCodePass(const HoistCommonMuxCodePass& pass) {}

  void runOnOperation() override {
    getOperation().walk<WalkOrder::PostOrder>([&](SwitchOp mux) {
      // If code in the mux is shared by multiple but not all mux arms, then
      // hoisting it reduces code size but increases execution cost. Since we
      // want code shared by all arms, search the first mux arm for hoistable
      // operations, and then search the other mux arms for matching operations.
      auto it = mux.getRegion(0).op_begin();
      while (it != mux->getRegion(0).op_end()) {
        Operation& op = *(it++);
        if (isHoistable(&op)) {
          SmallVector<Operation*> toHoist;
          toHoist.reserve(mux.getArms().size());
          toHoist.push_back(&op);

          bool hoistable = llvm::all_of(mux.getRegions(), [&](Region* region) {
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

          if (hoistable) {
            LLVM_DEBUG(llvm::dbgs() << "hoist: " << *toHoist[0] << "\n");
            toHoist[0]->moveBefore(mux);
            for (size_t i = 1; i < toHoist.size(); ++i) {
              toHoist[i]->replaceAllUsesWith(toHoist[0]->getResults());
              toHoist[i]->erase();
              ++opsDeleted;
            }
          }
        }
      }
    });
  }

  Statistic opsDeleted{this, "opsDeleted", "number of operations saved by mux hoisting"};
};

std::unique_ptr<OperationPass<ModuleOp>> createHoistCommonMuxCodePass() {
  return std::make_unique<HoistCommonMuxCodePass>();
}

} // namespace zirgen::Zhlt
