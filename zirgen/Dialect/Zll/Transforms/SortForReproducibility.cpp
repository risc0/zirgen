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

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/PassDetail.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"

using namespace mlir;

namespace zirgen::Zll {

namespace {

struct RemoveEqualZero : public OpRewritePattern<EqualZeroOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(EqualZeroOp op, PatternRewriter& rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct SortForReproducibilityPass : public SortForReproducibilityBase<SortForReproducibilityPass> {
  void runOnOperation() override {
    StringSet<llvm::BumpPtrAllocator> propStorage;
    DenseMap<Operation*, StringRef> propInfo;
    DenseMap<Operation*, size_t> argPositions;

    // Save some some things we can use to deterministically sort operations.
    getOperation()->walk([&](Operation* op) {
      for (auto [i, arg] : llvm::enumerate(op->getOperands())) {
        auto definer = arg.getDefiningOp();
        if (definer) {
          auto& pos = argPositions[definer];
          pos += i;
        }
      }
    });

    std::function<StringRef(Operation*)> getPropInfo = [&](Operation* op) -> StringRef {
      auto propIt = propInfo.find(op);
      if (propIt != propInfo.end())
        return propIt->second;

      std::string props = op->getName().getStringRef().str();
      llvm::raw_string_ostream os(props);
      op->getPropertiesAsAttribute().print(os);

      auto [it, didInsert] = propStorage.insert(props);
      return propInfo[op] = it->first();
    };

    std::function<bool(Operation & a, Operation & b)> operationCompare, cachedOperationCompare;
    operationCompare = [&](Operation& a, Operation& b) {
      if (&a == &b)
        return false;

      StringRef aInfo = getPropInfo(&a);
      StringRef bInfo = getPropInfo(&b);
      if (aInfo != bInfo) {
        return aInfo < bInfo;
      }

      size_t aPos = argPositions.lookup(&a);
      size_t bPos = argPositions.lookup(&b);
      if (aPos != bPos) {
        return aPos < bPos;
      }

      if (a.getNumOperands() != b.getNumOperands()) {
        return a.getNumOperands() < b.getNumOperands();
      }

      // If nothing else works, recursively compare operands
      for (auto [aOperand, bOperand] : llvm::zip_equal(a.getOperands(), b.getOperands())) {
        auto aDefiner = llvm::dyn_cast<OpResult>(aOperand);
        auto bDefiner = llvm::dyn_cast<OpResult>(bOperand);
        if (bool(aDefiner) != bool(bDefiner))
          return bool(aDefiner) < bool(bDefiner);

        if (aDefiner && bDefiner) {
          if (aDefiner.getResultNumber() != bDefiner.getResultNumber())
            return aDefiner.getResultNumber() < bDefiner.getResultNumber();

          if (cachedOperationCompare(*aDefiner.getOwner(), *bDefiner.getOwner()))
            return true;
          if (cachedOperationCompare(*bDefiner.getOwner(), *aDefiner.getOwner()))
            return false;
        }

        auto aArg = llvm::dyn_cast<BlockArgument>(aOperand);
        auto bArg = llvm::dyn_cast<BlockArgument>(bOperand);

        if (bool(aArg) != bool(bArg))
          return bool(aArg) < bool(bArg);

        if (aArg && bArg) {
          if (aArg.getArgNumber() != bArg.getArgNumber())
            return aArg.getArgNumber() < bArg.getArgNumber();
        }
      }

      return false;
    };

    llvm::DenseMap<std::pair<Operation*, Operation*>, bool> compareResults;
    cachedOperationCompare = [&](Operation& a, Operation& b) {
      auto [it, didInsert] = compareResults.try_emplace(std::make_pair(&a, &b), false);
      if (didInsert) {
        bool lessThan = operationCompare(a, b);
        // We can't use the original iterator since it may have been invalidated.
        compareResults[std::make_pair(&a, &b)] = lessThan;
        return lessThan;
      } else {
        return it->second;
      }
    };

    getOperation()->walk([&](Block* inner) {
      auto shouldSort = [&](Operation* op) {
        if (op->hasTrait<OpTrait::IsTerminator>())
          return false;
        if (isPure(op))
          return true;
        return false;
      };

      Block doneBlock;

      // Find runs of pure operations all in a row, and sort them.
      auto it = inner->getOperations().begin();
      while (it != inner->getOperations().end()) {
        if (!shouldSort(&*it)) {
          ++it;
          continue;
        }

        // Move the section of operations we're not sorting into doneBlock.
        doneBlock.getOperations().splice(doneBlock.getOperations().end(),
                                         inner->getOperations(),
                                         inner->getOperations().begin(),
                                         it);
        it = inner->getOperations().begin();

        while (it != inner->getOperations().end() && shouldSort(&*it)) {
          ++it;
        }

        Block sortBlock;
        // Move the sections of operations we *are* sorting into sortBlock
        sortBlock.getOperations().splice(sortBlock.getOperations().end(),
                                         inner->getOperations(),
                                         inner->getOperations().begin(),
                                         it);
        sortBlock.getOperations().sort(cachedOperationCompare);

        // After we sort reproducibly, we have to topologically sort to make sure we don't screw up
        // SSA dominance.
        mlir::sortTopologically(&sortBlock);

        // Append the sorted section to doneBlock.
        doneBlock.getOperations().splice(doneBlock.getOperations().end(),
                                         sortBlock.getOperations(),
                                         sortBlock.getOperations().begin(),
                                         sortBlock.getOperations().end());
        it = inner->getOperations().begin();
      }

      // Move all the operations we've processed out from doneBlock into the original block.
      inner->getOperations().splice(inner->getOperations().begin(),
                                    doneBlock.getOperations(),
                                    doneBlock.getOperations().begin(),
                                    doneBlock.getOperations().end());
    });
  }
};

} // End namespace

std::unique_ptr<Pass> createSortForReproducibilityPass() {
  return std::make_unique<SortForReproducibilityPass>();
}

} // namespace zirgen::Zll
