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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen::Zll {

namespace {

struct Sinker {
  DenseMap</*factor=*/Value, /*constraints=*/DenseSet<EqualZeroOp>> factorConstraints;
  DenseMap</*constraint=*/EqualZeroOp, /*factors=*/DenseSet<Value>> constraintFactors;

  size_t idx = 0;

  // Keep a sorting key so we can make the order deterministic even after we go through a densemap.
  DenseMap<Value, size_t> factorOrder;
  DenseMap<EqualZeroOp, size_t> eqzOpOrder;

  void countFactors(EqualZeroOp eqzOp, Value factor) {
    auto mulOp = factor.getDefiningOp<MulOp>();
    if (mulOp) {
      countFactors(eqzOp, mulOp.getLhs());
      countFactors(eqzOp, mulOp.getRhs());
    } else {
      ++idx;
      auto& eqzOrder = eqzOpOrder[eqzOp];
      eqzOrder = idx;
      auto& facOrder = factorOrder[factor];
      facOrder = idx;
      factorConstraints[factor].insert(eqzOp);
      constraintFactors[eqzOp].insert(factor);
    }
  }

  void sinkConstraints(Block* block) {
    for (EqualZeroOp eqzOp : block->getOps<EqualZeroOp>()) {
      countFactors(eqzOp, eqzOp.getIn());
    };

    for (;;) {
      Value bestFactor;
      size_t numBestFactor = 0;
      for (auto [k, v] : factorConstraints) {
        if (v.size() <= 1)
          continue;
        if (v.size() > numBestFactor ||
            (v.size() == numBestFactor && factorOrder.at(k) < factorOrder.at(bestFactor))) {
          numBestFactor = v.size();
          bestFactor = k;
        }
      }

      if (numBestFactor <= 1)
        break;
      assert(bestFactor);

      SmallVector<EqualZeroOp> ops = llvm::to_vector(factorConstraints.at(bestFactor));
      assert(ops.size() > 1);
      llvm::sort(ops, [&](auto a, auto b) -> bool { return eqzOpOrder.at(a) < eqzOpOrder.at(b); });

      OpBuilder builder = OpBuilder::atBlockTerminator(block);
      auto ifOp = builder.create<IfOp>(bestFactor.getLoc(), bestFactor);
      builder.createBlock(&ifOp.getInner());
      auto termOp = builder.create<Zll::TerminateOp>(bestFactor.getLoc());

      for (EqualZeroOp eqzOp : ops) {
        assert(eqzOp->getBlock() == block);
        eqzOp->moveBefore(termOp);
        builder.setInsertionPoint(eqzOp);

        Value newCond;
        SmallVector<Value> otherFactors = llvm::to_vector(constraintFactors.at(eqzOp));
        llvm::sort(otherFactors,
                   [&](auto a, auto b) -> bool { return factorOrder.at(a) < factorOrder.at(b); });
        for (Value otherFactor : otherFactors) {
          bool didErase = factorConstraints[otherFactor].erase(eqzOp);
          assert(didErase);
          if (otherFactor != bestFactor) {
            if (newCond) {
              newCond = builder.create<Zll::MulOp>(eqzOp.getLoc(), otherFactor, newCond);
            } else {
              newCond = otherFactor;
            }
          }
          if (!newCond) {
            // TODO: Why does this happen?
            newCond = builder.create<Zll::ConstOp>(eqzOp.getLoc(), 1);
          }
          eqzOp.getInMutable().set(newCond);
        }
        constraintFactors.erase(eqzOp);
      }
    }
  }
};

struct MultiplyToIfPass : public MultiplyToIfBase<MultiplyToIfPass> {
  void runOnOperation() override {
    getOperation()->walk<WalkOrder::PreOrder>([&](Block* block) {
      Sinker sinker;
      sinker.sinkConstraints(block);
    });
  }
};

} // End namespace

std::unique_ptr<Pass> createMultiplyToIfPass() {
  return std::make_unique<MultiplyToIfPass>();
}

} // namespace zirgen::Zll
