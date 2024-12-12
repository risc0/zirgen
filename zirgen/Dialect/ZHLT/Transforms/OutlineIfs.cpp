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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"
#include "zirgen/Dialect/ZStruct/Transforms/RewritePatterns.h"
#include "llvm/Support/Debug.h"

#include <set>
#include <vector>

using namespace mlir;
using namespace zirgen::Zll;
using namespace zirgen::ZStruct;

namespace zirgen::Zhlt {

namespace {

// Ugh, apparently makeRegionIsolatedFromAbove doesn't recurse into subregions so we have to
// do it ourself.
struct Outliner {
  Region topRegion;
  IRMapping mapper;
  OpBuilder builder;
  SmallVector<Value> capture;

  Outliner(MLIRContext* ctx) : builder(ctx) { builder.createBlock(&topRegion); }

  void ensureMapped(Value val) {
    if (mapper.contains(val))
      return;

    if (auto opVal = llvm::dyn_cast<OpResult>(val)) {
      Operation* owner = opVal.getOwner();
      insertCloned(owner);
      assert(mapper.contains(val));
    } else {
      mapper.map(val, topRegion.addArgument(val.getType(), val.getLoc()));
      capture.push_back(val);
    }
  }

  void insertCloned(Operation* op) { insertCloned(builder.getBlock(), op); }

  void insertCloned(Block* block, Operation* op) {
    for (auto arg : op->getOperands()) {
      ensureMapped(arg);
    }
    Operation* cloned = op->cloneWithoutRegions(mapper);
    block->push_back(cloned);

    for (auto [oldRegion, newRegion] : llvm::zip_equal(op->getRegions(), cloned->getRegions())) {
      cloneRegion(oldRegion, newRegion);
    }
  }

  void cloneRegion(Region& oldRegion, Region& newRegion) {
    assert(newRegion.empty());
    Block* block = &newRegion.emplaceBlock();

    for (auto& op : oldRegion.front()) {
      insertCloned(block, &op);
    }
  }
};

struct OutlineIfsPass : public OutlineIfsBase<OutlineIfsPass> {
  void outlineIfs(StepFuncOp funcOp) {
    OpBuilder builder(&getContext());

    size_t idx = 0;
    funcOp.walk<WalkOrder::PostOrder>([&](IfOp ifOp) {
      Outliner outliner(&getContext());

      for (auto& op : *ifOp.getBody()) {
        outliner.insertCloned(&op);
      }

      std::string newFuncName = (funcOp.getSymName() + "_" + std::to_string(idx++)).str();
      builder.setInsertionPointToEnd(getOperation().getBody());
      auto newFunc = builder.create<StepFuncOp>(
          ifOp.getLoc(),
          builder.getStringAttr(newFuncName),
          TypeAttr::get(builder.getFunctionType(/*inputs=*/ValueRange(outliner.capture).getTypes(),
                                                /*resultTypes=*/{})),
          /*visibility=*/mlir::StringAttr{},
          /*argAttrs=*/mlir::ArrayAttr{},

          /*res_attrs=*/mlir::ArrayAttr{});

      newFunc.getBody().takeBody(outliner.topRegion);

      newFunc.getBody().front().getTerminator()->erase();
      builder.setInsertionPointToEnd(&newFunc.getBody().front());
      builder.create<Zhlt::ReturnOp>(ifOp.getLoc());

      // Replace the original body of the if statement with a call
      ifOp.getBody()->clear();
      builder.setInsertionPointToEnd(ifOp.getBody());
      builder.create<StepCallOp>(ifOp.getLoc(), newFunc, outliner.capture);
      builder.create<TerminateOp>(ifOp.getLoc());
    });
  }

  void runOnOperation() override {
    for (auto f : llvm::to_vector(getOperation().getBody()->getOps<StepFuncOp>())) {
      outlineIfs(f);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createOutlineIfsPass() {
  return std::make_unique<OutlineIfsPass>();
}

} // namespace zirgen::Zhlt
