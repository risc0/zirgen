// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen::Zll {

namespace {

struct AddReductionsPass : public AddReductionsBase<AddReductionsPass> {
  void runOnOperation() override {
    // Get the function to run on + make a builder
    auto func = getOperation();
    Block* funcBlock = &func.front();
    DenseMap<Value, BigIntRange> ranges;
    OpBuilder builder(func);

    size_t totBits = 0;
    auto it = funcBlock->begin();
    auto term = funcBlock->end();
    term--;
    while (it != term) {
      Operation& origOp = *it;
      if (auto normOp = mlir::dyn_cast<NormalizeOp>(origOp)) {
        BigIntRange oldRange = ranges[normOp.getIn()];
        // llvm::errs() << "Hey, got a normalize, range = " << oldRange << "\n";
        if (oldRange.inRangeP()) {
          ++it;
          normOp.getOut().replaceAllUsesWith(normOp.getIn());
          normOp.erase();
          // llvm::errs() << "  GOODBYE!\n";
          continue;
        }
        normOp.setBits(oldRange.bits());
        normOp.setLow(oldRange.getLow().toStr());
        ranges[normOp.getOut()] = BigIntRange::rangeP();
        // llvm::errs() << "  Set output to rangeP!\n";
        totBits += oldRange.bits();
        ++it;
        continue;
      }
      auto op = mlir::dyn_cast<ReduceOp>(origOp);
      if (!op) {
        origOp.emitError("Invalid operation in AddReductionsPass");
        signalPassFailure();
        break;
      }
      // Compute op bounds (without reduction), this should always succeed
      // llvm::errs() << "Checking " << origOp << "\n";
      bool worked = op.updateRanges(ranges);
      assert(worked);
      if (origOp.getResults().size() == 1 && mlir::isa<ValType>(origOp.getResult(0).getType())) {
        Value result = origOp.getResult(0);
        BigIntRange oldRange = ranges[result];
        origOp.setAttr("bit_size", builder.getI32IntegerAttr(oldRange.bits()));
        // llvm::errs() << origOp << ", cur range = " << ranges[origOp.getResult(0)] << "\n";
        //  If this op is a ValType, check if all users are good with the range
        //  we computed, if not, we insert a reduction and carry on our way
        bool allWorked = true;
        for (auto& use : result.getUses()) {
          // llvm::errs() << "  Checking with user: " << *use.getOwner() << "\n";
          auto user = mlir::cast<ReduceOp>(use.getOwner());
          assert(user);
          if (!user.updateRanges(ranges)) {
            // llvm::errs() << "  Doing a reduce due to " << *use.getOwner() << "\n";
            allWorked = false;
            break;
          }
        }
        if (!allWorked) {
          builder.setInsertionPointAfter(&origOp);
          auto normOp = builder.create<NormalizeOp>(origOp.getLoc(), result, 0, "");
          result.replaceAllUsesExcept(normOp.getOut(), normOp);
        }
      }
      /*
      if (origOp.getResults().size() == 1 && mlir::isa<ValType>(origOp.getResult(0).getType())) {
        llvm::errs() << origOp << ":" << ranges[origOp.getResult(0)] << "\n";
      }
      */
      ++it;
    }
    llvm::errs() << "totBits = " << totBits << "\n";
  }
};

} // End namespace

std::unique_ptr<OperationPass<func::FuncOp>> createAddReductionsPass() {
  return std::make_unique<AddReductionsPass>();
}

} // namespace zirgen::Zll
