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
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen::Zll {

namespace {

struct PolyRewriter {
  OpBuilder& builder;
  IRMapping mapper;
  bool invalid;

  PolyRewriter(OpBuilder& builder) : builder(builder), invalid(false) {}

  Value runOnBlock(Location loc, Block& block, Value state) {
    for (Operation& origOp : block.without_terminator()) {
      TypeSwitch<Operation*>(&origOp)
          .Case<NondetOp>([&](auto) {})
          .Case<SetOp>([&](SetOp op) {
            auto newRHS = mapper.lookup(op.getIn());
            Value lhs =
                builder.create<GetOp>(op.getLoc(), op.getBuf(), op.getOffset(), 0, IntegerAttr());
            auto diff = builder.create<SubOp>(op.getLoc(), lhs, newRHS);
            state = builder.create<AndEqzOp>(op.getLoc(), state, diff);
          })
          .Case<SetGlobalOp>([&](SetGlobalOp op) {
            auto newRHS = mapper.lookup(op.getIn());
            Value lhs = builder.create<GetGlobalOp>(op.getLoc(), op.getBuf(), op.getOffset());
            auto diff = builder.create<SubOp>(op.getLoc(), lhs, newRHS);
            state = builder.create<AndEqzOp>(op.getLoc(), state, diff);
          })
          .Case<EqualZeroOp>([&](EqualZeroOp op) {
            auto newIn = mapper.lookup(op.getIn());
            state = builder.create<AndEqzOp>(op.getLoc(), state, newIn);
          })
          .Case<BarrierOp>([&](BarrierOp op) {})
          .Case<IfOp>([&](IfOp op) {
            auto newCond = mapper.lookup(op.getCond());
            Value innerState = builder.create<TrueOp>(loc);
            auto inner = runOnBlock(op.getLoc(), op.getInner().front(), innerState);
            state = builder.create<AndCondOp>(op.getLoc(), state, newCond, inner);
          })
          .Case<PolyOp>([&](PolyOp op) { builder.clone(origOp, mapper); })
          .Case<ExternOp>([&](ExternOp op) {
            if (op.getName() != "log") {
              op->emitError("Invalid extern op for MakePolynomial");
              invalid = true;
            }
          })
          .Default([&](Operation* op) {
            if (!llvm::isa<ZllDialect>(op->getDialect())) {
              // Just skip any non-zll operations.
              return;
            }
            llvm::errs() << *op;
            op->emitError("Invalid op for MakePolynomial");
            invalid = true;
          });
    }
    return state;
  }
};

struct MakePolynomialPass : public MakePolynomialBase<MakePolynomialPass> {
  void runOnOperation() override {
    // Get the function to run on + make a builder
    auto func = getOperation();
    auto loc = func->getLoc();
    Block* funcBlock = &func.getFunctionBody().front();
    Block* newBlock = new Block;
    func.getFunctionBody().push_back(newBlock);
    auto builder = OpBuilder::atBlockBegin(newBlock);

    // Change the function to output the final constraint
    func.setFunctionTypeAttr(TypeAttr::get(
        builder.getFunctionType(func.getArgumentTypes(), {builder.getType<ConstraintType>()})));

    // Make the rewriter + run on the entry block
    PolyRewriter rewriter(builder);
    Value state = builder.create<TrueOp>(loc);
    state = rewriter.runOnBlock(loc, *funcBlock, state);

    // Fail if it's bad
    if (rewriter.invalid) {
      signalPassFailure();
      return;
    }

    // Return result
    builder.create<func::ReturnOp>(loc, state);

    // Replace block with new block
    funcBlock->clear();
    funcBlock->getOperations().splice(
        funcBlock->begin(), newBlock->getOperations(), newBlock->begin(), newBlock->end());
    newBlock->erase();
  }
};

} // End namespace

std::unique_ptr<OperationPass<func::FuncOp>> createMakePolynomialPass() {
  return std::make_unique<MakePolynomialPass>();
}

} // namespace zirgen::Zll
