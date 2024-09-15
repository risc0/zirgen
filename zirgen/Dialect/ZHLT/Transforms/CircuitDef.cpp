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

#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"
#include "zirgen/Dialect/ZStruct/Analysis/BufferAnalysis.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"

#include <set>
#include <vector>

using namespace mlir;

namespace zirgen::Zhlt {

namespace {

#define GEN_PASS_DEF_CIRCUITDEF
#include "zirgen/Dialect/ZHLT/Transforms/Passes.h.inc"

// Build an EDSL-style FuncOp based on a StepFuncOp by analyzing what
// buffers it needs and adding those as arguments.
//
// TODO:
//   Right now we use the `zirgen.argName` argument attribute to associate
//   arguments with buffers.  It might be more ergonomic to have a
//   Zll::FuncOp that has this association builtin, and then we can
//   verify that type types match the types in the CircuitDefOp.
void stepFuncToFunc(mlir::OpBuilder& builder, StepFuncOp stepFuncOp, llvm::StringRef newName) {
  mlir::OpBuilder::InsertionGuard guard(builder);

  llvm::SmallVector<Type> argTypes;

  stepFuncOp.walk([&](ZStruct::GetBufferOp bufferOp) { argTypes.push_back(bufferOp.getType()); });

  auto funcType = builder.getFunctionType(argTypes, {builder.getType<Zll::ValType>()});
  auto funcOp = builder.create<mlir::func::FuncOp>(stepFuncOp.getLoc(), newName, funcType);

  auto* block = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(block);

  size_t argNum = 0;

  IRMapping mapper;
  for (auto& op : stepFuncOp.getBody().front()) {
    TypeSwitch<Operation*>(&op)
        .Case<ZStruct::GetBufferOp>([&](auto bufOp) {
          mapper.map(bufOp, block->getArgument(argNum));
          funcOp.setArgAttr(argNum, "zirgen.argName", builder.getStringAttr(bufOp.getName()));
          ++argNum;
        })
        .Case<Zhlt::ReturnOp>([&](auto returnOp) {
          auto zero = builder.create<Zll::ConstOp>(returnOp.getLoc(), 0);
          builder.create<func::ReturnOp>(returnOp.getLoc(), ValueRange{zero});
        })
        .Default([&](auto op) { builder.clone(*op, mapper); });
  }
}

struct CircuitDefPass : public impl::CircuitDefBase<CircuitDefPass> {
  using CircuitDefBase<CircuitDefPass>::CircuitDefBase;

  void runOnOperation() override {
    OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());

    SmallVector<Attribute> buffers;
    for (auto buf : getAnalysis<ZStruct::BufferAnalysis>().getAllBuffers()) {
      buffers.push_back(builder.getAttr<Zll::BufferDefAttr>(
          buf.name, buf.kind, buf.regCount, buf.getType(builder.getContext()), buf.regGroupId));
    }

    SmallVector<Attribute> steps;

    getOperation()->walk([&](StepFuncOp funcOp) {
      StringRef newName = llvm::StringSwitch<StringRef>(funcOp.getName())
                              .Case("step$Top$accum", "compute_accum")
                              .Case("step$Top", "exec");

      if (newName.empty()) {
        funcOp->emitOpError("unknown step function");
        signalPassFailure();
        return;
      }

      stepFuncToFunc(builder, funcOp, newName);
      funcOp.erase();

      steps.push_back(builder.getAttr<Zll::StepDefAttr>(
          newName, SymbolRefAttr::get(builder.getStringAttr(newName))));
    });

    builder.create<Zll::CircuitDefOp>(builder.getUnknownLoc(),
                                      circuitInfo,
                                      builder.getArrayAttr(buffers),
                                      builder.getArrayAttr(steps));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createCircuitDefPass(const CircuitDefOptions& opts) {
  return std::make_unique<CircuitDefPass>(opts);
}

} // namespace zirgen::Zhlt
