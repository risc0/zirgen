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

#include "llvm/ADT/DenseSet.h"

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/PassDetail.h"
#include "zirgen/Dialect/ZStruct/Analysis/BufferAnalysis.h"

#include <set>
#include <vector>

using namespace mlir;

namespace zirgen::Zhlt {

namespace {

#define GEN_PASS_DEF_CIRCUITDEF
#include "zirgen/Dialect/ZHLT/Transforms/Passes.h.inc"

struct CircuitDefPass : public impl::CircuitDefBase<CircuitDefPass> {
  using CircuitDefBase<CircuitDefPass>::CircuitDefBase;

  void runOnOperation() override {
    OpBuilder builder = OpBuilder::atBlockEnd(getOperation().getBody());

    SmallVector<Attribute> buffers;
    for (auto buf : getAnalysis<ZStruct::BufferAnalysis>().getAllBuffers()) {
      buffers.push_back(
          builder.getAttr<Zll::BufferDefAttr>(buf.name,
                                              buf.kind,
                                              buf.regCount,
                                              TypeAttr::get(buf.getType(builder.getContext())),
                                              buf.regGroupId));
    }

    SmallVector<Attribute> steps;
    getOperation()->walk([&](StepFuncOp funcOp) {
      StringRef newName = llvm::StringSwitch<StringRef>(funcOp.getName())
                              .Case("step$Top$accum", "compute_accum")
                              .Case("step$Top", "exec");

      if (!newName.empty()) {
        steps.push_back(
            builder.getAttr<Zll::StepDefAttr>(newName, SymbolRefAttr::get(funcOp.getNameAttr())));
      } else {
        funcOp->emitOpError("unknown step function");
        signalPassFailure();
      }
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
