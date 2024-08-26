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

#include <set>
#include <tuple>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/Analysis/TapsAnalysis.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/PassDetail.h"

using namespace mlir;

namespace zirgen::Zll {

namespace {

// TODO: OMG, this is so terrible
// Basically, the taps must be in order of 'accum', 'code', 'data'
// So we remap the key to be the right order
uint32_t getRegGroupId(BlockArgument ba) {
  constexpr uint32_t kInvalidRemap = std::numeric_limits<uint32_t>::max();
  static uint32_t remap[] = {1, kInvalidRemap, 2, kInvalidRemap, 0};
  assert(ba.getArgNumber() < std::size(remap));
  auto remapped = remap[ba.getArgNumber()];
  assert(remapped != kInvalidRemap);
  return remapped;
}

struct ComputeTapsPass : public ComputeTapsBase<ComputeTapsPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto loc = func.getLoc();
    Builder builder(&getContext());
    Type valType;

    // Find taps used in circuit.
    llvm::SmallVector<TapAttr> tapAttrs;
    func.walk([&](GetOp op) {
      if (op.getBufferKind() != BufferKind::Global) {
        tapAttrs.push_back(TapAttr::get(&getContext(),
                                        getRegGroupId(llvm::cast<BlockArgument>(op.getBuf())),
                                        op.getOffset(),
                                        op.getBack()));
      }
    });

    TapsAnalysis tapsAnalysis(&getContext(), std::move(tapAttrs));

    // Assign back to "tap" attributes on each operation
    func.walk([&](GetOp op) {
      if (op.getBufferKind() != BufferKind::Global) {
        uint32_t tapId = tapsAnalysis.getTapIndex(
            getRegGroupId(llvm::cast<BlockArgument>(op.getBuf())), op.getOffset(), op.getBack());

        op->setAttr("tap", builder.getUI32IntegerAttr(tapId));

        auto thisValType = op.getResult().getType();
        if (valType) {
          if (valType != thisValType) {
            emitError(op.getLoc()) << "All val types must be the same; found both " << valType
                                   << " and " << thisValType;
          }
        } else {
          valType = thisValType;
        }
      }
    });

    if (valType) {
      func->setAttr("tapType", TypeAttr::get(valType));
    } else {
      emitError(loc) << "No taps found";
    }

    func->setAttr("taps",
                  builder.getArrayAttr(llvm::to_vector_of<Attribute>(tapsAnalysis.getTapAttrs())));
    func->setAttr(
        "tapRegs",
        builder.getArrayAttr(llvm::to_vector_of<Attribute>(tapsAnalysis.getTapRegAttrs())));
    func->setAttr(
        "tapCombos",
        builder.getArrayAttr(llvm::to_vector_of<Attribute>(tapsAnalysis.getTapCombosAttrs())));
  }
};

} // End namespace

std::unique_ptr<OperationPass<func::FuncOp>> createComputeTapsPass() {
  return std::make_unique<ComputeTapsPass>();
}

} // namespace zirgen::Zll
