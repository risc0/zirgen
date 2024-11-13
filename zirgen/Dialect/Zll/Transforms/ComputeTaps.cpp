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

struct TapMap {
  TapMap(BuffersAttr bufs) : bufs(bufs) {}

  uint32_t getRegGroupId(BlockArgument ba) {
    if (regGroupIds.contains(ba))
      return regGroupIds.at(ba);

    auto func = llvm::cast<mlir::FunctionOpInterface>(ba.getOwner()->getParentOp());
    auto name = func.getArgAttrOfType<StringAttr>(ba.getArgNumber(), "zirgen.argName");
    if (!name) {
      llvm::errs() << "Cannot find argument name for arg " << ba << "\n";
      throw(std::runtime_error("taps computation failed"));
    }

    for (auto buf : bufs.getBuffers()) {
      if (buf.getName() == name) {
        if (!buf.getRegGroupId()) {
          llvm::errs() << "Referenced arg arg " << ba << " refers to a non-tap buffer\n";
        } else {
          size_t regGroupId = *buf.getRegGroupId();
          regGroupIds[ba] = regGroupId;
          return regGroupId;
        }
      }
    }
    llvm::errs() << "Cannot find reg group id for arg " << ba << "\n";
    throw(std::runtime_error("taps computation failed"));
  }

  // Map from block argument to register group id
  llvm::DenseMap<mlir::Value, size_t> regGroupIds;

  BuffersAttr bufs;
};

struct ComputeTapsPass : public ComputeTapsBase<ComputeTapsPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto loc = func->getLoc();
    Builder builder(&getContext());
    Type valType;

    auto mod = func->getParentOfType<mlir::ModuleOp>();

    BuffersAttr bufs = lookupModuleAttr<BuffersAttr>(mod);
    TapMap tapMap(bufs);

    // Find taps used in circuit.
    llvm::SmallVector<TapAttr> tapAttrs;
    mod.walk([&](GetOp op) {
      if (op.getBufferKind() != BufferKind::Global) {
        tapAttrs.push_back(
            TapAttr::get(&getContext(),
                         tapMap.getRegGroupId(llvm::cast<BlockArgument>(op.getBuf())),
                         op.getOffset(),
                         op.getBack()));
      }
    });

    // Make sure none of our three hardcoded buffers are empty.
    for (auto i : llvm::seq(3)) {
      tapAttrs.push_back(TapAttr::get(&getContext(), i, 0, 0));
    }

    setModuleAttr(mod, TapsAttr::sortAndPad(tapAttrs, bufs));

    auto& tapsAnalysis = getAnalysis<TapsAnalysis>();

    // Assign back to "tap" attributes on each operation
    mod.walk([&](GetOp op) {
      if (op.getBufferKind() != BufferKind::Global) {
        uint32_t tapId =
            tapsAnalysis.getTapIndex(tapMap.getRegGroupId(llvm::cast<BlockArgument>(op.getBuf())),
                                     op.getOffset(),
                                     op.getBack());

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
  }
};

} // End namespace

std::unique_ptr<OperationPass<func::FuncOp>> createComputeTapsPass() {
  return std::make_unique<ComputeTapsPass>();
}

} // namespace zirgen::Zll
