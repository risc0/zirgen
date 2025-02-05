// Copyright 2025 RISC Zero, Inc.
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include "zirgen/Dialect/ByteCode/Analysis/ArmAnalysis.h"
#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"
#include "zirgen/Dialect/ByteCode/Transforms/Encode.h"
#include "zirgen/Dialect/ByteCode/Transforms/Passes.h"

#define DEBUG_TYPE "calc-bit-widths"

using namespace mlir;

namespace zirgen::ByteCode {

namespace {

void updateMax(SmallVectorImpl<size_t>& origVec, ArrayRef<size_t> newVec) {
  if (origVec.empty())
    origVec.resize(newVec.size());
  for (auto [origVal, newVal] : llvm::zip_equal(origVec, newVec)) {
    origVal = std::max<size_t>(origVal, newVal);
  }
}

struct BitCalculator {
  void analyzeOp(BitWidthOpInterface op, ArrayRef<size_t> maxVals) {
    for (size_t maxVal : maxVals) {
      size_t bitsNeeded = 0;
      while ((size_t(1) << bitsNeeded) <= maxVal) {
        bitsNeeded += 8;
      }
      bits.push_back(bitsNeeded);
    }
  }

  void updateOp(BitWidthOpInterface op) {
    size_t numEncoded = op.getNumEncodedElements();
    op.setEncodedElementsBits(ArrayRef(bits).slice(updatePos, numEncoded));
    updatePos += numEncoded;
  }

  void assertDone() {
    assert(updatePos == bits.size());
  }

  SmallVector<size_t> bits;
  size_t updatePos = 0;
};

} // namespace

#define GEN_PASS_DEF_CALCBITWIDTHS
#include "zirgen/Dialect/ByteCode/Transforms/Passes.h.inc"

struct CalcBitWidthsPass : public impl::CalcBitWidthsBase<CalcBitWidthsPass> {
  using CalcBitWidthsBase::CalcBitWidthsBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    auto execOp = mod.lookupSymbol<ExecutorOp>(execSymbol);
    if (!execOp) {
      mod.emitError() << "Unable to find ExecutorOp named " << execSymbol << "\n";
      signalPassFailure();
      return;
    }

    DenseMap<BitWidthOpInterface, SmallVector<size_t, 1>> maxValue;

    auto updateOpMax = [&](BitWidthOpInterface op, EncodedOp encodedOp, size_t& elem) {
      size_t numEncoded = op.getNumEncodedElements();
      SmallVector<size_t> elems;
      for (size_t i = 0; i < numEncoded; ++i) {
        elems.push_back(std::get<size_t>(encodedOp.getElement(elem + i)));
      }
      updateMax(maxValue[op], elems);
      elem += numEncoded;
    };

    mod.walk([&](EncodedOp encodedOp) {
      if (!encodedOp.getOperands().empty() || !encodedOp.getResults().empty()) {
        encodedOp.emitError() << "Must bufferize operation before calculating bit widths\n";
        signalPassFailure();
        return WalkResult::interrupt();
      }

      size_t elem = 0;
      updateOpMax(execOp, encodedOp, elem);
      assert(elem == 1 && "Expected only one value for the arm dispatch key");

      size_t armIdx = std::get<size_t>(encodedOp.getElement(0));
      auto& arm = execOp.getArms()[armIdx];

      for (BitWidthOpInterface bitWidthOp : arm.getOps<BitWidthOpInterface>()) {
        updateOpMax(bitWidthOp, encodedOp, elem);
      }
      assert(elem == encodedOp.size());

      return WalkResult::advance();
    });

    SmallVector<DenseI64ArrayAttr> encodedBits;
    SmallVector<size_t, 1> zeros;
    Builder builder(&getContext());
    for (auto& arm : execOp.getArms()) {
      BitCalculator bc;
      bc.analyzeOp(execOp, maxValue[execOp]);
      for (auto op : arm.getOps<BitWidthOpInterface>()) {
        if (maxValue.contains(op)) {
          bc.analyzeOp(op, maxValue[op]);
        } else {
          // Use all zeros if we found no references.
          // TODO: We should probably DCE this arm.
          zeros.resize(op.getNumEncodedElements());
          bc.analyzeOp(op, zeros);
        }
      }
      encodedBits.emplace_back(builder.getDenseI64ArrayAttr(llvm::to_vector_of<int64_t>(bc.bits)));
      bc.updateOp(execOp);
      for (auto op : arm.getOps<BitWidthOpInterface>()) {
        bc.updateOp(op);
      }
      bc.assertDone();
    }

    mod.walk([&](EncodedOp encodedOp) {
      size_t armIdx = std::get<size_t>(encodedOp.getElement(0));
//      encodedOp.setEncodedBitsAttr(encodedBits[armIdx]);
    });
  }
};

} // namespace zirgen::ByteCode
