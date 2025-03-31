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

#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ZHLT/IR/Ops.cpp.inc"

namespace zirgen::Zhlt {

using namespace mlir;

SmallVector<int32_t> ComponentOp::getInputSegmentSizes() {
  SmallVector<int32_t> segmentSizes(2);
  ArrayRef<Type> inputTypes = getFunctionType().getInputs();
  Type lastType = inputTypes.empty() ? nullptr : inputTypes.back();
  if (lastType && ZStruct::isLayoutType(lastType)) {
    segmentSizes[0] = inputTypes.size() - 1;
    segmentSizes[1] = 1;
  } else {
    segmentSizes[0] = inputTypes.size();
    segmentSizes[1] = 0;
  }
  return segmentSizes;
}

SmallVector<int32_t> ConstructOp::getInputSegmentSizes() {
  auto sizes = (*this)->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
  return SmallVector<int32_t>(sizes.asArrayRef());
}

SmallVector<int32_t> CheckLayoutFuncOp::getInputSegmentSizes() {
  // TODO: this is trivial
  SmallVector<int32_t> segmentSizes(0);
  segmentSizes.push_back(getFunctionType().getNumInputs());
  return segmentSizes;
}

SmallVector<int32_t> CheckFuncOp::getInputSegmentSizes() {
  // TODO: this is trivial
  return {};
}

static SmallVector<int32_t> getExecSegmentSizes(FunctionType funcType) {
  SmallVector<int32_t> segmentSizes(2);
  ArrayRef<Type> inputTypes = funcType.getInputs();
  if (!inputTypes.empty() && ZStruct::isLayoutType(inputTypes.back())) {
    segmentSizes[0] = funcType.getNumInputs() - 1;
    segmentSizes[1] = 1;
  } else {
    segmentSizes[0] = funcType.getNumInputs();
    segmentSizes[1] = 0;
  }
  return segmentSizes;
}

SmallVector<int32_t> ExecFuncOp::getInputSegmentSizes() {
  return getExecSegmentSizes(getFunctionType());
}

SmallVector<int32_t> ExecCallOp::getInputSegmentSizes() {
  return getExecSegmentSizes(getCalleeType());
}

static SmallVector<int32_t> getBackSegmentSizes(FunctionType funcType) {
  // TODO: BackOp doesn't really need this interface!
  SmallVector<int32_t> segmentSizes(2);
  segmentSizes[0] = 1;
  if (funcType.getNumInputs() == 2) {
    segmentSizes[1] = 1;
  } else {
    segmentSizes[1] = 0;
  }
  return segmentSizes;
}

SmallVector<int32_t> BackFuncOp::getInputSegmentSizes() {
  return getBackSegmentSizes(getFunctionType());
}

SmallVector<int32_t> BackCallOp::getInputSegmentSizes() {
  return getBackSegmentSizes(getCalleeType());
}

mlir::LogicalResult MagicOp::verify() {
  return emitError() << "a MagicOp is never valid";
}

mlir::LogicalResult BackOp::verify() {
  if (getLayout()) {
    auto layoutType = getLayout().getType();
    if (layoutType && !ZStruct::isLayoutType(layoutType)) {
      return emitError() << layoutType << " must be a layout type";
    }
  }

  auto outType = getType();
  if (!ZStruct::isRecordType(outType)) {
    return emitError() << outType << " must be a value type";
  }
  return success();
}

} // namespace zirgen::Zhlt
