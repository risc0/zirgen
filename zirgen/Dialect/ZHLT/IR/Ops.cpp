// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
