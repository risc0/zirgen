// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/ZHL/IR/ZHL.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHL/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/ZHL/IR/Types.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "zirgen/Dialect/ZHL/IR/Attrs.cpp.inc"

using namespace mlir;

namespace zirgen::Zhl {

void ZhlDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zirgen/Dialect/ZHL/IR/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "zirgen/Dialect/ZHL/IR/Types.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "zirgen/Dialect/ZHL/IR/Attrs.cpp.inc"
      >();
}

} // namespace zirgen::Zhl
