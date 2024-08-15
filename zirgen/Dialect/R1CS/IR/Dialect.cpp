// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/R1CS/IR/R1CS.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/R1CS/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/R1CS/IR/Types.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "zirgen/Dialect/R1CS/IR/Attrs.cpp.inc"

// using namespace mlir;

namespace zirgen::R1CS {

void R1CSDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zirgen/Dialect/R1CS/IR/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "zirgen/Dialect/R1CS/IR/Types.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "zirgen/Dialect/R1CS/IR/Attrs.cpp.inc"
      >();
}

} // namespace zirgen::R1CS
