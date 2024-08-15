// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/IOP/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

mlir::ParseResult parseArrType(mlir::OpAsmParser& parser, llvm::SmallVectorImpl<mlir::Type>& out);
void printArrType(mlir::OpAsmPrinter& p, mlir::Operation* op, mlir::TypeRange types);

#include "zirgen/Dialect/IOP/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/IOP/IR/Types.cpp.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/IOP/IR/Ops.cpp.inc"

namespace zirgen::Iop {

void IopDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zirgen/Dialect/IOP/IR/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "zirgen/Dialect/IOP/IR/Types.cpp.inc"
      >();
}

} // namespace zirgen::Iop
