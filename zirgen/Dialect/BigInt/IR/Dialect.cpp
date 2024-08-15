// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/BigInt/IR/BigInt.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/BigInt/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/BigInt/IR/Types.cpp.inc"

using namespace mlir;

namespace zirgen::BigInt {

void BigIntDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zirgen/Dialect/BigInt/IR/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "zirgen/Dialect/BigInt/IR/Types.cpp.inc"
      >();
}

codegen::CodegenIdent<codegen::IdentKind::Type>
BigIntType::getTypeName(codegen::CodegenEmitter& cg) const {
  return cg.getStringAttr("byte_poly");
}

bool BigIntType::allowDuplicateTypeNames() const {
  return true;
}

namespace {

StringRef getIterationCountAttrName() {
  return "bigint.iteration_count";
}

} // namespace

size_t getIterationCount(func::FuncOp func) {
  if (auto attr = func->getAttrOfType<IntegerAttr>(getIterationCountAttrName())) {
    return attr.getUInt();
  }
  return 1;
}

void setIterationCount(func::FuncOp func, size_t iters) {
  assert(iters > 0);
  func->setAttr(getIterationCountAttrName(),
                IntegerAttr::get(func.getContext(), APSInt::getUnsigned(iters)));
}

} // namespace zirgen::BigInt
