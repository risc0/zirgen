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
