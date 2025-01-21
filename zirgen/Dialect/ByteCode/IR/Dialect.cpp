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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ByteCode/IR/ByteCode.h"
#include "zirgen/Dialect/ByteCode/IR/Dialect.cpp.inc"

using namespace mlir;

#define GET_OP_CLASSES
#include "zirgen/Dialect/ByteCode/IR/Ops.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/ByteCode/IR//Types.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "zirgen/Dialect/ByteCode/IR//Attrs.cpp.inc"

namespace zirgen::ByteCode {

mlir::Attribute getDispatchKeyIntKind(mlir::MLIRContext* ctx) {
  return StringAttr::get(ctx, "DispatchKey");
}

void ByteCodeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zirgen/Dialect/ByteCode/IR/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "zirgen/Dialect/ByteCode/IR/Types.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "zirgen/Dialect/ByteCode/IR/Attrs.cpp.inc"
      >();
}

} // namespace zirgen::ByteCode
