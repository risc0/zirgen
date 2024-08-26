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
