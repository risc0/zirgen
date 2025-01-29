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

#pragma once

#include <variant>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "zirgen/Dialect/ByteCode/Interfaces/Interfaces.h"

namespace zirgen {

class Interpreter;

namespace ByteCode {

// Representation of a value in a partially encoded bytecode program.
using EncodedElement =
    std::variant</*intArg=*/size_t, /*operand=*/mlir::Value, /*result=*/mlir::OpResult>;

// Representation of a value in a partially encoded bytecode program used
// to build an EncodedOp.
using BuildEncodedElement =
    std::variant</*intArg=*/size_t, /*operand=*/mlir::Value, /*result type=*/mlir::Type>;

} // namespace ByteCode

} // namespace zirgen

#include "zirgen/Dialect/ByteCode/IR/Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/ByteCode/IR/Types.h.inc"

#define GET_ATTRDEF_CLASSES
#include "zirgen/Dialect/ByteCode/IR/Attrs.h.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ByteCode/IR/Ops.h.inc"

namespace zirgen::ByteCode {

std::string getNameForIntKind(mlir::Attribute intKind);

} // namespace zirgen::ByteCode
