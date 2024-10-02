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

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "zirgen/Dialect/ZStruct/IR/Types.h"
#include "zirgen/Dialect/Zll/IR/IR.h"

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/ZStruct/IR/Types.h.inc"
#define GET_ATTRDEF_CLASSES
#include "zirgen/Dialect/ZStruct/IR/Attrs.h.inc"

#include "zirgen/Dialect/ZStruct/IR/Attrs.h"

#include "mlir/IR/Dialect.h"

#include "zirgen/Dialect/ZStruct/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ZStruct/IR/Ops.h.inc"

namespace zirgen::ZStruct {

inline size_t getIndexVal(mlir::IntegerAttr attr) {
  return attr.getValue().getZExtValue();
}

// Constant names for generated constants.
std::string getLayoutConstName(llvm::StringRef origName);

// Extract an integer constant from the given attribute by whatever means necessary.
//
// TODO: get attribute types straightened out so we don't have to use this.

int extractIntAttr(mlir::Attribute attr);

// Add ZStruct-specific generated code syntax
void addCppSyntax(codegen::CodegenOptions& opts);
void addRustSyntax(codegen::CodegenOptions& opts);

} // namespace zirgen::ZStruct
