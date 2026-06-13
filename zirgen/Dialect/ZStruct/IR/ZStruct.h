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

/// The ZStruct dialect models structured (struct/array/map) types layered on
/// top of the base Zll field-element types. It provides ops for constructing,
/// deconstructing, mapping over, and reducing arrays of structured values, as
/// well as a layout mechanism for associating circuit register addresses with
/// named fields.
namespace zirgen::ZStruct {

/// Extract the zero-extended integer value from an `IntegerAttr`.
inline size_t getIndexVal(mlir::IntegerAttr attr) {
  return attr.getValue().getZExtValue();
}

/// Return the name used for the GlobalConstOp that holds the layout constant
/// derived from the function named `origName`.
std::string getLayoutConstName(llvm::StringRef origName);

/// Extract an integer constant from `attr`, accepting multiple attribute
/// encodings (IntegerAttr, PolynomialAttr, etc.).
int extractIntAttr(mlir::Attribute attr);

/// Register ZStruct dialect-specific codegen lowerings for C++ emission.
void addCppSyntax(codegen::CodegenOptions& opts);
/// Register ZStruct dialect-specific codegen lowerings for Rust emission.
void addRustSyntax(codegen::CodegenOptions& opts);

} // namespace zirgen::ZStruct
