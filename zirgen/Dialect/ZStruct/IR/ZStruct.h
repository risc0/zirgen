// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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

} // namespace zirgen::ZStruct
