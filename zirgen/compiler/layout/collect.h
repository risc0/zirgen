// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/IR/Types.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "llvm/ADT/DenseMap.h"

namespace zirgen {
namespace layout {

using ZStruct::FieldInfo;
using ZStruct::LayoutArrayType;
using ZStruct::LayoutType;
using ZStruct::RefType;

struct Layout {
  Layout() = default;
  explicit Layout(LayoutType);
  LayoutType original;
  std::string id;
  bool isMux = false;
  std::vector<FieldInfo> fields;
};

struct Circuit {
  explicit Circuit(mlir::ModuleOp mod);
  llvm::DenseMap<LayoutType, Layout> structs;
  llvm::DenseMap<LayoutType, Layout> unions;
  std::vector<LayoutType> unionsInDfsPostorder;
  llvm::DenseMap<mlir::Type, unsigned> sizes;

private:
  unsigned visit(mlir::Type t);
  unsigned visit(RefType);
  unsigned visit(LayoutType);
  unsigned visitStruct(LayoutType);
  unsigned visitUnion(LayoutType);
  unsigned visit(LayoutArrayType);
};

} // namespace layout
} // namespace zirgen
