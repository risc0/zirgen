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
