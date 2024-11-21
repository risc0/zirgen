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

#include "zirgen/Dialect/ZStruct/IR/TypeUtils.h"

namespace zirgen::ZStruct {

using namespace mlir;
using namespace Zll;

StructType getTypeType(MLIRContext* ctx) {
  return StructType::get(ctx, "Type", {});
}

StructType getComponentType(MLIRContext* ctx) {
  return StructType::get(ctx, "Component", {});
}

LayoutType getEmptyLayoutType(MLIRContext* ctx) {
  return LayoutType::get(ctx, "Component", {});
}

ValType getValType(MLIRContext* ctx) {
  return ValType::getBaseType(ctx);
}

ValType getExtValType(MLIRContext* ctx) {
  return ValType::getExtensionType(ctx);
}

StringType getStringType(MLIRContext* ctx) {
  return StringType::get(ctx);
}

StructType getNondetRegType(MLIRContext* ctx) {
  SmallVector<ZStruct::FieldInfo> members;
  members.push_back({StringAttr::get(ctx, "@super"), getValType(ctx)});
  members.push_back({StringAttr::get(ctx, "@layout"), getNondetRegLayoutType(ctx)});
  return StructType::get(ctx, "NondetReg", members);
}

StructType getNondetExtRegType(MLIRContext* ctx) {
  SmallVector<ZStruct::FieldInfo> members;
  members.push_back({StringAttr::get(ctx, "@super"), getExtValType(ctx)});
  return StructType::get(ctx, "NondetExtReg", members);
}

LayoutType getNondetRegLayoutType(MLIRContext* ctx) {
  SmallVector<ZStruct::FieldInfo> members;
  members.push_back({StringAttr::get(ctx, "@super"), getRefType(ctx)});
  return LayoutType::get(ctx, "NondetReg", members);
}

LayoutType getNondetExtRegLayoutType(MLIRContext* ctx) {
  SmallVector<ZStruct::FieldInfo> members;
  members.push_back({StringAttr::get(ctx, "@super"), getExtRefType(ctx)});
  return LayoutType::get(ctx, "NondetExtReg", members);
}

RefType getRefType(MLIRContext* ctx) {
  return RefType::get(ctx, getValType(ctx));
}

RefType getExtRefType(MLIRContext* ctx) {
  return RefType::get(ctx, getExtValType(ctx));
}

} // namespace zirgen::ZStruct
