// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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

ValType getValExtType(MLIRContext* ctx) {
  return ValType::getExtensionType(ctx);
}

StringType getStringType(MLIRContext* ctx) {
  return StringType::get(ctx);
}

StructType getNondetRegType(MLIRContext* ctx) {
  SmallVector<ZStruct::FieldInfo> members;
  members.push_back({StringAttr::get(ctx, "@super"), getValType(ctx)});
  return StructType::get(ctx, "NondetReg", members);
}

LayoutType getNondetRegLayoutType(MLIRContext* ctx) {
  SmallVector<ZStruct::FieldInfo> members;
  members.push_back({StringAttr::get(ctx, "@super"), getRefType(ctx)});
  return LayoutType::get(ctx, "NondetReg", members);
}

RefType getRefType(MLIRContext* ctx) {
  return RefType::get(ctx, getValType(ctx));
}

RefType getExtRefType(MLIRContext* ctx) {
  return RefType::get(ctx, getValExtType(ctx));
}

} // namespace zirgen::ZStruct
