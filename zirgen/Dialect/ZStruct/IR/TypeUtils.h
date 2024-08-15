// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"

namespace zirgen::ZStruct {

using namespace mlir;

ZStruct::StructType getTypeType(MLIRContext* ctx);
ZStruct::StructType getComponentType(MLIRContext* ctx);
ZStruct::LayoutType getEmptyLayoutType(MLIRContext* ctx);
Zll::ValType getValType(MLIRContext* ctx);
Zll::ValType getValExtType(MLIRContext* ctx);
Zll::StringType getStringType(MLIRContext* ctx);
ZStruct::StructType getNondetRegType(MLIRContext* ctx);
ZStruct::LayoutType getNondetRegLayoutType(MLIRContext* ctx);
ZStruct::RefType getRefType(MLIRContext* ctx);
ZStruct::RefType getExtRefType(MLIRContext* ctx);

Zll::ValType getFieldTypeOfValType(Type valType);

} // namespace zirgen::ZStruct
