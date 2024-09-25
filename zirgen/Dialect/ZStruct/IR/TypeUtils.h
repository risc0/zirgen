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

// Returns the component ID of this component.
std::string getTypeId(Type type);

// Returns the type of this structure's "@super", if any.
Type getSuperType(Type subType, bool isLayout = false);

// Returns the layout type corresponding to the given value type
Type getLayoutType(Type valueType);

} // namespace zirgen::ZStruct
