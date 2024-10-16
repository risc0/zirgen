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
#include "llvm/ADT/TypeSwitch.h"

namespace zirgen::ZStruct {

using namespace mlir;
using namespace Zll;

StructType getTypeType(MLIRContext* ctx) {
  return StructType::get(ctx, "Type", {}, {});
}

StructType getComponentType(MLIRContext* ctx) {
  return StructType::get(ctx, "Component", {}, {});
}

LayoutType getEmptyLayoutType(MLIRContext* ctx) {
  return LayoutType::get(ctx, "Component", {}, {});
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
  auto layout = zirgen::ZStruct::getNondetRegLayoutType(ctx);
  return StructType::get(ctx, "NondetReg", members, layout);
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

std::string getTypeId(Type ty) {
  return TypeSwitch<Type, std::string>(ty)
      .Case<StringType>([](auto) { return "String"; })
      .Case<ValType>([](auto) { return "Val"; })
      .Case<RefType>([](auto) { return "Ref"; })
      .Case<StructType>([](auto structType) { return structType.getId(); })
      .Case<LayoutType>([](auto layoutType) { return layoutType.getId(); })
      .Case<ArrayLikeTypeInterface>([](auto arrayType) {
        return "Array<" + getTypeId(arrayType.getElement()) + ", " +
               std::to_string(arrayType.getSize()) + ">";
      })
      .Case<VariadicType>(
          [](auto packType) { return "Variadic<" + getTypeId(packType.getElement()) + ">"; })
      .Default([&](auto) -> std::string {
        llvm::errs() << "Type: " << ty << "\n";
        assert(0 && "Unexpected type for getTypeId");
      });
}

Type getSuperType(Type ty, bool isLayout) {
  Type componentType = isLayout ? Type() : getComponentType(ty.getContext());
  // Structs have a super of their @super element
  ArrayRef<FieldInfo> fields;
  if (auto zType = llvm::dyn_cast<LayoutType>(ty)) {
    fields = zType.getFields();
  } else if (auto zType = llvm::dyn_cast<StructType>(ty)) {
    fields = zType.getFields();
  }
  for (auto field : fields) {
    if (field.name == "@super") {
      return field.type;
    }
  }

  // Arrays are 'covariate'
  if (auto aType = llvm::dyn_cast<ArrayType>(ty)) {
    Type innerSuper = getSuperType(aType.getElement());
    if (!innerSuper) {
      return componentType;
    }
    return ArrayType::get(ty.getContext(), innerSuper, aType.getSize());
  }
  if (auto aType = llvm::dyn_cast<LayoutArrayType>(ty)) {
    Type innerSuper = getSuperType(aType.getElement());
    if (!innerSuper) {
      return componentType;
    }
    return LayoutArrayType::get(ty.getContext(), innerSuper, aType.getSize());
  }
  // All other type have componentType as a supertype
  if (ty != componentType) {
    if (isLayout) {
      // Layout use empty for componentType
      return {};
    }
    return componentType;
  }
  // Component has no super
  return {};
}

Type getLayoutType(Type valueType) {
  if (!valueType)
    return Type();

  return llvm::TypeSwitch<Type, Type>(valueType)
      .Case<StructType>([](auto t) -> Type {
        if (t.getLayout())
          return t.getLayout();
        return getLayoutType(getSuperType(t));
      })
      .Case<ArrayType>([](auto t) { return t.getLayoutArray(); })
      .Default([](auto t) { return Type(); });
}

} // namespace zirgen::ZStruct
