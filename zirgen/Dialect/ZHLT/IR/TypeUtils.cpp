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

#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace zirgen::Zhlt {

using namespace mlir;
using namespace Zll;
using namespace ZStruct;

bool isLegalTypeArg(Attribute attr) {
  // Would this attribute be a legal argument for a mangled component name?
  return attr && (isa<StringAttr>(attr) || isa<PolynomialAttr>(attr));
}

std::string mangledTypeName(StringRef componentName, llvm::ArrayRef<Attribute> typeArgs) {
  std::string out;
  llvm::raw_string_ostream stream(out);
  stream << componentName;
  if (typeArgs.size() != 0) {
    stream << "<";
    llvm::interleaveComma(typeArgs, stream, [&](Attribute typeArg) {
      if (auto strAttr = dyn_cast<StringAttr>(typeArg)) {
        stream << strAttr.getValue();
      } else if (auto intAttr = dyn_cast<PolynomialAttr>(typeArg)) {
        stream << intAttr[0];
      } else if (auto intAttr = dyn_cast<IntegerAttr>(typeArg)) {
        stream << intAttr.getUInt();
      } else {
        llvm::errs() << "Mangling type " << typeArg << " not implemented\n";
        assert(false && "not implemented");
      }
    });
    stream << ">";
  }
  return out;
}

std::string mangledArrayTypeName(Type element, unsigned size) {
  MLIRContext* ctx = element.getContext();
  SmallVector<Attribute> typeArgs;
  typeArgs.push_back(StringAttr::get(ctx, mangledTypeName(element)));
  typeArgs.push_back(Zll::getUI64Attr(ctx, size));
  return mangledTypeName("Array", typeArgs);
}

std::string mangledTypeName(Type type) {
  if (auto array = dyn_cast<ZStruct::ArrayType>(type)) {
    return mangledArrayTypeName(array.getElement(), array.getSize());
  } else if (auto array = dyn_cast<ZStruct::LayoutArrayType>(type)) {
    return mangledArrayTypeName(array.getElement(), array.getSize());
  } else if (auto component = dyn_cast<ZStruct::StructType>(type)) {
    return component.getId().str();
  } else if (auto component = dyn_cast<ZStruct::LayoutType>(type)) {
    return component.getId().str();
  } else if (auto component = dyn_cast<ZStruct::RefType>(type)) {
    return "Ref";
  } else if (auto val = dyn_cast<Zll::ValType>(type)) {
    return "Val";
  } else {
    assert(false && "not implemented");
    return "";
  }
}

Type getLeastCommonSuper(TypeRange components, bool isLayout) {
  // For types without layout, there is no common super layout
  if (isLayout && components.empty())
    return Type();

  assert(components.size() > 0);
  MLIRContext* ctx = components[0].getContext();
  Type commonSuper = components[0];

  for (Type type : components) {
    // check if type is a subtype of commonSuper
    llvm::SmallSet<Type, 8, TypeCmp> nextElemSuperChain;
    while (type && type != commonSuper) {
      nextElemSuperChain.insert(type);
      type = getSuperType(type, isLayout);
    }
    if (type == commonSuper) {
      continue;
    }

    // walk the old commonSuper's super chain until we find something in type's
    // super chain
    while (commonSuper && !nextElemSuperChain.contains(commonSuper)) {
      commonSuper = getSuperType(commonSuper, isLayout);
    }
  }

  if (commonSuper) {
    return commonSuper;
  } else {
    // if there is no common super, default to Component. Component has no layout type.
    if (isLayout)
      return Type();
    else
      return Zhlt::getComponentType(ctx);
  }
}

bool isCoercibleTo(Type src, Type dst, bool isLayout) {
  if (auto variadic = dyn_cast<VariadicType>(dst)) {
    dst = variadic.getElement();
  }
  while (src != dst) {
    Type newSrc = getSuperType(src, isLayout);
    if (!newSrc)
      break;
    src = newSrc;
  }
  if (src == dst)
    return true;

  if (auto srcArrayType = llvm::dyn_cast<ArrayLikeTypeInterface>(src)) {
    // Attempt to coerce arrays
    while (dst) {
      Type newDst = getSuperType(dst, isLayout);
      if (!newDst)
        break;
      dst = newDst;
    }

    auto dstArrayType = llvm::cast<ArrayLikeTypeInterface>(dst);
    if (srcArrayType.getSize() != dstArrayType.getSize())
      return false;

    if (llvm::isa<LayoutArrayType>(srcArrayType) || llvm::isa<ArrayType>(dstArrayType)) {
      // Cannot construct LayoutArrays at runtime.
      return isCoercibleTo(srcArrayType.getElement(), dstArrayType.getElement(), isLayout);
    }
  }

  return false;
}

template <typename StructLike>
Value coerceStructToSuper(TypedValue<StructLike> value, OpBuilder& builder) {
  // A previous error may have resulted in a malformed struct which lacks its
  // expected `@super` member; if we create a lookup, LLVM will assert because
  // it cannot infer the return type.
  auto super =
      find_if(value.getType().getFields(), [](auto field) { return field.name == "@super"; });

  if (super != value.getType().getFields().end()) {
    return builder.create<ZStruct::LookupOp>(value.getLoc(), value, super->name);
  } else {
    return {};
  }
}

Value coerceArrayTo(TypedValue<ArrayLikeTypeInterface> value,
                    ArrayLikeTypeInterface goalType,
                    OpBuilder& builder) {
  Location loc = value.getLoc();
  assert(value.getType().getSize() == goalType.getSize());

  auto map = builder.create<ZStruct::MapOp>(loc, goalType, value, Value());
  Block& block = map.getBody().emplaceBlock();
  OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPointToEnd(&block);
  Value originalElement = block.addArgument(value.getType().getElement(), loc);
  Value castedElement = coerceTo(originalElement, goalType.getElement(), builder);
  builder.create<ZStruct::YieldOp>(loc, castedElement);
  return map;
}

Value coerceTo(Value value, Type type, OpBuilder& builder) {
  MLIRContext* ctx = value.getContext();
  StructType componentType = Zhlt::getComponentType(ctx);
  Value casted = value;
  Location loc = value.getLoc();
  Type originalType = value.getType();
  if (!type) {
    // Previous errors may have left us with a null target type.
    emitError(value.getLoc()) << "cannot cast to invalid type";
    return value;
  }
  if (auto variadic = dyn_cast<VariadicType>(type)) {
    // A value trivially coerces to a singleton pack of its type
    type = variadic.getElement();
  }
  while (casted.getType() != type && casted.getType() != componentType) {
    if (isa<StructType>(casted.getType())) {
      auto castedStruct = cast<TypedValue<StructType>>(casted);
      casted = coerceStructToSuper<StructType>(castedStruct, builder);
      if (!casted) {
        auto loc = value.getLoc();
        emitError(loc) << "component struct must inherit from `Component`";
        return builder.create<Zhlt::MagicOp>(loc, type).getOut();
      }
    } else if (isa<LayoutType>(casted.getType())) {
      auto castedStruct = cast<TypedValue<LayoutType>>(casted);
      casted = coerceStructToSuper<LayoutType>(castedStruct, builder);
      if (!casted) {
        auto loc = value.getLoc();
        emitError(loc) << "type `" << getTypeId(originalType) << "` does not own a super layout of "
                       << "type `" << getTypeId(type) << "`";
        return builder.create<Zhlt::MagicOp>(loc, type).getOut();
      }
    } else if (UnionType ut = dyn_cast<UnionType>(casted.getType())) {
      // We require the super component to always be aligned to the beginning of
      // the component in its ultimate value representation. Because of this,the
      // common super must also occur at the beginning of each mux arm, and so
      // it makes no difference which arm we use to access the common super.
      // Without loss of generality, choose the first one.
      casted = builder.create<ZStruct::LookupOp>(value.getLoc(), casted, ut.getFields()[0].name);
    } else if (isa<ArrayLikeTypeInterface>(casted.getType()) && type != componentType) {
      if (auto goalArrayType = dyn_cast<ArrayLikeTypeInterface>(type)) {
        auto castedArray = dyn_cast<TypedValue<ArrayLikeTypeInterface>>(casted);
        casted = coerceArrayTo(castedArray, goalArrayType, builder);
      } else {
        auto loc = value.getLoc();
        emitError(loc) << "invalid coercion from array to non-array type";
        return builder.create<Zhlt::MagicOp>(loc, type).getOut();
      }
    } else {
      // If not a structural type, supertype is Component.
      casted =
          builder.create<ZStruct::PackOp>(loc, Zhlt::getComponentType(ctx), SmallVector<Value>());
    }
  }
  assert(casted.getType() == type && "attempted invalid coercion");
  return casted;
}

ArrayLikeTypeInterface getCoercibleArrayType(Type type) {
  while (type) {
    if (auto arrayType = dyn_cast<ArrayLikeTypeInterface>(type)) {
      return arrayType;
    }
    type = getSuperType(type);
  }
  return ArrayLikeTypeInterface();
}

Value coerceToArray(Value value, OpBuilder& builder) {
  MLIRContext* ctx = value.getContext();
  StructType componentType = Zhlt::getComponentType(ctx);
  Value casted = value;
  Location loc = value.getLoc();
  while (casted.getType() != componentType) {
    if (isa<StructType>(casted.getType())) {
      auto castedStruct = cast<TypedValue<StructType>>(casted);
      casted = coerceStructToSuper<StructType>(castedStruct, builder);
      if (!casted) {
        auto loc = value.getLoc();
        emitError(loc) << "component struct must inherit from `Component`";
        return castedStruct;
      }
    } else if (isa<LayoutType>(casted.getType())) {
      auto castedStruct = cast<TypedValue<LayoutType>>(casted);
      casted = coerceStructToSuper<LayoutType>(castedStruct, builder);
      if (!casted) {
        auto loc = value.getLoc();
        emitError(loc) << "component struct must inherit from `Component`";
        return castedStruct;
      }
    } else if (UnionType ut = dyn_cast<UnionType>(casted.getType())) {
      // We require the super component to always be aligned to the beginning of
      // the component in its ultimate value representation. Because of this,the
      // common super must also occur at the beginning of each mux arm, and so
      // it makes no difference which arm we use to access the common super.
      // Without loss of generality, choose the first one.
      casted = builder.create<ZStruct::LookupOp>(value.getLoc(), casted, ut.getFields()[0].name);
    } else if (isa<ArrayLikeTypeInterface>(casted.getType())) {
      return casted;
    } else {
      // If not a structural type, supertype is Component.
      casted =
          builder.create<ZStruct::PackOp>(loc, Zhlt::getComponentType(ctx), SmallVector<Value>());
    }
  }
  return casted;
}

bool isGenericBuiltin(StringRef name) {
  if (name == "Array") {
    return true;
  } else {
    return false;
  }
}

Value StructBuilder::getValue(Location loc) {
  return builder.create<PackOp>(loc, getType(), memberValues);
}

LayoutBuilder::LayoutBuilder(OpBuilder& builder, StringAttr typeName)
    : builder(builder), typeName(typeName) {
  layoutPlaceholder = builder.create<Zhlt::MagicOp>(builder.getUnknownLoc(), builder.getNoneType());
}

mlir::Value LayoutBuilder::addMember(Location loc, StringRef memberName, mlir::Type type) {
  assert(layoutPlaceholder);
  // If there is no layout type, return no layout value
  if (!type)
    return Value();

  for (auto member : members) {
    if (member.name == memberName) {
      llvm::errs() << "Duplicate name: " << memberName << "\n";
    }
    assert(member.name != memberName && "adding layout member with duplicate name");
  }
  StringAttr memberNameAttr = builder.getStringAttr(memberName);

  if (memberName == "@super") {
    // TODO: Why do we require @super to be first?  Document this requirement.
    members.insert(members.begin(), {memberNameAttr, type});
  } else {
    members.push_back({memberNameAttr, type});
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(layoutPlaceholder);
  auto lookupOp = builder.create<LookupOp>(loc, type, layoutPlaceholder, memberName);
  return lookupOp;
}

LayoutType LayoutBuilder::getType() {
  if (empty()) {
    return LayoutType();
  } else {
    return builder.getType<LayoutType>(typeName, members, kind);
  }
}

void LayoutBuilder::supplyLayout(std::function<Value(Type)> finalizeLayoutFunc) {
  assert(layoutPlaceholder);

  auto layoutType = getType();

  if (layoutType) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(layoutPlaceholder);
    Value layout = finalizeLayoutFunc(layoutType);
    layoutPlaceholder.getOut().replaceAllUsesWith(layout);
  } else {
    assert(layoutPlaceholder.use_empty());
  }

  layoutPlaceholder.erase();
  layoutPlaceholder = {};
}

std::string getTypeId(Type ty) {
  return TypeSwitch<Type, std::string>(ty)
      .Case<StringType>([](auto) { return "String"; })
      .Case<ValType>([](auto valType) { return (valType.getFieldK() == 1) ? "Val" : "ExtVal"; })
      .Case<RefType>(
          [](auto refType) { return (refType.getElement().getFieldK() == 1) ? "Ref" : "ExtRef"; })
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

void extractArguments(llvm::MapVector<Type, size_t>& out, Type in) {
  if (auto array = dyn_cast<LayoutArrayType>(in)) {
    llvm::MapVector<Type, size_t> inner;
    extractArguments(inner, array.getElement());
    for (auto& kvp : inner) {
      out[kvp.first] += kvp.second * array.getSize();
    }
  }
  if (auto layout = dyn_cast<LayoutType>(in)) {
    switch (layout.getKind()) {
    case LayoutKind::Argument:
      out[layout]++;
      return;
    case LayoutKind::Normal:
      for (auto& field : layout.getFields())
        extractArguments(out, field.type);
      break;
    case LayoutKind::MajorMux:
    case LayoutKind::Mux:
      // This is already handled by the @arguments$name member added to the
      // parent during lowering to ZHLT
      break;
    }
  }
}

llvm::MapVector<Type, size_t> muxArgumentCounts(TypeRange in) {
  llvm::MapVector<Type, size_t> worstCase;
  for (auto type : in) {
    llvm::MapVector<Type, size_t> curCase;
    Zhlt::extractArguments(curCase, cast<LayoutType>(type));
    for (const auto& kvp : curCase) {
      worstCase[kvp.first] = std::max(worstCase[kvp.first], kvp.second);
    }
  }
  return worstCase;
}

llvm::MapVector<Type, size_t> muxArgumentCounts(LayoutType in) {
  assert(in.getKind() == LayoutKind::Mux);
  SmallVector<Type> layoutFields =
      llvm::to_vector(llvm::map_range(in.getFields(), [](auto field) { return field.type; }));
  return Zhlt::muxArgumentCounts(layoutFields);
}

} // namespace zirgen::Zhlt
