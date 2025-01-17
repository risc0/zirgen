// Copyright 2025 RISC Zero, Inc.
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

#include "zirgen/Conversions/Typing/ZhlComponent.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "zirgen/Conversions/Typing/BuiltinComponents.h"
#include "zirgen/Conversions/Typing/ComponentManager.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <set>
#include <variant>

#define DEBUG_TYPE "zhlt"

namespace zirgen::Typing {
namespace {

using namespace mlir;
using namespace zirgen::Zhl;
using Zhlt::getTypeId;
using Zhlt::LayoutBuilder;
using Zhlt::mangledTypeName;
using Zhlt::StructBuilder;
using Zll::StringType;
using Zll::ValType;
using Zll::VariadicType;
using ZStruct::ArrayType;
using ZStruct::LayoutArrayType;
using ZStruct::LayoutKind;
using ZStruct::LayoutType;
using ZStruct::StructType;
using ZStruct::UnionType;

// A builder for a component or subcomponent, managing both layouts
// and values.  It may have zero or more layout members and zero or
// more value members.
class ComponentBuilder {
public:
  // Starts building a top-level component with the given type name.
  ComponentBuilder(Location loc, OpBuilder& builder, StringAttr typeName)
      : loc(loc), builder(builder), typeName(typeName) {
    layoutBuilder = std::make_unique<LayoutBuilder>(builder, typeName);
    valBuilder = std::make_unique<StructBuilder>(builder, typeName);
  }

  // Start building a component inside the given parent component.  If
  // name is empty, generates an anonymous name.
  ComponentBuilder(Location loc, ComponentBuilder* parent, StringAttr name)
      : loc(loc), builder(parent->builder), parent(parent), memberNameInParent(name) {
    push();
  }

  ComponentBuilder(Location loc, ComponentBuilder* parent, StringRef name)
      : loc(loc), builder(parent->builder), parent(parent) {
    memberNameInParent = builder.getStringAttr(name);
    push();
  }

  // Builds a new subcomponent outputting to the given Zhl value.  If
  // the subcomponent is used for a member definition, that name is
  // used; otherwise an anonymous name is used.
  ComponentBuilder(Location loc, ComponentBuilder* parent, Value outValue)
      : loc(loc), builder(parent->builder), parent(parent) {
    memberNameInParent = parent->getResultName(outValue);
    push();
  }

  // Movable but not copyable.
  ComponentBuilder(ComponentBuilder&&) = default;

  LayoutBuilder* layout() { return layoutBuilder.get(); }
  StructBuilder* val() { return valBuilder.get(); }

  Value getValue(Location loc) {
    assert(val());
    Value resultVal = val()->getValue(loc);
    valBuilder.reset();
    return resultVal;
  }

  void supplyLayout(std::function<Value(/*name=*/StringAttr, Type)> finalizeLayoutFn) {
    assert(layout());
    layout()->supplyLayout([&](Type layoutType) -> Value {
      Value layout = finalizeLayoutFn(memberNameInParent, layoutType);
      val()->addMember("@layout", layout);
      return layout;
    });
    layoutBuilder.reset();
  }

  Value addLayoutMember(Location loc, StringRef name, Type type) {
    return addLayoutMember(loc, builder.getStringAttr(name), type);
  }
  Value addLayoutMember(Location loc, StringAttr name, Type type) {
    assert(layout());
    if (!name || name.strref().empty())
      name = builder.getStringAttr("_" + std::to_string(subIndex++));
    return layout()->addMember(loc, name, type);
  }

  StringAttr getTypeName() const { return typeName; }

  // Checks to see if the single result value of the operation is
  // being used in a context that provides a name, otherwise uses an
  // anonymous layout name.
  StringAttr getResultName(Value resultValue) {
    StringAttr name;
    bool isSuper = false;
    for (auto& resultUse : resultValue.getUses()) {
      Operation* useOp = resultUse.getOwner();
      if (!useOp)
        continue;
      TypeSwitch<Operation*>(useOp)
          .Case<DefinitionOp>([&](auto defOp) {
            auto declaration = cast<DeclarationOp>(defOp.getDeclaration().getDefiningOp());
            name = declaration.getMemberAttr();
          })
          .Case<SuperOp>([&](auto) { isSuper = true; });
    }

    if (isSuper) {
      // It's more important to tag members as "@super" than it is to
      // use the specified name, so we can find a common subcomponent
      // when we only have the layout.  TODO: Implement proper
      // aliasing so we can refer to members by either the provided name or by @super.
      name = builder.getStringAttr("@super");
    } else if (!name || name.strref().empty()) {
      name = builder.getStringAttr("_" + std::to_string(subIndex++));
    }
    return name;
  }

  // Returns a layout for an operation returning the given result.
  Value addLayoutMember(Location loc, Value resultValue, Type type) {
    return addLayoutMember(loc, getResultName(resultValue), type);
  }
  // Mark this content's layout as a "mux" layout with a common supertype.
  void setLayoutKind(ZStruct::LayoutKind kind) {
    assert(layout());
    layout()->setKind(kind);
  }

  ZStruct::LayoutType getLayoutTypeSoFar() {
    assert(layout());
    return layout()->getType();
  }
  StructType getValueTypeSoFar() {
    assert(val());
    return val()->getType();
  }

private:
  void push() {
    assert(parent);
    if (!memberNameInParent || memberNameInParent.strref().empty()) {
      memberNameInParent = builder.getStringAttr("_" + std::to_string(parent->subIndex++));
    }

    typeName = builder.getStringAttr(parent->typeName.strref() + "_" + memberNameInParent.strref());
    layoutBuilder = std::make_unique<LayoutBuilder>(builder, typeName);
    valBuilder = std::make_unique<StructBuilder>(builder, typeName);
  }

  Location loc;
  OpBuilder& builder;
  StringAttr typeName;

  std::unique_ptr<LayoutBuilder> layoutBuilder;
  std::unique_ptr<StructBuilder> valBuilder;

  // Currently index of anonymous subcomponents
  size_t subIndex = 0;

  // If we're a subcomponent, the context we have in the parent.
  ComponentBuilder* parent = nullptr;
  StringAttr memberNameInParent;
};

// A converter for a top-level component definition, which converts a
// single zhl.component to a zhlt.component.
class LoweringImpl {
public:
  LoweringImpl(OpBuilder& builder, ComponentManager* componentManager, Operation* regionAnchor)
      : ctx(builder.getContext())
      , builder(builder)
      , componentManager(componentManager)
      , regionAnchor(regionAnchor) {}

  Zhlt::ComponentOp gen(ComponentOp origOp, StringAttr mangledName, ArrayRef<Attribute> typeArgs);

private:
  void gen(ConstructorParamOp, Block* topBlock);
  void gen(TypeParamOp, ArrayRef<Attribute> typeArgs);

  void gen(Region&, ComponentBuilder&);
  void gen(Block*, ComponentBuilder&);
  void gen(Operation*, ComponentBuilder&);
  void gen(LiteralOp, ComponentBuilder&);
  void gen(StringOp, ComponentBuilder&);
  void gen(GlobalOp, ComponentBuilder&);
  void gen(LookupOp, ComponentBuilder&);
  void gen(SubscriptOp, ComponentBuilder&);
  void gen(SpecializeOp, ComponentBuilder&);
  void gen(ConstructOp, ComponentBuilder&);
  void gen(DirectiveOp, ComponentBuilder&);
  void gen(BlockOp, ComponentBuilder&);
  void gen(MapOp, ComponentBuilder&);
  void gen(ReduceOp, ComponentBuilder&);
  void gen(SwitchOp, ComponentBuilder&);
  void gen(RangeOp, ComponentBuilder&);
  void gen(BackOp, ComponentBuilder&);
  void gen(ArrayOp, ComponentBuilder&);
  void gen(DefinitionOp, ComponentBuilder&);
  void gen(DeclarationOp, ComponentBuilder&);
  void gen(ConstraintOp, ComponentBuilder&);
  void gen(SuperOp, ComponentBuilder&);
  void gen(ExternOp, ComponentBuilder&);
  void gen(ConstructGlobalOp, ComponentBuilder&);
  void gen(GetGlobalOp, ComponentBuilder&);

  // Creates a sequence of lookup ops from the given ZHLT value to the indicated
  // member, inserting @super lookups as necessary.
  Value lookup(Value component, StringRef member);

  // Coerces the given component to its first array-like super, and then looks
  // up the given index.
  Value subscript(Value component, Value index);

  void buildZeroInitialize(Value toInit);

  /// Walk the value's super chain until type is reached or the chain ends.
  Value coerceTo(Value value, Type type);

  /// Walk the value's super chain until we hit an array or the chain ends
  Value coerceToArray(Value value);

  /// Extract the layout from the given ZHL value.  This may either be
  /// a layout, or an array of values which might have layouts
  /// elementwise.
  Value asAliasableLayout(Value);

  /// Extract the two given aliasable layouts together.  If either are
  /// ArrayType (as opposed to LayoutArrayType), they will be treated
  /// as arrays of values from which we can extract `@layout` fields.
  void genAliasLayout(Location loc, Value left, Value right);

  /// Alias the layouts of two arrays.  If convertLeftValue is true,
  /// `left` is an ArrayType of `Value` objects; their layout will
  /// need to be extracted from the `@layout` member.
  void genAliasLayoutArray(
      Location loc, Value left, bool convertLeftValue, Value right, bool convertRightValue);

  Value expandLayoutMember(Location loc, ComponentBuilder& cb, Value origLayout, Type newType);

  // Converts the given value/attribute into a constant attribute by
  // interpreting it if it's not already an attribute.
  Attribute asConstant(Value v);
  StringAttr asTypeName(Value v);
  PolynomialAttr asFieldElement(Value v);

  Value asValue(Value zhlVal);

  // Same as addLayoutMember, but handles the case where we add a
  // definition for a previously declared member.
  Value addOrExpandLayoutMember(Location loc, ComponentBuilder& cb, Value result, Type type) {
    assert(cb.layout());
    // If the new member has no layout, don't make any changes to the layout
    if (!type)
      return Value();

    Value layoutValue;
    StringAttr name;
    bool isSuper = false;
    for (Operation* useOp : result.getUsers()) {
      TypeSwitch<Operation*>(useOp)
          .Case<DefinitionOp>([&](auto defOp) {
            auto declaration = cast<DeclarationOp>(defOp.getDeclaration().getDefiningOp());
            name = declaration.getMemberAttr();

            // If previously declared, use existing layout instead of adding a new field
            layoutValue = layoutMapping.lookup(declaration.getOut());
          })
          .Case<SuperOp>([&](auto) { isSuper = true; });
    }

    if (layoutValue) {
      Value expanded = expandLayoutMember(loc, cb, layoutValue, type);
      if (!expanded) {
        auto diag = emitError(loc) << "definition of type `" << getTypeId(type)
                                   << "` is not a subtype of the declared type `"
                                   << getTypeId(layoutValue.getType()) << "`";
        diag.attachNote(layoutValue.getLoc()) << "declared here:";
        throw MalformedIRException();
      }
      assert(expanded.getType() == type);
      return expanded;
    }

    if (isSuper) {
      // It's more important to tag members as "@super" than it is to
      // use the specified name, so we can find a common subcomponent
      // when we only have the layout.  TODO: Implement proper
      // aliasing so we can refer to members by either the provided name or by @super.
      name = builder.getStringAttr("@super");
    }
    return cb.addLayoutMember(loc, name, type);
  }

  // Generates a specialization of "Array". reports errors to "op".
  Zhlt::ComponentOp genArrayCtor(Operation* op, Type elementType, size_t size);

  // Reconstructs a value from layout, or returns a null value if not possible.
  Value reconstructFromLayout(mlir::Location loc, mlir::Value layout, size_t distance = 0) {
    while (layout) {
      LLVM_DEBUG({ llvm::dbgs() << loc << ": reconstruct from layout: " << layout << "\n"; });
      if (auto layoutArrayType = llvm::dyn_cast<ZStruct::LayoutArrayType>(layout.getType())) {
        // Emit a map to reconstruct each of the elements' layouts.
        Region mapBody(regionAnchor);
        size_t size = layoutArrayType.getSize();
        Type emptyElemType = Zhlt::getComponentType(ctx);
        Value emptyElem = builder.create<ZStruct::PackOp>(loc, emptyElemType, ValueRange{});
        Type bodyElemType;
        {
          OpBuilder::InsertionGuard insertionGuard(builder);
          Block* bodyBlock = builder.createBlock(&mapBody);
          bodyBlock->addArgument(emptyElemType, loc);
          Value elemLayout = bodyBlock->addArgument(layoutArrayType.getElement(), loc);
          Value reconstructed = reconstructFromLayout(loc, elemLayout, distance);
          if (!reconstructed)
            return {};
          bodyElemType = reconstructed.getType();
          builder.create<ZStruct::YieldOp>(loc, reconstructed);
        }

        SmallVector<Value> emptyElems(size, emptyElem);
        Value emptyArray = builder.create<ZStruct::ArrayOp>(
            loc, builder.getType<ArrayType>(emptyElemType, size), emptyElems);

        Type bodyArrayType = builder.getType<ArrayType>(bodyElemType, size);
        ZStruct::MapOp mapOp =
            builder.create<ZStruct::MapOp>(loc, bodyArrayType, emptyArray, layout);
        mapOp.getBody().takeBody(mapBody);
        return mapOp.getOut();
      }
      if (Value reconstructed =
              componentManager->reconstructFromLayout(builder, loc, layout, distance)) {
        return reconstructed;
      }

      Type superType = Zhlt::getSuperType(layout.getType());
      if (!superType)
        return {};

      layout = coerceTo(layout, superType);
    }
    return {};
  }

  MLIRContext* ctx;
  OpBuilder builder;
  ComponentManager* componentManager;

  // Type names returned by ZHL operations
  DenseMap</*ZhlOp=*/Value, /*TypeName=*/StringAttr> typeNameMapping;
  // Values in ZHL operations that have been converted to ZHLT operations
  DenseMap</*ZhlOp=*/Value, /*ZHLT value=*/Value> valueMapping;
  // Layouts in ZHL operations that have been converted to ZHLT operations
  DenseMap</*ZhlOp=*/Value, /*layout=*/Value> layoutMapping;

  // Layout types for globals
  DenseMap<StringAttr, ZStruct::LayoutType> globalLayouts;

  // Temporary anchor when building regions so that evaluating and ASM printing can find the module
  // we're working on.
  Operation* regionAnchor;
};

void LoweringImpl::gen(Operation* op, ComponentBuilder& cb) {
  LLVM_DEBUG({
    llvm::dbgs() << "Typing operation ";
    op->print(llvm::dbgs(), OpPrintingFlags().printGenericOpForm());
    llvm::dbgs() << "\n";
  });

  llvm::TypeSwitch<Operation*, void>(op)
      .Case<ComponentOp>(
          [&](ComponentOp op) { op.emitError() << "named nested components are not allowed"; })
      .Case<TypeParamOp,
            LiteralOp,
            StringOp,
            GlobalOp,
            LookupOp,
            SubscriptOp,
            SpecializeOp,
            ConstructOp,
            DirectiveOp,
            BlockOp,
            MapOp,
            ReduceOp,
            SwitchOp,
            RangeOp,
            BackOp,
            ArrayOp,
            DefinitionOp,
            DeclarationOp,
            ConstraintOp,
            SuperOp,
            ExternOp,
            ConstructGlobalOp,
            GetGlobalOp>([&](auto op) { gen(op, cb); })
      .Default([&](auto) { op->emitError("Unexpected operation"); });
}

void LoweringImpl::gen(ConstructorParamOp ctorParam, Block* topBlock) {
  auto paramType = asTypeName(ctorParam.getType());
  if (!paramType) {
    ctorParam.emitError("constructor parameter must be a type attribute");
    throw MalformedIRException();
  }
  auto ctor = componentManager->lookupComponent(paramType);
  Type valueType;
  Type layoutType;
  if (ctor) {
    valueType = ctor.getOutType();
    layoutType = ctor.getLayoutType();
  } else {
    ctorParam.emitError("expected a valid parameter type");
    valueType = Zhlt::getComponentType(ctorParam.getContext());
  }
  if (ctorParam.getVariadic()) {
    valueType = VariadicType::get(ctx, valueType);
  }
  auto value = topBlock->addArgument(valueType, ctorParam.getLoc());
  valueMapping[ctorParam.getOut()] = value;
}

void LoweringImpl::gen(TypeParamOp typeParam, ArrayRef<Attribute> typeArgs) {
  if (typeParam.getIndex() >= typeArgs.size()) {
    typeParam.emitError() << "expected a value for type parameter #" << typeParam.getIndex();
    throw MalformedIRException();
  }
  Attribute paramValue = typeArgs[typeParam.getIndex()];

  StringAttr paramTypeName = asTypeName(typeParam.getType());
  if (!paramTypeName) {
    typeParam.emitError("a type parameter's type must be a compile-time constant");
    throw MalformedIRException();
  }
  if (paramTypeName == "Type") {
    if (auto typeName = llvm::dyn_cast<StringAttr>(paramValue))
      typeNameMapping[typeParam] = typeName;
    else {
      typeParam.emitError("expected a type parameter of type `Type`");
      throw MalformedIRException();
    }
  } else if (paramTypeName == "Val") {
    if (!isa<PolynomialAttr>(paramValue)) {
      typeParam.emitError("expected a type parameter of type `Val`");
      paramValue = PolynomialAttr::get(ctx, {0});
    }

    auto constOp =
        builder.create<Zll::ConstOp>(typeParam.getLoc(), llvm::cast<PolynomialAttr>(paramValue));
    valueMapping[typeParam.getOut()] = constOp;
  } else {
    typeParam.emitError("a type parameter's type should be either `Type` or `Val`");
    throw MalformedIRException();
  }
}

Zhlt::ComponentOp LoweringImpl::gen(ComponentOp component,
                                    StringAttr mangledName,
                                    llvm::ArrayRef<Attribute> typeArgs) {
  Region body(regionAnchor);
  Block* bodyBlock;
  Location loc = component.getLoc();
  Type layoutType, valueType;
  SmallVector<Type> constructArgsTypes;

  LLVM_DEBUG({
    llvm::dbgs() << "Converting ZHL component " << component.getName() << " to ZHLT component "
                 << mangledName << " with type args <";
    interleaveComma(typeArgs, llvm::dbgs());
    llvm::dbgs() << ">\n";
  });

  {
    OpBuilder::InsertionGuard insertionGuard(builder);
    bodyBlock = builder.createBlock(&body);

    ComponentBuilder cb(loc, builder, mangledName);
    if (component->hasAttr("argument")) {
      cb.setLayoutKind(ZStruct::LayoutKind::Argument);
    }

    try {
      for (Operation& op : component.getBody().front()) {
        // Handle operations which are only present in the top block of the component.
        TypeSwitch<Operation*>(&op)
            .Case<ConstructorParamOp>([&](auto op) { gen(op, bodyBlock); })
            .Case<TypeParamOp>([&](auto op) { gen(op, typeArgs); })
            .Default([&](auto op) { gen(op, cb); });
      }
    } catch (MalformedIRException&) {
      // All semantic errors, even those that are difficult to recover from, are
      // confined to the component definition in which they occur, so we can
      // resume lowering from here.

      if (!valueType)
        valueType = Zhlt::getComponentType(ctx);
    }

    constructArgsTypes = llvm::to_vector(bodyBlock->getArgumentTypes());
    cb.supplyLayout([&](StringAttr memberNameInParent, Type layoutTypeArg) -> Value {
      assert(!memberNameInParent && "Top-level components should not have a member name");
      layoutType = layoutTypeArg;
      return bodyBlock->addArgument(layoutTypeArg, loc);
    });
    Value returnValue = cb.getValue(loc);
    builder.create<Zhlt::ReturnOp>(loc, returnValue);
    valueType = returnValue.getType();
  }

  auto ctor = builder.create<Zhlt::ComponentOp>(
      loc, mangledName, valueType, constructArgsTypes, layoutType);
  ctor.getBody().takeBody(body);

  for (NamedAttribute attr : component->getDiscardableAttrs()) {
    StringRef name = attr.getName();
    if (name == "function" || name == "argument" || name == "generic" || name == "picus_analyze" ||
        name == "picus_inline") {
      ctor->setAttr(name, attr.getValue());
    } else {
      ctor->emitError() << "unknown attribute `" << name << "`";
    }
  }

  return ctor;
}

void LoweringImpl::gen(ExternOp ext, ComponentBuilder& cb) {
  // Since zirgen::ExternOp doesn't know how to pass around a
  // structure, we walk through the structure we're supposed to send
  // and ocllect all the value types we need to send to the extern.
  SmallVector<mlir::Value> params;
  std::function<void(Value)> gatherParams;
  gatherParams = [&](Value val) {
    TypeSwitch<Type>(val.getType())
        .Case<ValType>([&](ValType ty) { params.push_back(val); })
        .Case<StringType>([&](StringType ty) { params.push_back(val); })
        .Case<StructType>([&](StructType ty) {
          for (auto field : ty.getFields()) {
            auto fieldVal = builder.create<ZStruct::LookupOp>(ext.getLoc(), val, field.name);
            gatherParams(fieldVal);
          }
        })
        .Case<VariadicType>([&](auto) { params.push_back(val); })
        .Default([&](auto ty) {
          ext.emitError() << "Unsupported extern parameter type " << ty;
          throw MalformedIRException();
        });
  };
  for (auto param : ext.getArgs()) {
    gatherParams(asValue(param));
  }

  // Since zirgen::ExternOp doesn't know how to return a structure, we
  // walk through the structure we're supposed to be returning and
  // collect all the value types we need to return from the extern.
  auto returnTypeAttr = asTypeName(ext.getReturnType());
  if (!returnTypeAttr) {
    ext.emitError() << "Unsupported non-const extern return type";
    throw MalformedIRException();
  }
  auto returnCtor = componentManager->lookupComponent(returnTypeAttr);
  Type returnType = returnCtor.getOutType();
  llvm::SmallVector<Type> valTypes;
  std::function<void(Type)> countVals;
  countVals = [&](Type ty) {
    TypeSwitch<Type>(ty)
        .Case<ValType>([&](ValType ty) { valTypes.push_back(ty); })
        .Case<StructType>([&](StructType ty) {
          for (auto field : ty.getFields()) {
            countVals(field.type);
          }
        })
        .Default([&](auto ty) {
          ext.emitError() << "Unsupported extern return type " << ty;
          throw MalformedIRException();
        });
  };
  countVals(returnType);

  auto extOp =
      builder.create<Zll::ExternOp>(ext.getLoc(), valTypes, params, ext.getName(), /*extra=*/"");

  // Now, walk through the structure we're supposed to be returning and pack
  // it with the results from the Zll::ExternOp.
  size_t outIndex = 0;
  std::function<mlir::Value(Type)> wrapVals;
  wrapVals = [&](Type ty) -> mlir::Value {
    return TypeSwitch<Type, mlir::Value>(ty)
        .Case<ValType>([&](ValType ty) { return extOp.getOut()[outIndex++]; })
        .Case<StructType>([&](StructType ty) {
          llvm::SmallVector<Value> fields;
          for (auto field : ty.getFields()) {
            fields.push_back(wrapVals(field.type));
          }
          return builder.create<ZStruct::PackOp>(ext.getLoc(), ty, fields);
        })
        .Default([&](auto ty) {
          ext.emitError() << "Unsupported extern return type " << ty;
          return mlir::Value();
        });
  };
  auto wrapped = wrapVals(returnType);
  assert(extOp.getOut().size() == outIndex);

  valueMapping[ext] = wrapped;
}

void LoweringImpl::gen(ConstructGlobalOp construct, ComponentBuilder& cb) {
  StringAttr typeNameAttr = asTypeName(construct.getConstructType());
  if (!typeNameAttr) {
    construct.emitError("an invoked constructor must be a compile-time constant");
    throw MalformedIRException();
  }
  Zhlt::ComponentOp ctor = componentManager->lookupComponent(typeNameAttr);
  if (!ctor) {
    construct.emitError("cannot construct an undefined type");
    throw MalformedIRException();
  }

  auto& layoutType = globalLayouts[construct.getNameAttr()];
  if (layoutType && layoutType != ctor.getLayoutType()) {
    construct.emitError() << "Mismatched types (" << layoutType << " vs " << ctor.getLayoutType()
                          << ") for global";
    throw MalformedIRException();
  }
  layoutType = llvm::cast<ZStruct::LayoutType>(ctor.getLayoutType());

  if (construct.getArgs().size() != ctor.getConstructParamTypes().size()) {
    construct.emitError() << "Mismatched arguments for global";
    throw MalformedIRException();
  }

  SmallVector<Value> args;
  for (auto argAndType : llvm::zip_equal(construct.getArgs(), ctor.getConstructParamTypes())) {
    auto [arg, argType] = argAndType;
    auto argVal = asValue(arg);
    if (!Zhlt::isCoercibleTo(argVal.getType(), argType)) {
      construct.emitError() << "argument of type `" << getTypeId(argVal.getType())
                            << "` is not convertible to `" << getTypeId(argType) << "`";
      return;
    }
    args.push_back(coerceTo(argVal, argType));
  }

  auto layout =
      builder.create<Zhlt::GetGlobalLayoutOp>(construct.getLoc(), layoutType, construct.getName());
  builder.create<Zhlt::ConstructOp>(
      construct.getLoc(), typeNameAttr, ctor.getOutType(), args, layout);
}

void LoweringImpl::gen(GetGlobalOp getGlobal, ComponentBuilder& cb) {
  StringAttr typeNameAttr = asTypeName(getGlobal.getConstructType());
  if (!typeNameAttr) {
    getGlobal.emitError("an invoked constructor must be a compile-time constant");
    throw MalformedIRException();
  }

  Zhlt::ComponentOp ctor = componentManager->lookupComponent(typeNameAttr);
  if (!ctor) {
    getGlobal.emitError("cannot get a global with an undefined type");
    throw MalformedIRException();
  }

  auto& layoutType = globalLayouts[getGlobal.getNameAttr()];
  if (layoutType && layoutType != ctor.getLayoutType()) {
    getGlobal.emitError() << "Mismatched types (" << layoutType << " vs " << ctor.getLayoutType()
                          << ") for global";
    throw MalformedIRException();
  }
  layoutType = llvm::cast<ZStruct::LayoutType>(ctor.getLayoutType());

  layoutMapping[getGlobal.getOut()] =
      builder.create<Zhlt::GetGlobalLayoutOp>(getGlobal.getLoc(), layoutType, getGlobal.getName());
}

void LoweringImpl::gen(LiteralOp literal, ComponentBuilder& cb) {
  auto constOp = builder.create<Zll::ConstOp>(literal.getLoc(), literal.getValue());
  valueMapping[literal.getOut()] = constOp.getOut();
}

void LoweringImpl::gen(StringOp string, ComponentBuilder& cb) {
  // Strings are only used for logging, so there's no point in doing constant
  // propagation on them. Lower directly to a low-level StringOp instead of a
  // StringAttr, since StringAttrs in the scope table currently denote symbols.
  auto op = builder.create<Zll::StringOp>(string.getLoc(), string.getValue());
  valueMapping[string.getOut()] = op.getOut();
}

void LoweringImpl::gen(GlobalOp global, ComponentBuilder& cb) {
  if (componentManager->isGeneric(global.getName())) {
    typeNameMapping[global.getOut()] = global.getNameAttr();
  } else {
    Zhlt::ComponentOp c =
        componentManager->getComponent(global.getLoc(), global.getNameAttr(), /*typeArgs=*/{});
    if (!c)
      throw MalformedIRException();
    typeNameMapping[global.getOut()] = c.getNameAttr();
  }
}

Value LoweringImpl::lookup(Value component, StringRef member) {
  auto componentType = Zhlt::getComponentType(ctx);
  while (component.getType() && component.getType() != componentType) {
    ArrayRef<ZStruct::FieldInfo> fields;
    if (auto structType = dyn_cast<StructType>(component.getType()))
      fields = structType.getFields();
    else if (auto layoutType = dyn_cast<LayoutType>(component.getType()))
      fields = layoutType.getFields();
    else
      break;

    bool foundSuper = false;
    for (ZStruct::FieldInfo field : fields) {
      if (field.name == member && !field.isPrivate) {
        return builder.create<ZStruct::LookupOp>(component.getLoc(), component, member);
      }
      foundSuper |= (field.name == "@super");
    }
    // An error upstream may have produced a malformed component struct.
    if (!foundSuper)
      break;
    component = builder.create<ZStruct::LookupOp>(component.getLoc(), component, "@super");
  }
  // If we haven't returned yet, we searched the whole super chain and didn't
  // find the member we're looking for. Return a null value and handle the error
  // upstream.
  return nullptr;
}

Value LoweringImpl::subscript(Value array, Value index) {
  auto derivedType = array.getType();
  ZStruct::ArrayLikeTypeInterface arrayType = Zhlt::getCoercibleArrayType(derivedType);
  if (!arrayType) {
    if (derivedType) {
      emitError(array.getLoc()) << "subscripted component of type `" << getTypeId(derivedType)
                                << "` is not convertible to an array type.";
    } else {
      emitError(array.getLoc()) << "subscript requires an instance of an array type";
    }
    // Since we don't know what the element type is supposed to be, there isn't
    // a good way to recover from this error.
    throw MalformedIRException();
  }
  Value casted = coerceTo(array, arrayType);
  return builder.create<ZStruct::SubscriptOp>(array.getLoc(), casted, index);
}

void LoweringImpl::gen(LookupOp lookupOp, ComponentBuilder& cb) {
  Value component = asValue(lookupOp.getComponent());
  StringRef member = lookupOp.getMember();
  Value subcomponent = lookup(component, member);
  if (!subcomponent) {
    emitError(component.getLoc()) << "type `" << getTypeId(component.getType())
                                  << "` has no member named \"" << member << "\"";
  }
  valueMapping[lookupOp.getOut()] = subcomponent;
}

void LoweringImpl::gen(SubscriptOp subscriptOp, ComponentBuilder& cb) {
  Value array = asValue(subscriptOp.getArray());
  Value index = asValue(subscriptOp.getElement());
  Value newSubscriptOp = subscript(array, index);
  valueMapping[subscriptOp.getOut()] = newSubscriptOp;
}

void LoweringImpl::gen(SpecializeOp specialize, ComponentBuilder& cb) {
  StringAttr typeNameAttr = asTypeName(specialize.getType());
  if (!typeNameAttr) {
    specialize.emitError("The type to be specialized must be a compile-time constant");
    return;
  }

  SmallVector<Attribute> typeArguments;
  for (Value arg : specialize.getArgs()) {
    Attribute typedArg = asConstant(arg);
    if (!typedArg || !Zhlt::isLegalTypeArg(typedArg)) {
      // recover from this error by synthesizing a type parameter value.
      arg.getDefiningOp()->emitError("type parameter must be a compile-time constant");
      typedArg = PolynomialAttr::get(ctx, {0});
    }
    typeArguments.push_back(typedArg);
  }

  Zhlt::ComponentOp component =
      componentManager->getComponent(specialize.getLoc(), typeNameAttr, typeArguments);
  if (!component)
    throw MalformedIRException();
  typeNameMapping[specialize.getOut()] = component.getNameAttr();
}

void LoweringImpl::gen(ConstructOp construct, ComponentBuilder& cb) {
  StringAttr typeNameAttr = asTypeName(construct.getType());
  if (!typeNameAttr) {
    // Since we don't know what type the constructed component is supposed to
    // be, there isn't a great way to recover and even if we did there would be
    // spurious diagnostics for every use of the constructed value.
    construct.emitError("an invoked constructor must be a compile-time constant");
    throw MalformedIRException();
  }
  Zhlt::ComponentOp ctor = componentManager->lookupComponent(typeNameAttr);
  if (!ctor) {
    // We might fail to look up an erroneous type name.
    construct.emitError("cannot construct an undefined type");
    throw MalformedIRException();
  }
  SmallVector<Value> arguments;
  auto argumentTypes = ctor.getConstructParamTypes();
  auto expectedArgType = argumentTypes.begin();
  SmallVector<Value> variadicArguments;
  for (Value zhlArg : construct.getArgs()) {
    Value arg = asValue(zhlArg);
    if (expectedArgType == argumentTypes.end()) {
      construct.emitError() << "expected " << argumentTypes.size()
                            << " arguments in component constructor, got "
                            << construct.getArgs().size();
      return;
    }
    if (!Zhlt::isCoercibleTo(arg.getType(), *expectedArgType)) {
      construct.emitError() << "argument of type `" << getTypeId(arg.getType())
                            << "` is not convertible to `" << getTypeId(*expectedArgType) << "`";
      return;
    }
    Value casted = coerceTo(arg, *expectedArgType);
    if (isa<VariadicType>(*expectedArgType)) {
      // Only the last parameter may be variadic, which is caught in the
      // AST -> ZHL lowering, so we no longer need to advance expectedArgType.
      variadicArguments.push_back(casted);
    } else {
      expectedArgType++;
      arguments.push_back(casted);
    }
  }
  // If there is a variadic parameter, add its pack to the argument list
  if (expectedArgType != argumentTypes.end() && isa<VariadicType>(*expectedArgType)) {
    auto packed = builder.create<Zll::VariadicPackOp>(
        construct.getLoc(), *expectedArgType, variadicArguments);
    arguments.push_back(packed);
    // advance the iterator past the (expectedly last) variadic parameter for
    // the subsequent check that all arguments were provided
    expectedArgType++;
  }
  // If we still haven't reached the end of the parameters, there aren't enough arguments
  if (expectedArgType != argumentTypes.end()) {
    bool isVariadic = (!argumentTypes.empty() && isa<VariadicType>(argumentTypes.back()));
    size_t minimumArgCount = isVariadic ? argumentTypes.size() - 1 : argumentTypes.size();
    size_t actualArgCount = construct.getArgs().size();
    auto diag = construct.emitError() << "expected ";
    if (isVariadic)
      diag << "at least ";
    diag << minimumArgCount << " arguments in component constructor, got " << actualArgCount;
    return;
  }

  // Don't expand the member layout until the actual definition -- even in the
  // arguments to the ConstructOp. This ensures that we don't leave any dangling
  // uses of the "old" layout after we expand it, since the declaration should
  // not be referenced after the definition.
  Zhlt::ConstructOp call;
  Value tmpLayout;
  if (ctor.getLayoutType()) {
    tmpLayout = builder.create<Zhlt::MagicOp>(construct.getLoc(), ctor.getLayoutType());
  }
  call = builder.create<Zhlt::ConstructOp>(construct.getLoc(), ctor, arguments, tmpLayout);

  // Now that we've built the constructor call, expand the layout and clean up
  // the temporary one if necessary.
  Value layout;
  if (tmpLayout) {
    layout =
        addOrExpandLayoutMember(construct.getLoc(), cb, construct.getOut(), ctor.getLayoutType());
    tmpLayout.replaceAllUsesWith(layout);
    tmpLayout.getDefiningOp()->erase();
  }

  if (layout && ctor->getAttr("alwaysinline")) {
    construct.emitError() << "Cannot construct non-trivial layouts inside of a function";
    throw MalformedIRException();
  }

  valueMapping[construct.getOut()] = call.getOut();
  layoutMapping[construct.getOut()] = layout;
}

// Gets the value of the layout corresponding to a ZHL value
Value LoweringImpl::asAliasableLayout(Value value) {
  Value layout = layoutMapping[value];
  if (layout)
    return layout;

  Value val = valueMapping[value];
  if (!val)
    return {};
  Value superLayout = lookup(val, "@layout");
  if (superLayout) {
    layoutMapping[value] = superLayout;
    return superLayout;
  }

  return coerceToArray(val);
}

void LoweringImpl::genAliasLayoutArray(
    Location loc, Value left, bool convertLeftValue, Value right, bool convertRightValue) {
  ZStruct::ArrayLikeTypeInterface leftArrayType = Zhlt::getCoercibleArrayType(left.getType());
  ZStruct::ArrayLikeTypeInterface rightArrayType = Zhlt::getCoercibleArrayType(right.getType());
  if (!leftArrayType) {
    auto diag = emitError(loc) << "Unable to coerce value into array: " << left;
    diag.attachNote(left.getLoc()) << "this value";
  }
  if (!rightArrayType) {
    auto diag = emitError(loc) << "Unable to coerce value into array: " << right;
    diag.attachNote(right.getLoc()) << "this value";
  }
  if (!leftArrayType || !rightArrayType) {
    return;
  }

  if (leftArrayType.getSize() != rightArrayType.getSize()) {
    emitError(loc) << "Unable to coerce " << left << " and " << right
                   << " to the same size array\n";
    return;
  }

  Value leftArray = coerceToArray(left);
  Value rightArray = coerceToArray(right);

  for (auto i : llvm::seq(leftArrayType.getSize())) {
    auto constOp = builder.create<Zll::ConstOp>(loc, i);
    Value leftElem = builder.create<ZStruct::SubscriptOp>(loc, leftArray, constOp);
    Value rightElem = builder.create<ZStruct::SubscriptOp>(loc, rightArray, constOp);

    if (convertLeftValue) {
      leftElem = lookup(leftElem, "@layout");
    }
    if (convertRightValue) {
      rightElem = lookup(rightElem, "@layout");
    }
    genAliasLayout(loc, leftElem, rightElem);
  }
}

void LoweringImpl::genAliasLayout(Location loc, Value left, Value right) {
  bool leftArray = llvm::isa<ArrayType>(left.getType());
  bool rightArray = llvm::isa<ArrayType>(right.getType());

  if (leftArray || rightArray) {
    genAliasLayoutArray(loc, left, leftArray, right, rightArray);
    return;
  }
  Type type = Zhlt::getLeastCommonSuper({left.getType(), right.getType()}, /*isLayout=*/true);
  if (!type)
    mlir::emitError(loc) << "attempting to alias layouts without a common super";

  if (left.getType() == right.getType() || !llvm::isa<LayoutArrayType>(type)) {
    left = coerceTo(left, type);
    right = coerceTo(right, type);
    builder.create<ZStruct::AliasLayoutOp>(loc, left, right);
  } else {
    genAliasLayoutArray(loc, left, /*convertLeftValue=*/false, right, /*convertRightValue=*/false);
  }
}

Value LoweringImpl::expandLayoutMember(Location loc,
                                       ComponentBuilder& cb,
                                       Value origLayout,
                                       Type newType) {
  if (origLayout.getType() == newType)
    return origLayout;

  bool isCoercible = Zhlt::isCoercibleTo(newType, origLayout.getType(), /*isLayout=*/true);
  bool isCoercibleArray = false;
  auto origLayoutArrayType = Zhlt::getCoercibleArrayType(origLayout.getType());
  auto newLayoutArrayType = Zhlt::getCoercibleArrayType(newType);
  if (origLayoutArrayType && newLayoutArrayType &&
      Zhlt::isCoercibleTo(
          newLayoutArrayType.getElement(), origLayoutArrayType.getElement(), /*isLayout=*/true))
    isCoercibleArray = true;

  if (!(isCoercible || isCoercibleArray)) {
    emitError(loc) << "Unable to expand layout " << origLayout.getType()
                   << "\nto incompatible type " << newType;
    return {};
  }

  auto lookupOp = origLayout.getDefiningOp<ZStruct::LookupOp>();
  if (!lookupOp)
    return {};
  auto memberName = lookupOp.getMember();
  auto newMember = cb.addLayoutMember(loc, memberName.str() + "$redef", newType);
  genAliasLayout(loc, origLayout, newMember);
  return newMember;
}

void LoweringImpl::gen(DirectiveOp directive, ComponentBuilder& cb) {
  if (directive.getName() == "AliasLayout") {
    if (directive.getArgs().size() != 2) {
      size_t args = directive.getArgs().size();
      directive.emitError() << "'AliasLayout' directive expects two arguments, got " << args;
    }
    Value left = asAliasableLayout(directive.getArgs()[0]);
    Value right = asAliasableLayout(directive.getArgs()[1]);
    if (left && right) {
      genAliasLayout(directive.getLoc(), left, right);
      return;
    }
    if (!left)
      directive.emitError() << "Unable to determine layout of " << directive.getArgs()[0];
    if (!right)
      directive.emitError() << "Unable to determine layout of " << directive.getArgs()[1];
  } else {
    directive.emitError() << "Unknown compiler directive '" << directive.getName() << "'";
  }
}

void LoweringImpl::gen(BlockOp block, ComponentBuilder& cb) {
  ComponentBuilder subBlock(block.getLoc(), &cb, block.getOut());
  gen(block.getInner(), subBlock);
  subBlock.supplyLayout([&](StringAttr memberName, Type layoutType) -> Value {
    Value layout = cb.addLayoutMember(block.getLoc(), memberName, layoutType);
    layoutMapping[block.getOut()] = layout;
    return layout;
  });
  valueMapping[block.getOut()] = subBlock.getValue(block.getLoc());
}

void LoweringImpl::gen(MapOp map, ComponentBuilder& cb) {
  Value array = coerceToArray(asValue(map.getArray()));
  if (!isa<ArrayType>(array.getType())) {
    map.emitError() << "this map expression's array value has non-array "
                       "type `"
                    << getTypeId(array.getType()) << "`";
    return;
  }
  size_t size = llvm::cast<ArrayType>(array.getType()).getSize();
  auto elemType = llvm::cast<ArrayType>(array.getType()).getElement();

  Region mapBody(regionAnchor);
  Value outLayout;
  Type outValueType;
  {
    OpBuilder::InsertionGuard insertionGuard(builder);
    Block* mapBodyBlock = builder.createBlock(&mapBody);
    ComponentBuilder subBlock(map.getLoc(), &cb, map.getOut());
    auto mapArg = map.getFunction().getArgument(0);
    valueMapping[mapArg] = mapBodyBlock->addArgument(elemType, mapArg.getLoc());
    gen(map.getFunction(), subBlock);
    subBlock.supplyLayout([&](StringAttr name, Type layoutType) -> Value {
      Value bodyLayout = mapBodyBlock->addArgument(layoutType, mapArg.getLoc());
      Type outLayoutType = builder.getType<ZStruct::LayoutArrayType>(layoutType, size);
      outLayout = addOrExpandLayoutMember(map.getLoc(), cb, map.getOut(), outLayoutType);
      return bodyLayout;
    });
    Value outValue = subBlock.getValue(mapArg.getLoc());
    outValueType = outValue.getType();
    builder.create<ZStruct::YieldOp>(map.getLoc(), outValue);
  }

  auto outArrayType = builder.getType<ArrayType>(outValueType, size);
  auto mapOp = builder.create<ZStruct::MapOp>(map.getLoc(), outArrayType, array, outLayout);
  mapOp.getBody().takeBody(mapBody);
  valueMapping[map.getOut()] = mapOp.getOut();
  layoutMapping[map.getOut()] = outLayout;
}

void LoweringImpl::gen(ReduceOp reduce, ComponentBuilder& cb) {
  Location loc = reduce.getLoc();
  Value array = asValue(reduce.getArray());
  Value init = asValue(reduce.getInit());
  auto typeName = asTypeName(reduce.getType());
  if (!typeName) {
    reduce.emitError("constructor of a reduce expression must be compile-time constant");
    throw MalformedIRException();
  }
  auto ctor = componentManager->lookupComponent(typeName);
  auto type = ctor.getOutType();
  if (ctor.getConstructParamTypes().size() != 2) {
    reduce.emitError() << "The constructor of a reduce expression should take 2 arguments, but "
                       << typeName << " takes " << ctor.getConstructParamTypes().size();
    return;
  }
  auto lhsType = ctor.getConstructParamTypes()[0];
  auto rhsType = ctor.getConstructParamTypes()[1];
  if (!Zhlt::isCoercibleTo(init.getType(), lhsType)) {
    reduce.emitError() << "this reduce expression's initial value must be coercible to `"
                       << getTypeId(lhsType) << "`";
  } else {
    init = coerceTo(init, lhsType);
  }
  if (!isa<ArrayType>(array.getType())) {
    reduce.emitError() << "this reduce expression's array value has non-array "
                          "type `"
                       << getTypeId(array.getType()) << "`";
    return;
  }
  auto arrayType = cast<ArrayType>(array.getType());
  auto elemType = arrayType.getElement();
  if (!Zhlt::isCoercibleTo(elemType, rhsType)) {
    reduce.emitError() << "this reduce expression's array's elements must be coercible to `"
                       << getTypeId(rhsType) << "`";
    return;
  }
  if (!Zhlt::isCoercibleTo(type, lhsType)) {
    reduce.emitError() << "this reduce expression's constructor must be coercible to its own "
                          "first argument type, `"
                       << getTypeId(lhsType) << "`";
    return;
  }
  size_t elems = arrayType.getSize();

  Region reduceBody(regionAnchor);
  {
    OpBuilder::InsertionGuard insertionGuard(builder);
    Block* bodyBlock = builder.createBlock(&reduceBody);
    auto lhs = bodyBlock->addArgument(lhsType, reduce.getLoc());
    auto rhs = bodyBlock->addArgument(elemType, reduce.getLoc());
    Value bodyLayout;
    if (ctor.getLayoutType())
      bodyLayout = bodyBlock->addArgument(ctor.getLayoutType(), reduce.getLoc());

    auto callReduceOp = builder.create<Zhlt::ConstructOp>(
        loc, typeName, type, ValueRange{lhs, coerceTo(rhs, rhsType)}, bodyLayout);
    builder.create<ZStruct::YieldOp>(loc, coerceTo(callReduceOp.getOut(), lhsType));
  }

  Value outLayout;
  if (ctor.getLayoutType()) {
    Type outLayoutType = builder.getType<ZStruct::LayoutArrayType>(ctor.getLayoutType(), elems);
    outLayout = cb.addLayoutMember(reduce.getLoc(), reduce.getOut(), outLayoutType);
  }
  auto reduceOp =
      builder.create<ZStruct::ReduceOp>(reduce.getLoc(), lhsType, array, init, outLayout);
  reduceOp.getBody().takeBody(reduceBody);
  valueMapping[reduce.getOut()] = reduceOp.getOut();
}

void LoweringImpl::gen(SwitchOp sw, ComponentBuilder& cb) {
  bool isMajor = sw->hasAttr("isMajor");

  Value origSelector = asValue(sw.getSelector());
  unsigned size = sw.getCases().size();
  ArrayType selectorType = ArrayType::get(ctx, Zhlt::getValType(ctx), size);
  if (!Zhlt::isCoercibleTo(origSelector.getType(), selectorType)) {
    sw.emitError() << "the selector of a mux with " << size << " arms must "
                   << "be convertible to Array<Val, " << size << ">";
    return;
  }
  auto selector = coerceTo(origSelector, selectorType);

  ComponentBuilder muxContext(sw.getLoc(), &cb, sw.getOut());
  muxContext.setLayoutKind(isMajor ? ZStruct::LayoutKind::MajorMux : ZStruct::LayoutKind::Mux);

  if (isMajor) {
    // Require major mux selectors to be register like
    if (!Zhlt::isCoercibleTo(origSelector.getType(),
                             ArrayType::get(ctx, Zhlt::getNondetRegType(ctx), size))) {
      sw.emitError() << "Major mux selectors must be registers";
      return;
    }
    // Make an array of selector regs in the mux layout
    LayoutArrayType selectorSaveType =
        LayoutArrayType::get(ctx, Zhlt::getNondetRegLayoutType(ctx), size);
    Value saveArray = muxContext.addLayoutMember(sw.getLoc(), "@selector", selectorSaveType);
    // Alias them to the actual selectors
    Value selectorArray = asAliasableLayout(sw.getSelector());
    genAliasLayout(sw.getLoc(), saveArray, selectorArray);
  }

  // Create components for each arm
  std::vector<std::unique_ptr<ComponentBuilder>> armContexts;
  std::vector<std::unique_ptr<Region>> armRegions;
  SmallVector<Type> armTypes;
  SmallVector<Type> armLayouts;
  for (size_t i = 0; i != size; ++i) {
    auto armLoc = sw.getCases()[i].getLoc();
    OpBuilder::InsertionGuard insertionGuard(builder);
    std::unique_ptr<Region> armRegion = std::make_unique<Region>(regionAnchor);
    builder.createBlock(armRegion.get());
    auto armContext =
        std::make_unique<ComponentBuilder>(armLoc, &muxContext, "arm" + std::to_string(i));
    gen(sw.getCases()[i], *armContext);

    Type armLayoutType = armContext->getLayoutTypeSoFar();
    LLVM_DEBUG({ llvm::dbgs() << "Switch arm " << i << " layout: " << armLayoutType << "\n"; });
    armTypes.push_back(armContext->getValueTypeSoFar());
    if (armLayoutType)
      armLayouts.push_back(armLayoutType);
    armContexts.emplace_back(std::move(armContext));
    armRegions.emplace_back(std::move(armRegion));
  }

  // Figure out the greatest number of each argument type used by any mux arm
  llvm::MapVector<Type, size_t> worstCase = Zhlt::muxArgumentCounts(armLayouts);

  // Don't propagate things if we are the major mux
  if (isMajor) {
    worstCase.clear();
  }

  // Hoist arguments out of the mux. Since the hoisted argument layouts will be
  // aliased on each arm of the mux, build a table of lookups for each argument
  // for reuse.
  llvm::MapVector<Type, SmallVector<Value>> hoistedArgumentLookups;
  if (!worstCase.empty()) {
    // Build a layout for all arguments used by the mux
    SmallVector<ZStruct::FieldInfo> members;
    for (auto argument : worstCase) {
      auto name = builder.getStringAttr(Zhlt::getTypeId(argument.first));
      auto arrayType = LayoutArrayType::get(ctx, argument.first, argument.second);
      members.push_back({name, arrayType});
    }
    std::string memberName = cb.getResultName(sw.getOut()).str();
    std::string typeName = "@Arguments$" + muxContext.getTypeName().str();
    auto hoistedArgumentsType = LayoutType::get(ctx, typeName, members);
    std::string name = "@arguments$" + muxContext.getTypeName().str();
    Value hoistedArguments = cb.addLayoutMember(sw->getLoc(), name, hoistedArgumentsType);

    // Build a table of lookups of all the hoisted arguments
    for (auto argument : worstCase) {
      hoistedArgumentLookups[argument.first] = {};
      SmallVector<Value>& argTable = hoistedArgumentLookups.back().second;
      auto name = builder.getStringAttr(Zhlt::getTypeId(argument.first));
      Value argArray = builder.create<ZStruct::LookupOp>(sw->getLoc(), hoistedArguments, name);
      for (size_t i = 0; i < argument.second; i++) {
        Value index = builder.create<arith::ConstantOp>(sw->getLoc(), builder.getIndexAttr(i));
        auto subscriptOp = builder.create<ZStruct::SubscriptOp>(sw->getLoc(), argArray, index);
        argTable.push_back(subscriptOp.getOut());
      }
    }
  }

  using ArgCounter = llvm::MapVector<Type, size_t>;
  std::function<void(Value, ArgCounter&)> addArgumentAliases;
  addArgumentAliases = [&](Value layout, ArgCounter& counter) {
    if (auto layoutType = dyn_cast<LayoutArrayType>(layout.getType())) {
      for (size_t i = 0; i < layoutType.getSize(); i++) {
        Value index = builder.create<arith::ConstantOp>(sw->getLoc(), builder.getIndexAttr(i));
        auto sublayout = builder.create<ZStruct::SubscriptOp>(sw->getLoc(), layout, index);
        addArgumentAliases(sublayout, counter);
      }
    }

    auto layoutType = dyn_cast<LayoutType>(layout.getType());
    if (!layoutType)
      return;

    switch (layoutType.getKind()) {
    case LayoutKind::Argument: {
      size_t index = counter[layoutType]++;
      Value hoistedLayout = hoistedArgumentLookups[layoutType][index];
      builder.create<ZStruct::AliasLayoutOp>(layout.getLoc(), hoistedLayout, layout);
    }
      return;
    case LayoutKind::Normal:
      for (auto field : layoutType.getFields()) {
        Value sublayout = builder.create<ZStruct::LookupOp>(sw->getLoc(), layout, field.name);
        addArgumentAliases(sublayout, counter);
      }
      break;
    case LayoutKind::MajorMux:
      // A major mux inside another mux is invalid
      sw.emitError() << "Major mux inside non-major mux";
      return;
    case LayoutKind::Mux:
      // This is already handled by the @arguments$name member on the parent
      break;
    }
  };

  // Argument layouts within each arm should alias the "hoisted" arguments. We
  // also need to add zero-initialized arguments as padding so that all arms
  // have the same number of arguments.
  for (size_t i = 0; i != size; ++i) {
    auto* armContext = armContexts[i].get();
    Type armLayoutType = armContext->getLayoutTypeSoFar();
    // If we don't need a layout for this arm, don't bother
    if (!armLayoutType && worstCase.empty())
      continue;

    auto armLoc = sw.getCases()[i].getLoc();
    // Recompute this arms requirement
    llvm::MapVector<Type, size_t> curCase;
    if (armLayoutType) {
      Zhlt::extractArguments(curCase, armLayoutType);
    }

    // Add zero-initialized "extra" members so all arms have the same arguments
    OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToEnd(&armRegions[i]->front());
    size_t count = 0;
    for (const auto& kvp : worstCase) {
      size_t needed = kvp.second - curCase[kvp.first];
      for (size_t j = 0; j < needed; j++) {
        std::string name = "@extra" + std::to_string(count++);
        Value extra = armContext->addLayoutMember(armLoc, name, kvp.first);
        buildZeroInitialize(extra);
      }
    }
  }

  Type armResultType = Zhlt::getLeastCommonSuper(armTypes);
  assert(armResultType);
  Type commonArmLayoutType = Zhlt::getLeastCommonSuper(armLayouts, /*isLayout=*/1);

  Value superLayout;
  Zhlt::ComponentOp commonCtor;
  if (auto layoutType = llvm::dyn_cast_if_present<ZStruct::LayoutType>(commonArmLayoutType)) {
    commonCtor = componentManager->lookupComponent(layoutType.getId());
  } else if (auto refType = llvm::dyn_cast_if_present<ZStruct::RefType>(commonArmLayoutType)) {
    commonCtor = componentManager->lookupComponent("NondetReg");
  } else if (auto arrayType =
                 llvm::dyn_cast_if_present<ZStruct::LayoutArrayType>(commonArmLayoutType)) {
    commonCtor = genArrayCtor(sw, arrayType.getElement(), arrayType.getSize());
  }
  if (commonCtor) {
    assert(commonCtor.getLayoutType() == commonArmLayoutType &&
           "Mux common type does not have expected layout type");
    superLayout = muxContext.addLayoutMember(sw.getLoc(), "@super", commonArmLayoutType);
  }
  LLVM_DEBUG({ llvm::dbgs() << "Switch arm common layout: " << commonArmLayoutType << "\n"; });
  SmallVector<Value> selectorValues;
  for (size_t i = 0; i != size; ++i) {
    auto indexOp = builder.create<Zll::ConstOp>(sw.getLoc(), i);
    selectorValues.push_back(builder.create<ZStruct::SubscriptOp>(sw.getLoc(), selector, indexOp));
  }

  for (size_t i = 0; i != size; ++i) {
    auto armLoc = sw.getCases()[i].getLoc();
    Region* armRegion = armRegions[i].get();
    ComponentBuilder* armContext = armContexts[i].get();

    OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToEnd(&armRegion->front());

    armContext->supplyLayout([&](StringAttr name, Type layoutType) -> Value {
      Value fullArmLayout = muxContext.addLayoutMember(armLoc, name, layoutType);
      // If the common super has layout, the common part must have the same
      // layout on each mux arm
      if (superLayout)
        genAliasLayout(armLoc, superLayout, fullArmLayout);
      // We hoist all argument layouts out of the major mux, and so the argument
      // sublayouts within each major mux arm need to alias the hoisted ones.
      if (!worstCase.empty()) {
        ArgCounter counter;
        addArgumentAliases(fullArmLayout, counter);
      }
      return fullArmLayout;
    });
    Value armValue = coerceTo(armContext->getValue(armLoc), armResultType);
    builder.create<ZStruct::YieldOp>(armLoc, armValue);
  }

  Type layoutType;
  muxContext.supplyLayout([&](StringAttr name, Type layoutTypeArg) -> Value {
    layoutType = layoutTypeArg;
    return addOrExpandLayoutMember(sw.getLoc(), cb, sw.getOut(), layoutType);
  });
  auto switchOp = builder.create<ZStruct::SwitchOp>(
      sw.getLoc(), armResultType, selectorValues, /*numArms=*/size);

  for (size_t i = 0; i != size; ++i) {
    switchOp.getArms()[i].takeBody(*armRegions[i]);
  }

  // If the SwitchOp has hoisted arguments, annotate it with the layout type
  if (layoutType) {
    switchOp->setAttr("layoutType", TypeAttr::get(layoutType));
  }

  if (superLayout) {
    // Layout present!  Reconstruct the value of the switch from the registers
    // to keep the degree of the validity polynomial low.
    layoutMapping[sw.getOut()] = superLayout;
  } else {
    // No layout present; return value type of switch result.
    valueMapping[sw.getOut()] = switchOp.getOut();
  }
}

void LoweringImpl::gen(RangeOp range, ComponentBuilder& cb) {
  PolynomialAttr startAttr = asFieldElement(range.getStart());
  PolynomialAttr endAttr = asFieldElement(range.getEnd());
  if (!startAttr) {
    range.emitError() << "the start of range must be a compile-time constant";
    startAttr = PolynomialAttr::get(ctx, {0});
  }
  uint64_t start = startAttr[0];
  if (!endAttr) {
    range.emitError() << "the end of range must be a compile-time constant";
    endAttr = PolynomialAttr::get(ctx, {start + 1});
  }
  uint64_t end = endAttr[0];
  if (start >= end) {
    range.emitError() << "the start of a range must be strictly less than its end";
    std::swap(start, end);
    if (start == end) {
      // generating at least one iteration prevents noisy downstream failures
      end++;
    }
  }
  SmallVector<Value> content;
  content.reserve(end - start);
  for (uint64_t i = start; i < end; i++) {
    content.push_back(builder.create<Zll::ConstOp>(range.getLoc(), i));
  }
  valueMapping[range.getOut()] = builder.create<ZStruct::ArrayOp>(range.getLoc(), content);
}

void LoweringImpl::gen(BackOp back, ComponentBuilder& cb) {
  PolynomialAttr distance = asFieldElement(back.getDistance());
  if (!distance) {
    back.emitError() << "the distance of a back expression must be a compile-time constant";
    return;
  }

  Value layout = asAliasableLayout(back.getTarget());
  if (!layout || !ZStruct::isLayoutType(layout.getType())) {
    back.emitError() << "back operation must apply to a subcomponent with a layout";
    throw MalformedIRException();
  }
  layoutMapping[back.getOut()] = layout;

  Value reconstructed = reconstructFromLayout(back.getLoc(), layout, distance[0]);
  if (!reconstructed) {
    back.emitError() << "Unable to reconstruct " << back.getTarget() << " from layout " << layout;
    throw MalformedIRException();
  }

  valueMapping[back.getOut()] = reconstructed;
}

void LoweringImpl::gen(ArrayOp array, ComponentBuilder& cb) {
  llvm::SmallVector<Value> elements;
  llvm::SmallVector<Value> layouts;
  for (Value elem : array.getElements()) {
    Value elemVal = asValue(elem);
    elements.push_back(elemVal);
    layouts.push_back(layoutMapping.lookup(elem));
  }
  if (elements.empty()) {
    array.emitError("An array must contain at least one element");
    return;
  }
  Type elemType = Zhlt::getLeastCommonSuper(ValueRange(elements).getTypes());
  for (Value& elem : elements) {
    elem = coerceTo(elem, elemType);
  }

  auto arrayOp = builder.create<ZStruct::ArrayOp>(array.getLoc(), elements);
  valueMapping[array.getOut()] = arrayOp;

  if (llvm::all_of(layouts, [](Value v) { return v; })) {
    Type layoutElemType =
        Zhlt::getLeastCommonSuper(ValueRange(layouts).getTypes(), /*isLayout=*/true);
    auto arrayLayout =
        cb.addLayoutMember(array.getLoc(),
                           array.getOut(),
                           builder.getType<LayoutArrayType>(layoutElemType, layouts.size()));

    for (auto [i, layout] : llvm::enumerate(layouts)) {
      auto constOp = builder.create<Zll::ConstOp>(array.getLoc(), i);
      auto subscriptOp = builder.create<ZStruct::SubscriptOp>(array.getLoc(), arrayLayout, constOp);
      layout = coerceTo(layout, layoutElemType);
      genAliasLayout(array.getLoc(), layout, subscriptOp);
    }

    layoutMapping[array.getOut()] = arrayLayout;
  }
}

void LoweringImpl::gen(DefinitionOp definition, ComponentBuilder& cb) {
  auto declaration = definition.getDeclaration().getDefiningOp<DeclarationOp>();
  Value def = asValue(definition.getDefinition());

  if (declaration.getTypeExp()) {
    StringAttr declaredTypeName = asTypeName(declaration.getTypeExp());
    if (!declaredTypeName) {
      definition.emitError() << "Unable to resolve type name of definition";
      return;
    }
    Type declaredType = componentManager->lookupComponent(declaredTypeName).getResultType();
    if (!Zhlt::isCoercibleTo(def.getType(), declaredType)) {
      auto diag = definition->emitError()
                  << "definition of type `" << getTypeId(def.getType())
                  << "` is not a subtype of the declared type `" << getTypeId(declaredType) << "`";
      diag.attachNote(declaration.getLoc()) << "declared here:";
    }
  }

  auto memberName = declaration.getMember();
  if (declaration.getIsPublic()) {
    cb.val()->addMember(memberName, def);
  } else {
    cb.val()->addPrivateMember(memberName, def);
  }
}

void LoweringImpl::gen(DeclarationOp declaration, ComponentBuilder& cb) {
  // Store declaration member for declarations that we might need to
  // reference before construction.
  if (!declaration.getTypeExp()) {
    // No type specified; can't reference until constructed.
    return;
  }

  StringAttr typeName = asTypeName(declaration.getTypeExp());
  if (!typeName) {
    declaration.emitError() << "Unable to resolve type name of declaration";
    return;
  }
  auto ctor = componentManager->lookupComponent(typeName);
  auto layoutType = ctor.getLayoutType();
  Value layout = cb.addLayoutMember(declaration.getLoc(), declaration.getMember(), layoutType);
  layoutMapping[declaration.getOut()] = layout;
}

void LoweringImpl::gen(ConstraintOp constraint, ComponentBuilder& cb) {
  Value lhs = asValue(constraint.getLhs());
  if (!Zhlt::isCoercibleTo(lhs.getType(), Zhlt::getValType(ctx))) {
    emitError(lhs.getLoc())
        << "a component of type `" << getTypeId(lhs.getType())
        << "` cannot be coerced to `Val`, but the left side of a constraint must be a `Val`";
    lhs = builder.create<Zll::ConstOp>(lhs.getLoc(), 0);
  }
  Value lhsVal = coerceTo(lhs, Zhlt::getValType(ctx));
  Value rhs = asValue(constraint.getRhs());
  if (!Zhlt::isCoercibleTo(rhs.getType(), Zhlt::getValType(ctx))) {
    emitError(rhs.getLoc())
        << "a component of type `" << getTypeId(rhs.getType())
        << "` cannot be coerced to `Val`, but the right side of a constraint must be a `Val`";
    rhs = builder.create<Zll::ConstOp>(rhs.getLoc(), 0);
  }
  Value rhsVal = coerceTo(rhs, Zhlt::getValType(ctx));
  auto diff = builder.create<Zll::SubOp>(constraint.getLoc(), lhsVal, rhsVal);
  builder.create<Zll::EqualZeroOp>(constraint.getLoc(), diff);
}

void LoweringImpl::gen(SuperOp superOp, ComponentBuilder& cb) {
  auto superVal = asValue(superOp.getValue());
  cb.val()->addMember("@super", superVal);
}

void LoweringImpl::gen(Region& region, ComponentBuilder& cb) {
  for (auto& block : region)
    gen(&block, cb);
}

void LoweringImpl::gen(Block* block, ComponentBuilder& cb) {
  for (auto& op : *block)
    gen(&op, cb);
}

void LoweringImpl::buildZeroInitialize(Value toInit) {
  // Get the layout type
  auto layout = mlir::cast<ZStruct::LayoutType>(toInit.getType());
  auto loc = toInit.getDefiningOp()->getLoc();
  auto zero = builder.create<Zll::ConstOp>(loc, 0);
  // We only zero initalize the first field, which is the 'count' field
  // This allows aliasing of other 'dont-care' fields.
  const auto& field = layout.getFields().front();
  auto layoutType = llvm::dyn_cast<ZStruct::LayoutType>(field.type);
  if (!layoutType || layoutType.getId() != "NondetReg") {
    toInit.getDefiningOp()->emitError() << "Argument types must be composed solely of NondetRegs";
    return;
  }
  auto valType = Zhlt::getValType(ctx);
  auto elem = builder.create<ZStruct::LookupOp>(loc, toInit, field.name);
  auto unwrap = builder.create<ZStruct::LookupOp>(loc, elem, "@super");
  builder.create<ZStruct::StoreOp>(loc, unwrap, zero);
  Value zeroDistance = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
  auto reload = builder.create<ZStruct::LoadOp>(loc, valType, unwrap, zeroDistance);
  builder.create<Zll::EqualZeroOp>(loc, reload);
}

Zhlt::ComponentOp LoweringImpl::genArrayCtor(Operation* op, Type elementType, size_t size) {
  MLIRContext* ctx = op->getContext();
  auto elemNameAttr = StringAttr::get(ctx, mangledTypeName(elementType));
  SmallVector<Attribute> typeArgs = {elemNameAttr, PolynomialAttr::get(ctx, size)};
  return componentManager->getComponent(op->getLoc(), "Array", typeArgs);
}

Value LoweringImpl::coerceTo(Value value, Type type) {
  return Zhlt::coerceTo(value, type, builder);
}

Value LoweringImpl::coerceToArray(Value value) {
  return Zhlt::coerceToArray(value, builder);
}

Value LoweringImpl::asValue(Value zhlVal) {
  Value zhltVal = valueMapping.lookup(zhlVal);
  if (zhltVal)
    return zhltVal;

  Value layout = layoutMapping.lookup(zhlVal);
  if (layout) {
    Value reconstructed = reconstructFromLayout(zhlVal.getLoc(), layout);
    if (!reconstructed) {
      emitError(zhlVal.getLoc()) << "Unable to reconstruct from layout";
      throw MalformedIRException();
    }
    return reconstructed;
  }

  if (auto typeName = typeNameMapping.lookup(zhlVal))
    emitError(zhlVal.getLoc()) << "attempted to use type name '" << typeName.strref()
                               << "' as a value";
  else
    emitError(zhlVal.getLoc()) << "Unable to resolve value " << zhlVal;
  throw MalformedIRException();
}

Attribute LoweringImpl::asConstant(Value arg) {
  // If it's a type name, return it as a StringAttr.
  if (auto attr = typeNameMapping.lookup(arg)) {
    return attr;
  }

  // Otherwise, try to evaluate it using the interpreter
  Value val = asValue(arg);

  Zll::Interpreter interp(ctx);
  auto attr = interp.evaluateConstant(val);
  if (!attr) {
    return {};
  }

  // If we're interpreting, we can get a wrapped component back.  Unwrap it.
  while (auto structAttr = dyn_cast_if_present<ZStruct::StructAttr>(attr)) {
    attr = structAttr.getFields().get("@wrapped");
    if (!attr || !llvm::isa<IntegerAttr, PolynomialAttr>(attr))
      attr = structAttr.getFields().get("@super");
  }

  return attr;
}

StringAttr LoweringImpl::asTypeName(Value v) {
  Attribute attr = asConstant(v);
  if (!attr) {
    emitError(v.getLoc()) << "component type must be a compile-time constant";
    throw MalformedIRException();
  }
  auto typeName = llvm::dyn_cast<StringAttr>(attr);
  if (!typeName) {
    emitError(v.getLoc()) << v << " does not evaluate to a type";
    throw MalformedIRException();
  }
  return typeName;
}

PolynomialAttr LoweringImpl::asFieldElement(Value v) {
  return llvm::dyn_cast_if_present<PolynomialAttr>(asConstant(v));
}

} // namespace

Zhlt::ComponentOp generateTypedComponent(OpBuilder& builder,
                                         ComponentManager* componentManager,
                                         Zhl::ComponentOp component,
                                         mlir::StringAttr mangledName,
                                         llvm::ArrayRef<mlir::Attribute> typeArgs) {
  auto regionAnchor = builder.create<Zhlt::MagicOp>(component.getLoc(), builder.getNoneType());
  Zhlt::ComponentOp converted;
  {
    LoweringImpl impl(builder, componentManager, regionAnchor);
    converted = impl.gen(component, mangledName, typeArgs);
  }
  regionAnchor.erase();
  return converted;
}

} // namespace zirgen::Typing
