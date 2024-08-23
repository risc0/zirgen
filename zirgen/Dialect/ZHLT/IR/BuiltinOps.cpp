#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ZHLT/IR/BuiltinOps.cpp.inc"

#define DEBUG_TYPE "zhlt"

using namespace mlir;

namespace zirgen::Zhlt {

Value BuiltinLogOp::buildConstruct(ComponentManager* manager,
                                   OpBuilder& builder,
                                   Location loc,
                                   ComponentTypeAttr name,
                                   ValueRange constructArgs,
                                   Value layout) {
  if (constructArgs.size() < 1) {
    mlir::emitError(loc) << "Log must have at least one argument";
    return {};
  }
  Zll::Interpreter interp(getContext());
  auto fmt = llvm::dyn_cast_if_present<StringAttr>(interp.evaluateConstant(constructArgs[0]));
  if (!fmt) {
    mlir::emitError(loc) << "Unable to evaluate log format to a string";
    return {};
  }
  SmallVector<Value> args;
  auto valType = getValType(getContext());
  for (auto arg : constructArgs.drop_front(1)) {
    if (!isCoercibleTo(arg.getType(), valType)) {
      mlir::emitError(loc) << "Cannot coerce " << arg << " to " << valType << "\n";
    } else
      args.push_back(coerceTo(arg, valType, builder));
  }
  builder.create<Zll::ExternOp>(loc, /*outTypes=*/TypeRange{}, args, "Log", fmt);
  return builder.create<ZStruct::PackOp>(loc, manager->getValueType(name), ValueRange{});
}

LogicalResult
BuiltinArrayOp::requireComponent(ComponentManager* manager, Location loc, ComponentTypeAttr name) {
  if (name.getTypeArgs().size() != 2) {
    return mlir::emitError(loc) << "Array must have two type arguments, but got " << name;
  }

  auto elemType = llvm::dyn_cast<ComponentTypeAttr>(name.getTypeArgs()[0]);
  if (!elemType) {
    return mlir::emitError(loc) << "Unknown type of array element";
  }
  if (failed(manager->requireComponent(loc, elemType))) {
    return mlir::emitError(loc) << "Can't instantiate element type";
  }
  auto elemCount = llvm::dyn_cast<PolynomialAttr>(name.getTypeArgs()[1]);
  if (!elemCount) {
    return mlir::emitError(loc) << "Unable to determine array length";
  }
  if (elemCount[0] < 1) {
    return mlir::emitError(loc) << "Array must have at least one element";
  }

  return success();
}

mlir::Type BuiltinArrayOp::getValueType(ComponentManager* manager, ComponentTypeAttr name) {
  auto elemType = llvm::cast<ComponentTypeAttr>(name.getTypeArgs()[0]);
  auto elemCount = llvm::cast<PolynomialAttr>(name.getTypeArgs()[1]);
  return ZStruct::ArrayType::get(getContext(), manager->getValueType(elemType), elemCount[0]);
}

mlir::Type BuiltinArrayOp::getLayoutType(ComponentManager* manager, ComponentTypeAttr name) {
  auto elemType = llvm::cast<ComponentTypeAttr>(name.getTypeArgs()[0]);
  auto elemCount = llvm::cast<PolynomialAttr>(name.getTypeArgs()[1]);
  auto elemLayout = manager->getLayoutType(elemType);
  if (elemLayout)
    return ZStruct::LayoutArrayType::get(getContext(), elemLayout, elemCount[0]);
  else
    return {};
}

mlir::Value BuiltinArrayOp::buildConstruct(ComponentManager* manager,
                                           OpBuilder& builder,
                                           Location loc,
                                           ComponentTypeAttr name,
                                           ValueRange constructArgs,
                                           Value layout) {
  if (constructArgs.size() != 1 && name.getName() == "Array") {
    mlir::emitError(loc) << "Array constructor must have exactly one argument";
    return {};
  }

  ZStruct::ArrayType valueType = getCoercibleArrayType(manager->getValueType(name));
  assert(valueType &&
         "Shouldn't have been able to instantiate this component without a valid value type");

  if (constructArgs.size() == 1) {
    if (!isCoercibleTo(constructArgs[0].getType(), valueType)) {
      mlir::emitError(loc) << "Unable to convert " << constructArgs[0] << " to " << valueType;
      return {};
    }
    return coerceTo(constructArgs[0], valueType, builder);
  }

  assert((name.getName() == "ConcatArray") && "`Array` component cannot be used to concatinate");

  SmallVector<Value> concatValues;
  for (auto arg : constructArgs) {
    auto argType = getCoercibleArrayType(arg.getType());
    if (!argType) {
      mlir::emitError(loc) << "Unable to use " << argType << " as an array";
      return {};
    }
    auto arrayArg = coerceToArray(arg, builder);
    if (!isCoercibleTo(argType.getElement(), valueType.getElement())) {
      mlir::emitError(loc) << "Unable to convert " << argType << " elements to " << valueType
                           << "\n";
      return {};
    }
    for (size_t i = 0; i != argType.getSize(); ++i) {
      Value index = builder.create<Zll::ConstOp>(loc, i);
      auto elem = builder.create<ZStruct::SubscriptOp>(loc, arrayArg, index);

      concatValues.push_back(coerceTo(elem, valueType.getElement(), builder));
    }
  }

  return builder.create<ZStruct::ArrayOp>(loc, valueType, concatValues);
}

void BuiltinArrayOp::inferType(ComponentManager* manager,
                               ComponentTypeAttr& name,
                               ValueRange constructArgs) {
  LLVM_DEBUG({ llvm::dbgs() << "Attempting to infer type of " << name << "\n"; });
  if (!name.getTypeArgs().empty()) {
    // Already specialized
    return;
  }

  assert(name.getName() == "Array" || name.getName() == "ConcatArray");
  if (name.getName() == "Array" && constructArgs.size() != 1) {
    LLVM_DEBUG({ llvm::dbgs() << "Array does not have a single constructor arg\n"; });

    // To concatinate more than one array together they must be invoked as `ConcatArray`
    return;
  }
  if (constructArgs.empty()) {
    LLVM_DEBUG({ llvm::dbgs() << "Array can not be empty."; });

    return;
  }
  size_t numElem = 0;
  SmallVector<Type> elemTypes;
  for (auto constructArg : constructArgs) {
    auto arrType = getCoercibleArrayType(constructArg.getType());
    if (!arrType) {
      LLVM_DEBUG(
          { llvm::dbgs() << "Non-array type supplied: " << constructArg.getType() << "\n"; });

      // Non-array argument
      return;
    }

    numElem += arrType.getSize();
    elemTypes.push_back(arrType.getElement());
  }

  ComponentTypeAttr elemType = manager->getNameForType(getLeastCommonSuper(elemTypes));
  if (!elemType) {
    LLVM_DEBUG({ llvm::dbgs() << "No common element type\n"; });
    return;
  }
  name = ComponentTypeAttr::get(
      getContext(), name.getName(), {elemType, PolynomialAttr::get(getContext(), numElem)});
  LLVM_DEBUG({ llvm::dbgs() << "Inferred " << name << "\n"; });
}

} // namespace zirgen::Zhlt
