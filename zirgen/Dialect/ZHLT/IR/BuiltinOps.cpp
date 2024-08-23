#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ZHLT/IR/BuiltinOps.cpp.inc"

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
      mlir::emitError(loc) << "Cannot coerce to " << valType << "\n";
    }
    args.push_back(coerceTo(arg, valType, builder));
  }
  builder.create<Zll::ExternOp>(loc, /*outTypes=*/TypeRange{}, args, "log", fmt);
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

  return ZStruct::LayoutArrayType::get(getContext(), manager->getValueType(elemType), elemCount[0]);
}

mlir::Value BuiltinArrayOp::buildConstruct(ComponentManager* manager,
                                           OpBuilder& builder,
                                           Location loc,
                                           ComponentTypeAttr name,
                                           ValueRange constructArgs,
                                           Value layout) {
  if (constructArgs.size() != 1) {
    mlir::emitError(loc) << "Array constructor must have exactly one argument";
    return {};
  }
  Type valueType = manager->getValueType(name);
  if (!isCoercibleTo(constructArgs[0].getType(), valueType)) {
    mlir::emitError(loc) << "Unable to convert " << constructArgs[0] << " to " << valueType;
    return {};
  }
  return coerceTo(constructArgs[0], valueType, builder);
}

} // namespace zirgen::Zhlt
