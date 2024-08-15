// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZHLT/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "zirgen/Dialect/ZHLT/IR/Attrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/ZHLT/IR/Types.cpp.inc"

using namespace mlir;

namespace zirgen::Zhlt {

class ZhltInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation* call, Operation* callable, bool wouldBeCloned) const final {
    if (isa<ComponentOp>(callable))
      // Only inline aspects that have been generated for components, not for the original
      // components.
      return false;
    return isa<FunctionOpInterface>(callable);
  }

  bool isLegalToInline(Operation* op,
                       Region* dest,
                       bool wouldBeCloned,
                       IRMapping& valueMapping) const final {
    return true;
  }

  bool isLegalToInline(Region* dest,
                       Region* src,
                       bool wouldBeCloned,
                       IRMapping& valueMapping) const final {
    return true;
  }

  void handleTerminator(Operation* op, ValueRange valuesToRepl) const final {
    // ReturnOp is the only teriminator in the dialect
    auto returnOp = cast<Zhlt::ReturnOp>(op);

    // Replace the values directly with the return operand.
    if (!valuesToRepl.empty()) {
      assert(valuesToRepl.size() == 1);
      valuesToRepl[0].replaceAllUsesWith(returnOp.getValue());
    }
  }
};

void ZhltDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zirgen/Dialect/ZHLT/IR/Ops.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "zirgen/Dialect/ZHLT/IR/ComponentOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "zirgen/Dialect/ZHLT/IR/Attrs.cpp.inc"
      >();

  addInterface<ZhltInlinerInterface>();

  addTypes<
#define GET_TYPEDEF_LIST
#include "zirgen/Dialect/ZHLT/IR/Types.cpp.inc"
      >();
}

Operation*
ZhltDialect::materializeConstant(OpBuilder& builder, Attribute value, Type type, Location loc) {
  return TypeSwitch<Attribute, Operation*>(value)
      .Case<IntegerAttr>([&](auto intAttr) {
        uint64_t numericalValue = intAttr.getValue().getSExtValue();
        return TypeSwitch<Type, Operation*>(type)
            .Case<IndexType>([&](auto t) {
              // Index must be signless
              auto signlessAttr = IntegerAttr::get(t, numericalValue);
              return builder.create<arith::ConstantOp>(loc, t, signlessAttr);
            })
            .Default([&](auto) {
              auto polyAttr = PolynomialAttr::get(intAttr.getContext(), {numericalValue});
              return builder.create<Zll::ConstOp>(loc, polyAttr);
            });
      })
      .Default([&](auto) -> Operation* { return nullptr; });
}

// Constant names for generated constants.
std::string getTapsConstName() {
  return "tapList";
}

} // namespace zirgen::Zhlt
