// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/Zll/IR/IR.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/IR/Dialect.cpp.inc"
#include "zirgen/Dialect/Zll/IR/Enums.cpp.inc"

namespace mlir {

class ZirgenInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool
  isLegalToInline(Operation* op, Region* dest, bool wouldBeCloned, IRMapping& valueMapping) const {
    // All ops in this dialect are inlinable
    return true;
  }
};

} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "zirgen/Dialect/Zll/IR/Types.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "zirgen/Dialect/Zll/IR/Attrs.cpp.inc"

using namespace mlir;

namespace zirgen::Zll {

void ZllDialect::initialize() {
  addInterfaces<ZirgenInlinerInterface>();

  addOperations<
#define GET_OP_LIST
#include "zirgen/Dialect/Zll/IR/Ops.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "zirgen/Dialect/Zll/IR/Types.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "zirgen/Dialect/Zll/IR/Attrs.cpp.inc"
      >();

  constexpr size_t kGoldilocksPrime = (0xffffffffffffffff << 32) + 1;

  fields = {{"BabyBear",
             FieldAttr::get(getContext(),
                            "BabyBear",
                            /*prime=*/kBabyBearP,
                            /*extDegree=*/kBabyBearExtSize,
                            /*polynomial=*/{kBabyBearP - 11, 0, 0, 0})},
            {"Goldilocks",
             FieldAttr::get(getContext(),
                            "Goldilocks",
                            /*prime=*/kGoldilocksPrime,
                            /*extDegree=*/2,
                            /*polynomial=*/{kGoldilocksPrime - 11, 0})}};
}

Operation*
ZllDialect::materializeConstant(OpBuilder& builder, Attribute value, Type type, Location loc) {
  if (auto polyAttr = value.dyn_cast<PolynomialAttr>()) {
    return builder.create<ConstOp>(loc, type, polyAttr);
  }
  return nullptr;
}

FieldAttr ZllDialect::getField(llvm::StringRef fieldName) {
  return fields.lookup(fieldName);
}

FieldAttr getDefaultField(MLIRContext* ctx) {
  return ctx->getOrLoadDialect<ZllDialect>()->getDefaultField();
}

FieldAttr getField(MLIRContext* ctx, llvm::StringRef fieldName) {
  return ctx->getOrLoadDialect<ZllDialect>()->getField(fieldName);
}

} // namespace zirgen::Zll
