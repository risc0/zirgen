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

#include "zirgen/Dialect/Zll/IR/IR.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/Zll/IR/Dialect.cpp.inc"
#include "zirgen/Dialect/Zll/IR/Enums.cpp.inc"

namespace mlir {

class ZirgenInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation* call, Operation* callable, bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Region* dest,
                       Region* src,
                       bool wouldBeCloned,
                       IRMapping& valueMapping) const final {
    return true;
  }
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
  if (auto polyAttr = dyn_cast<PolynomialAttr>(value)) {
    // Promote to requested return type.
    SmallVector<uint64_t> elems = llvm::to_vector(polyAttr.asArrayRef());
    elems.resize(llvm::cast<ValType>(type).getFieldK());
    return builder.create<ConstOp>(loc, builder.getAttr<PolynomialAttr>(elems));
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
