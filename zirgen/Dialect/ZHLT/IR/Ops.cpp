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
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "zirgen/Dialect/ZHLT/IR/Ops.cpp.inc"

namespace zirgen::Zhlt {

using namespace mlir;

LogicalResult GetLayoutOp::inferReturnTypes(MLIRContext* ctx,
                                            std::optional<Location> loc,
                                            Adaptor adaptor,
                                            SmallVectorImpl<Type>& out) {
  Type valueType = adaptor.getIn().getType();
  Type layoutType = ZStruct::getLayoutType(valueType);
  if (!layoutType)
    return mlir::emitError(*loc) << getTypeId(valueType) << " has no layout";
  out.push_back(layoutType);
  return success();
}

namespace {

Value resolveLayout(Value value) {
  Operation* op = value.getDefiningOp();
  if (!op)
    return Value();

  return TypeSwitch<Operation*, Value>(op)
      .Case<ZStruct::PackOp>([&](auto pack) {
        if (pack.getLayout())
          return pack.getLayout();

        // The pack has no layout, look to its super
        ArrayRef<FieldInfo> fields = pack.getType().getFields();
        if (fields.size() == pack.getMembers().size()) {
          for (size_t i = 0; i < fields.size(); i++) {
            if (fields[i].name == "@super") {
              return resolveLayout(pack.getMembers()[i]);
            }
          }
        }
        return Value();
      })
      .Case<Zhlt::BackOp>([](auto back) { return back.getLayout(); })
      .Default([](auto op) { return Value(); });
}

} // namespace

OpFoldResult GetLayoutOp::fold(FoldAdaptor adaptor) {
  Value foldedValue = resolveLayout(getIn());
  if (foldedValue)
    return foldedValue;

  if (auto structAttr = dyn_cast_if_present<ZStruct::StructAttr>(adaptor.getIn())) {
    return structAttr.getFields().get("@layout");
  }

  return {};
}

mlir::LogicalResult MagicOp::verify() {
  return emitError() << "a MagicOp is never valid";
}

mlir::LogicalResult BackOp::verify() {
  if (getLayout()) {
    auto layoutType = getLayout().getType();
    if (layoutType && !ZStruct::isLayoutType(layoutType)) {
      return emitError() << layoutType << " must be a layout type";
    }
  }

  auto outType = getType();
  if (!ZStruct::isValidValueType(outType)) {
    return emitError() << outType << " must be a value type";
  }
  return success();
}

} // namespace zirgen::Zhlt
