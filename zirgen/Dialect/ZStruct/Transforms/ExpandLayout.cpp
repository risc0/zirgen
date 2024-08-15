// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZStruct/IR/Types.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/PassDetail.h"

#include <set>
#include <vector>

using namespace mlir;

namespace zirgen::ZStruct {
namespace {

struct LayoutExpander {
  LayoutExpander(ModuleOp mod, OpBuilder& builder) : mod(mod), builder(builder) {}

  Attribute expandInside(Location loc, Type ty, Attribute attr) {
    return TypeSwitch<Type, Attribute>(ty)
        .Case<LayoutType>(
            [&](auto ty) { return expandInside(loc, ty, llvm::cast<StructAttr>(attr)); })
        .Case<LayoutArrayType>(
            [&](auto ty) { return expandInside(loc, ty, llvm::cast<ArrayAttr>(attr)); })
        .Default([&](auto) {
          // Not a layout type; no change.
          return attr;
        });
  }

  Attribute expandInside(Location loc, LayoutType ty, StructAttr attr) {
    SmallVector<NamedAttribute> newFields;

    for (auto field : ty.getFields()) {
      newFields.emplace_back(field.name,
                             generateExpansion(loc, field.type, attr.getFields().get(field.name)));
    }
    return builder.getAttr<StructAttr>(builder.getAttr<DictionaryAttr>(newFields), ty);
  }

  Attribute expandInside(Location loc, LayoutArrayType ty, ArrayAttr attr) {
    SmallVector<Attribute> newElems;

    for (Attribute elem : attr) {
      newElems.emplace_back(generateExpansion(loc, ty.getElement(), elem));
    }

    return builder.getArrayAttr(newElems);
  }

  Attribute generateExpansion(Location loc, mlir::Type type, mlir::Attribute attr) {
    if (!llvm::isa<LayoutType, LayoutArrayType>(type))
      // Only expand structs and arrays
      return attr;

    if (layouts.contains(attr)) {
      return layouts.at(attr);
    }

    // If we only have a couple of substructures, don't bother expanding this one.
    size_t numSubattrs = 0;
    constexpr size_t kMaxSubattrs = 3;
    if (!attr.walk([&](StructAttr structAttr) {
               if (++numSubattrs > kMaxSubattrs)
                 return WalkResult::interrupt();
               return WalkResult::advance();
             })
             .wasInterrupted()) {
      return attr;
    }

    auto newSymName = builder.getStringAttr(SymbolTable::generateSymbolName<64>(
        "layout$",
        [&](StringRef candidate) -> bool { return mod.lookupSymbol(candidate) ? true : false; },
        uniquingCounter));
    auto newSymRef = builder.getAttr<SymbolRefAttr>(newSymName);

    builder.create<GlobalConstOp>(loc, newSymName, type, expandInside(loc, type, attr));
    bool didInsert = layouts.try_emplace(attr, newSymRef).second;
    assert(didInsert);

    return newSymRef;
  }

  ModuleOp mod;
  OpBuilder& builder;
  DenseMap<Attribute, /*symbol=*/SymbolRefAttr> layouts;
  unsigned uniquingCounter = 0;
};

struct ExpandLayoutPass : public ExpandLayoutBase<ExpandLayoutPass> {
  void runOnOperation() override {
    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(getOperation().getBody());

    LayoutExpander expander(getOperation(), builder);

    getOperation().walk([&](GlobalConstOp op) {
      op.setConstantAttr(expander.expandInside(op.getLoc(), op.getType(), op.getConstant()));
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createExpandLayoutPass() {
  return std::make_unique<ExpandLayoutPass>();
}

} // namespace zirgen::ZStruct
