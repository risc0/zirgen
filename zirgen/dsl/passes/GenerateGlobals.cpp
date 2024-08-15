// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/ZHLT/IR/TypeUtils.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/dsl/passes/PassDetail.h"

using namespace mlir;

namespace zirgen {
namespace dsl {

namespace {

struct StringAttrCmp {
  bool operator()(const StringAttr& a, const StringAttr& b) const {
    return a.strref() < b.strref();
  }
};

template <typename T> using StringAttrMap = std::map<StringAttr, T, StringAttrCmp>;

struct GenerateGlobalsPass : public GenerateGlobalsBase<GenerateGlobalsPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    ModuleOp mod = getOperation();

    // compute layout type for all globals used by the program
    StringAttrMap<StringAttrMap<ZStruct::LayoutType>> globalLayouts;
    mod.walk([&](Zhlt::GetGlobalLayoutOp op) {
      globalLayouts[op.getBufferAttr()].insert({op.getNameAttr(), op.getType()});
    });

    for (auto buffer : globalLayouts) {
      SmallVector<FieldInfo> members;
      for (auto member : buffer.second) {
        members.push_back({member.first, member.second});
      }

      std::string name = "@" + buffer.first.str();
      auto layoutType = ZStruct::LayoutType::get(ctx, name, members);
      createBufferPseudoComponent(name, layoutType);
    }
  }

  void createBufferPseudoComponent(StringRef name, ZStruct::LayoutType layoutType) {
    MLIRContext* ctx = &getContext();
    ModuleOp mod = getOperation();
    OpBuilder builder(ctx);
    Location loc = builder.getUnknownLoc();

    builder.setInsertionPointToStart(mod.getBody());
    auto ctor = builder.create<Zhlt::ComponentOp>(loc,
                                                  name,
                                                  /*valueType=*/Zhlt::getComponentType(ctx),
                                                  /*constructArgs=*/ValueRange{},
                                                  layoutType);
    Block& block = ctor.getBody().emplaceBlock();
    block.addArgument(layoutType, loc);
    builder.setInsertionPointToStart(&block);
    auto packed =
        builder.create<ZStruct::PackOp>(loc, Zhlt::getComponentType(ctx), SmallVector<Value>());
    builder.create<Zhlt::ReturnOp>(loc, packed);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createGenerateGlobalsPass() {
  return std::make_unique<GenerateGlobalsPass>();
}

} // namespace dsl
} // namespace zirgen
