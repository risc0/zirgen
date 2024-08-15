// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/compiler/layout/collect.h"
#include "mlir/IR/Types.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "llvm/ADT/TypeSwitch.h"

#include <queue>

namespace zirgen {
namespace layout {

namespace {

using Zll::ValType;
using ZStruct::FieldInfo;
using ZStruct::LayoutKind;
using ZStruct::LayoutType;

} // namespace

Layout::Layout(LayoutType t) : original(t), id(t.getId().str()) {
  fields.assign(t.getFields().begin(), t.getFields().end());
}

Circuit::Circuit(mlir::ModuleOp mod) {
  mod.walk([&](Zhlt::ComponentOp comp) { (void)visit(comp.getLayoutType()); });
}

unsigned Circuit::visit(mlir::Type t) {
  if (!t)
    return 0;
  unsigned sz = llvm::TypeSwitch<mlir::Type, unsigned>(t)
                    .Case<RefType>([&](RefType x) { return visit(x); })
                    .Case<LayoutType>([&](LayoutType x) { return visit(x); })
                    .Case<LayoutArrayType>([&](LayoutArrayType x) { return visit(x); });
  sizes[t] = sz;
  return sz;
}

unsigned Circuit::visit(RefType t) {
  return 1;
}

unsigned Circuit::visit(LayoutType t) {
  switch (t.getKind()) {
  case LayoutKind::Normal:
  case LayoutKind::Argument:
    return visitStruct(t);
  case LayoutKind::Mux:
    return visitUnion(t);
  }
  assert(false && "unknown LayoutKind");
  return 0;
}

unsigned Circuit::visitUnion(LayoutType ut) {
  if (unions.find(ut) == unions.end()) {
    unsigned usz = 0;
    unions.insert({ut, Layout(ut)});
    for (auto& field : ut.getFields()) {
      unsigned elsz = visit(field.type);
      usz = std::max(elsz, usz);
    }
    unionsInDfsPostorder.push_back(ut);
    return usz;
  } else {
    return sizes[ut];
  }
}

unsigned Circuit::visitStruct(LayoutType st) {
  if (structs.find(st) == structs.end()) {
    unsigned elsz = 0;
    structs.insert({st, Layout(st)});
    for (auto& field : st.getFields()) {
      elsz += visit(field.type);
    }
    return elsz;
  } else {
    return sizes[st];
  }
}

unsigned Circuit::visit(LayoutArrayType t) {
  unsigned elsz = visit(t.getElement());
  return elsz * t.getSize();
}

} // namespace layout
} // namespace zirgen
