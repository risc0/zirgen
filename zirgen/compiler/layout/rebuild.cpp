// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/compiler/layout/rebuild.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

namespace zirgen {
namespace layout {

namespace {

using Zll::ValType;
using ZStruct::FieldInfo;
using ZStruct::LayoutArrayType;
using ZStruct::LayoutKind;
using ZStruct::LayoutType;
using ZStruct::RefType;

struct Builder {
  explicit Builder(Circuit& top);
  llvm::DenseMap<mlir::Type, mlir::Type> oldToNew;

private:
  mlir::Type build(mlir::Type);
  mlir::Type build(LayoutType&);
  mlir::Type build(LayoutArrayType&);
  mlir::Type buildStruct(LayoutType&);
  mlir::Type buildUnion(LayoutType&);
  void buildFields(std::vector<FieldInfo>&);
  Circuit& circuit;
  llvm::DenseSet<mlir::Type> targets;
};

Builder::Builder(Circuit& circuit) : circuit(circuit) {
  for (auto iter : circuit.sizes) {
    (void)build(iter.first);
  }
}

mlir::Type Builder::build(mlir::Type oldType) {
  auto found = oldToNew.find(oldType);
  if (found != oldToNew.end()) {
    return found->second;
  }
  mlir::Type newType = llvm::TypeSwitch<mlir::Type, mlir::Type>(oldType)
                           .Case<LayoutType, LayoutArrayType>([&](auto& at) { return build(at); })
                           .Default([&](auto& t) { return t; });
  oldToNew.insert({oldType, newType});
  return newType;
}

mlir::Type Builder::build(LayoutArrayType& at) {
  // Convert the element type, then wrap in a new array of same size.
  mlir::Type element = build(at.getElement());
  return LayoutArrayType::get(at.getContext(), element, at.getSize());
}

mlir::Type Builder::build(LayoutType& t) {
  switch (t.getKind()) {
  case LayoutKind::Normal:
  case LayoutKind::Argument:
    return buildStruct(t);
  case LayoutKind::Mux:
    return buildUnion(t);
  }
  assert(false && "unknown LayoutKind");
  return mlir::Type();
}

mlir::Type Builder::buildStruct(LayoutType& st) {
  auto found = circuit.structs.find(st);
  if (found == circuit.structs.end()) {
    return st;
  }
  Layout& sl = found->second;
  buildFields(sl.fields);
  auto kind = sl.original.getKind();
  auto out = LayoutType::get(st.getContext(), sl.id, sl.fields, kind);
  return out;
}

mlir::Type Builder::buildUnion(LayoutType& ut) {
  auto found = circuit.unions.find(ut);
  if (found == circuit.unions.end()) {
    return ut;
  }
  Layout& ul = found->second;
  buildFields(ul.fields);
  auto kind = LayoutKind::Mux;
  auto out = LayoutType::get(ut.getContext(), ul.id, ul.fields, kind);
  return out;
}

void Builder::buildFields(std::vector<FieldInfo>& fields) {
  // maintain invariant: super field must be first, if present
  for (size_t i = 1; i < fields.size(); ++i) {
    if ("@super" == fields[i].name) {
      auto fi = fields[i];
      fields.erase(fields.begin() + i);
      fields.insert(fields.begin(), fi);
      break;
    }
  }
  for (auto& fi : fields) {
    fi.type = build(fi.type);
  }
}

} // namespace

llvm::DenseMap<mlir::Type, mlir::Type> rebuild(Circuit& circuit) {
  return Builder(circuit).oldToNew;
}

} // namespace layout
} // namespace zirgen
