// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "zirgen/Dialect/ZStruct/IR/Enums.h.inc"
#include "zirgen/Dialect/ZStruct/IR/TypeInterfaces.h.inc"

namespace zirgen::ZStruct {

enum class StorageKind : uint32_t {
  Normal = 0,
  Reserve = 1,
  Use = 2,
};

// member of struct or union
struct FieldInfo {
  mlir::StringAttr name;
  mlir::Type type;
  StorageKind storage = StorageKind::Normal;
};

} // namespace zirgen::ZStruct

namespace mlir {

using zirgen::ZStruct::FieldInfo;

/// Enable FieldInfo to be introspected for sub-elements.
template <> struct AttrTypeSubElementHandler<FieldInfo> {
  static void walk(FieldInfo param, AttrTypeImmediateSubElementWalker& walker) {
    walker.walk(param.name);
    walker.walk(param.type);
  }
  static FieldInfo replace(FieldInfo param,
                           AttrSubElementReplacements& attrRepls,
                           TypeSubElementReplacements& typeRepls) {
    return FieldInfo{.name = cast<StringAttr>(attrRepls.take_front(1)[0]),
                     .type = typeRepls.take_front(1)[0],
                     .storage = param.storage};
  }
};

} // namespace mlir

namespace zirgen::ZStruct {

bool operator==(const FieldInfo& a, const FieldInfo& b);

llvm::hash_code hash_value(const FieldInfo& fi);

/// Parse a list of field names and types within <>. E.g.:
/// <foo: i7, bar: i8>
mlir::ParseResult parseFields(mlir::AsmParser& p, llvm::SmallVectorImpl<FieldInfo>& parameters);

/// Print out a list of named fields surrounded by <>.
void printFields(mlir::AsmPrinter& p, llvm::ArrayRef<FieldInfo> fields);

/// Returns true if t is a valid witness structure type that only contains references to registers.
bool isLayoutType(mlir::Type t);

/// Returns true if t is a valid record type that contains only values and no register references.
bool isRecordType(mlir::Type t);

/// An implicit argument type which can supply buffers when passed to lookupNearestImplicitArg
template <typename ConcreteType>
struct BufferContextTypeTrait
    : public mlir::TypeTrait::TraitBase<ConcreteType, BufferContextTypeTrait> {};

} // namespace zirgen::ZStruct
