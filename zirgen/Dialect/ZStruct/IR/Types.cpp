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

#include "llvm/ADT/TypeSwitch.h"

#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/Zll/IR/Codegen.h"

#include "zirgen/Dialect/ZStruct/IR/Enums.cpp.inc"
#include "zirgen/Dialect/ZStruct/IR/TypeInterfaces.cpp.inc"

using namespace mlir;
using namespace zirgen::codegen;

namespace zirgen::ZStruct {

bool operator==(const FieldInfo& a, const FieldInfo& b) {
  return a.name == b.name && a.type == b.type;
}

llvm::hash_code hash_value(const FieldInfo& fi) {
  return llvm::hash_combine(fi.name, fi.type);
}

mlir::ParseResult parseFields(mlir::AsmParser& p, llvm::SmallVectorImpl<FieldInfo>& parameters) {
  return p.parseCommaSeparatedList(
      mlir::AsmParser::Delimiter::LessGreater, [&]() -> mlir::ParseResult {
        std::string name;
        if (p.parseKeywordOrString(&name) || p.parseColon()) {
          return mlir::failure();
        }
        StorageKind storage = StorageKind::Normal;
        if (succeeded(p.parseOptionalKeyword("reserve"))) {
          storage = StorageKind::Reserve;
        } else if (succeeded(p.parseOptionalKeyword("use"))) {
          storage = StorageKind::Use;
        }
        mlir::Type type;
        if (p.parseType(type)) {
          return mlir::failure();
        }
        parameters.push_back(FieldInfo{mlir::StringAttr::get(p.getContext(), name), type, storage});
        return mlir::success();
      });
}

void printFields(mlir::AsmPrinter& p, llvm::ArrayRef<FieldInfo> fields) {
  p << '<';
  llvm::interleaveComma(fields, p, [&](const FieldInfo& field) {
    p.printKeywordOrString(field.name.getValue());
    p << ": ";
    switch (field.storage) {
    case StorageKind::Normal:
      break;
    case StorageKind::Reserve:
      p << "reserve ";
      break;
    case StorageKind::Use:
      p << "use ";
      break;
    }
    p << field.type;
  });
  p << ">";
}

mlir::Type StructType::parse(mlir::AsmParser& p) {
  if (p.parseLess()) {
    return Type();
  }
  std::string id;
  if (p.parseKeywordOrString(&id) || p.parseComma()) {
    return Type();
  }
  llvm::SmallVector<FieldInfo, 4> parameters;
  if (parseFields(p, parameters)) {
    return Type();
  }
  if (p.parseGreater()) {
    return Type();
  }
  return StructType::get(p.getContext(), id, parameters);
}

mlir::Type LayoutType::parse(mlir::AsmParser& p) {
  if (p.parseLess()) {
    return Type();
  }
  std::string id;
  if (p.parseKeywordOrString(&id)) {
    return Type();
  }
  LayoutKind kind = LayoutKind::Normal;
  StringRef strKind;
  if (succeeded(p.parseOptionalKeyword(&strKind, {"mux", "argument", "major"}))) {
    kind = symbolizeEnum<LayoutKind>(strKind).value_or(LayoutKind::Normal);
  }
  if (p.parseComma())
    return Type();
  llvm::SmallVector<FieldInfo, 4> parameters;
  if (parseFields(p, parameters)) {
    return Type();
  }
  if (p.parseGreater()) {
    return Type();
  }
  return LayoutType::get(p.getContext(), id, parameters, kind);
}

void StructType::print(mlir::AsmPrinter& p) const {
  p << "<";
  p.printKeywordOrString(getId());
  p << ", ";
  printFields(p, getFields());
  p << ">";
}

void LayoutType::print(mlir::AsmPrinter& p) const {
  p << "<";
  p.printKeywordOrString(getId());
  if (getKind() != LayoutKind::Normal) {
    p << " " << stringifyLayoutKind(getKind());
  }
  p << ", ";
  printFields(p, getFields());
  p << ">";
}

mlir::Type UnionType::parse(mlir::AsmParser& p) {
  if (p.parseLess()) {
    return Type();
  }
  std::string id;
  if (p.parseKeywordOrString(&id) || p.parseComma()) {
    return Type();
  }
  llvm::SmallVector<FieldInfo, 4> parameters;
  if (parseFields(p, parameters)) {
    return Type();
  }
  if (p.parseGreater()) {
    return Type();
  }
  return get(p.getContext(), id, parameters);
}

void UnionType::print(mlir::AsmPrinter& p) const {
  p << "<";
  p.printKeywordOrString(getId());
  p << ", ";
  printFields(p, getFields());
  p << ">";
}

bool isLayoutType(mlir::Type container) {
  // A component without any registers may have a null layout type
  if (!container)
    return true;

  auto walked = container.walk<mlir::WalkOrder::PreOrder>([&](mlir::Type t) {
    return llvm::TypeSwitch<mlir::Type, mlir::WalkResult>(t)
        .Case<LayoutType>([&](auto ty) { return mlir::WalkResult::advance(); })
        .Case<LayoutArrayType>([&](auto) { return mlir::WalkResult::advance(); })
        .Case<RefType>([&](auto) {
          // Allow refs to contain a ValType, even though ValType is
          // otherwise disallowed in a witness structure.
          return mlir::WalkResult::skip();
        })
        .Default([&](auto) { return mlir::WalkResult::interrupt(); });
  });
  return !walked.wasInterrupted();
}

bool isRecordType(mlir::Type container) {
  auto walked = container.walk<mlir::WalkOrder::PreOrder>([&](mlir::Type t) {
    return llvm::TypeSwitch<mlir::Type, mlir::WalkResult>(t)
        .Case<Zll::ValType, UnionType, ArrayType>([&](auto) { return mlir::WalkResult::advance(); })
        .Case<StructType>([&](StructType t) {
          if (t.getId() == "NondetReg") {
            // Regs can contain a "@wrapped" referring to a register, which is otherwise disallowed.
            return mlir::WalkResult::skip();
          } else {
            return mlir::WalkResult::advance();
          }
        })
        .Default([&](auto) { return mlir::WalkResult::interrupt(); });
  });
  return !walked.wasInterrupted();
}

CodegenIdent<IdentKind::Type> StructType::getTypeName(zirgen::codegen::CodegenEmitter& cg) const {
  return cg.getStringAttr((getId() + "Struct").str());
}

mlir::LogicalResult StructType::emitLiteral(zirgen::codegen::CodegenEmitter& cg,
                                            mlir::Attribute value) const {
  llvm::SmallVector<CodegenIdent<IdentKind::Field>> names;
  llvm::SmallVector<CodegenValue> values;

  auto dictVal = llvm::cast<StructAttr>(value).getFields();

  for (auto field : getFields()) {
    if (dictVal.contains(field.name)) {
      names.push_back(field.name);
      values.push_back(CodegenValue(field.type, dictVal.get(field.name)));
    }
  }

  cg.emitStructConstruct(*this, names, values);

  return mlir::success();
}

void StructType::emitTypeDefinition(zirgen::codegen::CodegenEmitter& cg) const {
  auto names = llvm::to_vector_of<CodegenIdent<IdentKind::Field>>(
      llvm::map_range(getFields(), [&](auto field) { return field.name; }));
  auto types =
      llvm::to_vector(llvm::map_range(getFields(), [&](auto field) { return field.type; }));
  cg.emitStructDef(*this, names, types);
}

CodegenIdent<IdentKind::Type> ArrayType::getTypeName(zirgen::codegen::CodegenEmitter& cg) const {
  auto elemName = cg.getTypeName(getElement());
  return cg.getStringAttr((elemName.strref() + std::to_string(getSize()) + "Array").str());
}

CodegenIdent<IdentKind::Type>
LayoutArrayType::getTypeName(zirgen::codegen::CodegenEmitter& cg) const {
  auto elemName = cg.getTypeName(getElement());
  return cg.getStringAttr((elemName.strref() + std::to_string(getSize()) + "LayoutArray").str());
}

CodegenIdent<IdentKind::Type> LayoutType::getTypeName(zirgen::codegen::CodegenEmitter& cg) const {
  if (cg.getOpts().zkpLayoutCompat) {
    return cg.getStringAttr(getId());
  }
  return cg.getStringAttr((getId() + "Layout").str());
}

mlir::LogicalResult LayoutType::emitLiteral(zirgen::codegen::CodegenEmitter& cg,
                                            mlir::Attribute value) const {
  if (auto symRef = llvm::dyn_cast<SymbolRefAttr>(value)) {
    // Reference by symbolname to to a layout defined with a GlobalConstOp
    cg << CodegenIdent<IdentKind::Const>(symRef.getLeafReference());
    return mlir::success();
  }

  if (auto boundLayout = llvm::dyn_cast<BoundLayoutAttr>(value)) {
    cg.emitInvokeMacro(cg.getStringAttr("bind_layout"),
                       {CodegenIdent<IdentKind::Const>(boundLayout.getBuffer()),
                        CodegenValue(*this, boundLayout.getLayout())});
    return mlir::success();
  }

  llvm::SmallVector<CodegenIdent<IdentKind::Field>> names;
  llvm::SmallVector<CodegenValue> values;

  auto dictVal = llvm::cast<StructAttr>(value).getFields();

  for (auto field : getFields()) {
    if (dictVal.contains(field.name)) {
      names.push_back(field.name);
      values.push_back(CodegenValue(field.type, dictVal.get(field.name)));
    }
  }

  cg.emitStructConstruct(*this, names, values);

  return mlir::success();
}

void LayoutType::emitTypeDefinition(zirgen::codegen::CodegenEmitter& cg) const {
  auto names = llvm::to_vector_of<CodegenIdent<IdentKind::Field>>(
      llvm::map_range(getFields(), [&](auto field) { return field.name; }));
  auto types =
      llvm::to_vector(llvm::map_range(getFields(), [&](auto field) { return field.type; }));
  cg.emitLayoutDef(*this, names, types);
}

LogicalResult LayoutType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 StringRef id,
                                 ArrayRef<FieldInfo> fields,
                                 LayoutKind kind) {
  for (size_t index = 1; index < fields.size(); ++index) {
    if (fields[index].name.str() == "@super") {
      return emitError() << "field \"@super\" must be first, when present";
    }
  }
  for (auto field : fields) {
    if (!llvm::isa<LayoutType, RefType, LayoutArrayType>(field.type))
      return emitError() << "invalid field type in layout: " << field.type;
  }

  return success();
}

void ArrayType::emitTypeDefinition(zirgen::codegen::CodegenEmitter& cg) const {
  cg.emitArrayDef(*this, getElement(), getSize());
}

mlir::LogicalResult ArrayType::emitLiteral(zirgen::codegen::CodegenEmitter& cg,
                                           mlir::Attribute value) const {
  auto arrayVal = llvm::cast<ArrayAttr>(value);
  auto elemType = getElement();
  auto values = llvm::to_vector_of<CodegenValue>(
      llvm::map_range(arrayVal, [&](auto elem) { return CodegenValue(elemType, elem); }));
  cg.emitArrayConstruct(*this, getElement(), values);
  return success();
}

void LayoutArrayType::emitTypeDefinition(zirgen::codegen::CodegenEmitter& cg) const {
  cg.emitArrayDef(*this, getElement(), getSize());
}

mlir::LogicalResult LayoutArrayType::emitLiteral(zirgen::codegen::CodegenEmitter& cg,
                                                 mlir::Attribute value) const {
  if (auto symRef = llvm::dyn_cast<SymbolRefAttr>(value)) {
    // Reference by symbolname to to a layout defined with a GlobalConstOp
    cg << CodegenIdent<IdentKind::Const>(symRef.getLeafReference());
    return mlir::success();
  }
  auto arrayVal = llvm::cast<ArrayAttr>(value);
  auto elemType = getElement();
  auto values = llvm::to_vector_of<CodegenValue>(
      llvm::map_range(arrayVal, [&](auto elem) { return CodegenValue(elemType, elem); }));
  cg.emitArrayConstruct(*this, getElement(), values);
  return success();
}

mlir::Type getLayoutArgumentType(mlir::FunctionType funcType) {
  return funcType.getInputs().back();
}

llvm::ArrayRef<mlir::Type> getRecordArgumentTypes(mlir::FunctionType funcType) {
  return funcType.getInputs().drop_back(1);
}

CodegenIdent<IdentKind::Type> RefType::getTypeName(zirgen::codegen::CodegenEmitter& cg) const {
  if (cg.getOpts().zkpLayoutCompat) {
    size_t k = getElement().getFieldK();
    if (k == 1) {
      return cg.getStringAttr(getBuffer().str() + "Reg");
    } else {
      return cg.getStringAttr(getBuffer().str() + std::to_string(k) + "Reg");
    }
  } else {
    return cg.getStringAttr("Reg");
  }
}

mlir::LogicalResult RefType::emitLiteral(zirgen::codegen::CodegenEmitter& cg,
                                         mlir::Attribute value) const {
  auto refVal = llvm::cast<RefAttr>(value);
  // TODO: Add make_ref macro to risc0-zkp::layout library and remove this special case.
  if (cg.getOpts().zkpLayoutCompat) {
    if (cg.getLanguageKind() == LanguageKind::Rust) {
      auto refVal = llvm::cast<RefAttr>(value);
      cg << "&" << cg.getTypeName(*this) << "{offset:" << refVal.getIndex() << "}";
      return mlir::success();
    } else {
      auto refVal = llvm::cast<RefAttr>(value);
      cg << refVal.getIndex();
      return mlir::success();
    }
  }
  if (!cg.getOpts().zkpLayoutCompat) {
    assert(!getBuffer() && "Buffer-specfic reference types should not be used outside of edsl "
                           "layout compatibility mode");
  }

  cg.emitInvokeMacroV(cg.getStringAttr("makeRef"), refVal.getIndex());
  return mlir::success();
}

mlir::LogicalResult TapType::emitLiteral(zirgen::codegen::CodegenEmitter& cg,
                                         mlir::Attribute value) const {
  auto tapValue = llvm::cast<Zll::TapAttr>(value);
  cg.emitInvokeMacroV(cg.getStringAttr("makeTap"),
                      tapValue.getRegGroupId(),
                      tapValue.getOffset(),
                      tapValue.getBack());
  return mlir::success();
}

} // namespace zirgen::ZStruct
