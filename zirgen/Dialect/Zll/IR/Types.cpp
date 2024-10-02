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

#include "zirgen/Dialect/Zll/IR/Types.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/TypeInterfaces.cpp.inc"

using namespace mlir;

namespace zirgen {
namespace Zll {

codegen::CodegenIdent<codegen::IdentKind::Type>
ValType::getTypeName(codegen::CodegenEmitter& cg) const {
  if (getFieldK() == 1) {
    return cg.getStringAttr("Val");
  } else {
    return cg.getStringAttr("ExtVal");
  }
}

ExtensionField ValType::getExtensionField() const {
  if (getExtended())
    return getField().getExtExtensionField();
  else
    return getField().getBaseExtensionField();
}

// Deprecated
ValType ValType::get(mlir::MLIRContext* ctx, uint64_t fieldP, unsigned fieldK) {
  assert(fieldP == kFieldPrimeDefault);
  auto field = getDefaultField(ctx);
  if (fieldK == 1)
    return get(ctx, field, /*extended=*/{});
  else {
    assert(fieldK == field.getExtDegree());
    return get(ctx, field, /*extended=*/UnitAttr::get(ctx));
  }
}

ValType ValType::getBaseType(mlir::MLIRContext* ctx) {
  return get(ctx, getDefaultField(ctx), /*extended=*/{});
}

ValType ValType::getExtensionType(mlir::MLIRContext* ctx) {
  return get(ctx, getDefaultField(ctx), /*extended=*/UnitAttr::get(ctx));
}

// Deprecated
uint64_t ValType::getFieldP() const {
  return getField().getPrime();
}

// Deprecated
unsigned ValType::getFieldK() const {
  return getExtended() ? getField().getExtDegree() : 1;
}

codegen::CodegenIdent<codegen::IdentKind::Type>
StringType::getTypeName(codegen::CodegenEmitter& cg) const {
  switch (cg.getLanguageKind()) {
  case codegen::LanguageKind::Rust:
    return cg.getStringAttr("&str");
  case codegen::LanguageKind::Cpp:
    return {cg.getStringAttr("std::string"), true};
  default:
    assert(false && "unreachable");
    return cg.getStringAttr("String");
  }
}

mlir::LogicalResult StringType::emitLiteral(zirgen::codegen::CodegenEmitter& cg,
                                            mlir::Attribute attr) const {
  // Only emit a literal if it's an unextended field element.
  auto strAttr = llvm::dyn_cast<StringAttr>(attr);
  if (!strAttr)
    return failure();

  cg.emitEscapedString(strAttr.getValue());
  return success();
}

codegen::CodegenIdent<codegen::IdentKind::Type>
VariadicType::getTypeName(codegen::CodegenEmitter& cg) const {
  std::string name;
  switch (cg.getLanguageKind()) {
  case codegen::LanguageKind::Rust:
    name = "[" + cg.getTypeName(getElement()).str() + "]";
    return cg.getStringAttr(name);
  case codegen::LanguageKind::Cpp:
    name = "std::initializer_list<" + cg.getTypeName(getElement()).str() + ">";
    return {cg.getStringAttr(name), true};
  default:
    assert(false && "unreachable");
    return cg.getStringAttr(name);
  }
}

codegen::CodegenIdent<codegen::IdentKind::Type>
BufferType::getTypeName(codegen::CodegenEmitter& cg) const {
  std::string name = stringifyBufferKind(getKind()).str();
  if (getElement().getFieldK() > 1)
    name += "Ext";
  name += "Buf";

  return cg.getStringAttr(name);
}

} // namespace Zll
} // namespace zirgen
