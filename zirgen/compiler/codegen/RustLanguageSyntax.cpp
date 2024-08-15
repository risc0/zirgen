// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "zirgen/Dialect/Zll/IR/Codegen.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace zirgen::Zll;

namespace zirgen::codegen {

namespace {

bool isReserved(StringRef ident) {
  static StringSet<> reserved = {"match"};
  return reserved.contains(ident);
}

} // namespace

void RustLanguageSyntax::emitConditional(CodegenEmitter& cg,
                                         CodegenValue condition,
                                         EmitPart emitThen) {
  cg << "if is_nonzero(" << condition << ") {\n";
  cg << emitThen;
  cg << "}\n";
}

void RustLanguageSyntax::emitSwitchStatement(CodegenEmitter& cg,
                                             CodegenIdent<IdentKind::Var> resultName,
                                             mlir::Type resultType,
                                             llvm::ArrayRef<CodegenValue> conditions,
                                             llvm::ArrayRef<EmitArmPartFunc> emitArm) {
  cg << "let ";
  if (Type(resultType).hasTrait<CodegenOnlyPassByReferenceTypeTrait>())
    cg << "&'static ";
  cg << resultName << ": " << cg.getTypeName(resultType) << ";\n";
  for (const auto& [cond, emitArm] : llvm::zip(conditions, emitArm)) {
    cg << "if is_nonzero(" << cond << ") {\n";
    auto result = emitArm();
    cg << resultName << " = " << result << ";\n";
    cg << "} else\n";
  }
  cg << "{\n";
  cg << "  bail!(\"Reached unreachable mux arm\")\n";
  cg << "}";
}

void RustLanguageSyntax::fallbackEmitLiteral(CodegenEmitter& cg,
                                             mlir::Type ty,
                                             mlir::Attribute value) {
  TypeSwitch<Attribute>(value)
      .Case<IntegerAttr>([&](auto intAttr) { cg << intAttr.getValue().getZExtValue(); })
      .Case<StringAttr>([&](auto strAttr) { cg.emitEscapedString(strAttr); })
      .Default([&](auto) {
        llvm::errs() << "Don't know how to emit type " << ty << " into rust++ with value " << value
                     << "\n";
        abort();
      });
}

void RustLanguageSyntax::emitFuncDefinition(CodegenEmitter& cg,
                                            CodegenIdent<IdentKind::Func> funcName,
                                            llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                            mlir::FunctionType funcType,
                                            mlir::Region* body) {
  cg << "pub fn " << funcName << "(";

  if (!contextArgDecls.empty()) {
    cg.interleaveComma(contextArgDecls, [&](auto contextArg) { cg << EmitPart(contextArg); });
    if (!argNames.empty())
      cg << ",";
  }

  cg.interleaveComma(zip(argNames, funcType.getInputs()), [&](auto nt) {
    CodegenIdent<IdentKind::Var> name = std::get<0>(nt);
    Type ty = std::get<1>(nt);
    cg << name << ": ";
    if (ty.hasTrait<CodegenLayoutTypeTrait>())
      cg << "BoundLayout<" << cg.getTypeName(ty) << ", impl BufferRow<ValType = Val>>";
    else {
      if (ty.hasTrait<CodegenNeedsCloneTypeTrait>() ||
          ty.hasTrait<CodegenOnlyPassByReferenceTypeTrait>())
        cg << "&";
      else if (ty.hasTrait<CodegenPassByMutRefTypeTrait>())
        cg << "&mut ";
      cg << cg.getTypeName(ty);
    }
  });
  cg << ") -> Result<";
  if (funcType.getNumResults() != 1) {
    cg << "(";
  }
  cg.interleaveComma(funcType.getResults(), [&](auto ty) { cg << cg.getTypeName(ty); });
  if (funcType.getNumResults() != 1) {
    cg << ")";
  }

  cg << "> {\n";
  cg.emitRegion(*body);
  cg << "}\n";
}

void RustLanguageSyntax::emitReturn(CodegenEmitter& cg, llvm::ArrayRef<CodegenValue> values) {
  cg << "return Ok(";
  if (values.size() != 1) {
    cg << "(";
  }
  cg.interleaveComma(values);
  if (values.size() != 1) {
    cg << ")";
  }
  cg << ");\n";
}

void RustLanguageSyntax::emitSaveResults(CodegenEmitter& cg,
                                         ArrayRef<CodegenIdent<IdentKind::Var>> names,
                                         llvm::ArrayRef<mlir::Type> types,
                                         EmitPart emitExpression) {
  if (names.empty()) {
    cg << emitExpression << ";\n";
  } else if (names.size() == 1) {
    cg << "let " << names[0];
    Type ty = types[0];
    if (ty.hasTrait<CodegenLayoutTypeTrait>()) {
      cg << " : BoundLayout<" << cg.getTypeName(types[0]) << ", _>";
    } else {
      cg << " : ";
      if (Type(types[0]).hasTrait<CodegenOnlyPassByReferenceTypeTrait>())
        cg << "&'static ";
      cg << cg.getTypeName(types[0]);
    }
    cg << " = " << emitExpression << ";\n";
  } else {
    cg << "let (";
    cg.interleaveComma(names);
    cg << ") = " << emitExpression << ";\n";
  }
}

void RustLanguageSyntax::emitSaveConst(CodegenEmitter& cg,
                                       CodegenIdent<IdentKind::Const> name,
                                       CodegenValue value) {
  Type ty = value.getType();
  cg << "pub const " << name << ": ";
  if (ty.hasTrait<CodegenOnlyPassByReferenceTypeTrait>())
    cg << "&";
  cg << cg.getTypeName(value.getType()) << " = " << value << ";\n";
}

static bool isReferenceType(CodegenValue value) {
  // TODO: This seems kludgy; maybe figure out some way to propagate whether it's already a
  // reference?
  auto blockArg = llvm::dyn_cast_if_present<BlockArgument>(value.getValue());
  if (!blockArg)
    return false;
  return llvm::isa<FunctionOpInterface>(blockArg.getOwner()->getParentOp());
}

void RustLanguageSyntax::emitCall(CodegenEmitter& cg,
                                  CodegenIdent<IdentKind::Func> callee,
                                  llvm::ArrayRef<StringRef> contextArgs,
                                  llvm::ArrayRef<CodegenValue> args) {
  cg << callee << "(";
  if (!contextArgs.empty()) {
    cg.interleaveComma(contextArgs, [&](auto contextArg) { cg << EmitPart(contextArg); });
    if (!args.empty())
      cg << ",";
  }
  cg.interleaveComma(args, [&](auto arg) {
    Type ty = arg.getType();
    bool isAlreadyRef = isReferenceType(arg);
    if ((ty.hasTrait<CodegenNeedsCloneTypeTrait>() || ty.hasTrait<CodegenNeedsCloneTypeTrait>()) &&
        !isAlreadyRef) {
      cg << "&";
    }

    cg.emitValue(arg);
  });
  cg << ")?";
}

void RustLanguageSyntax::emitInvokeMacro(CodegenEmitter& cg,
                                         CodegenIdent<IdentKind::Macro> callee,
                                         llvm::ArrayRef<StringRef> contextArgs,
                                         llvm::ArrayRef<EmitPart> emitArgs) {
  bool isItemsMacro = itemsMacros.contains(canonIdent(callee.strref(), IdentKind::Macro));
  cg << callee << "!" << (isItemsMacro ? "{" : "(");
  if (!contextArgs.empty()) {
    cg.interleaveComma(contextArgs, [&](auto contextArg) { cg << EmitPart(contextArg); });
    if (!emitArgs.empty())
      cg << ",";
  }
  cg.interleaveComma(emitArgs);
  cg << (isItemsMacro ? "}" : ")");
}

std::string RustLanguageSyntax::canonIdent(llvm::StringRef ident, IdentKind kind) {
  switch (kind) {
  case IdentKind::Var:
  case IdentKind::Field:
  case IdentKind::Func:
    if (isReserved(ident))
      return "r#" + convertToSnakeFromCamelCase(ident);
    else
      return convertToSnakeFromCamelCase(ident);
  case IdentKind::Macro:
    return convertToSnakeFromCamelCase(ident);
  case IdentKind::Type:
    return convertToCamelFromSnakeCase(ident, /*capitalizeFirst=*/true);
  case IdentKind::Const: {
    std::string snake = convertToSnakeFromCamelCase(ident);
    std::string constName;
    for (char c : snake) {
      constName.push_back(llvm::toUpper(c));
    }
    return constName;
  }
  }
  throw(std::runtime_error("Unknown ident type"));
}

void RustLanguageSyntax::emitClone(CodegenEmitter& cg, CodegenIdent<IdentKind::Var> value) {
  cg << value << ".clone()";
}
void RustLanguageSyntax::emitTakeReference(CodegenEmitter& cg, EmitPart emitTarget) {
  cg << "&" << emitTarget;
}

namespace detail {
void emitStructDef(CodegenEmitter& cg,
                   mlir::Type ty,
                   llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                   llvm::ArrayRef<mlir::Type> types) {
  cg << "pub struct " << cg.getTypeName(ty) << " {\n";
  assert(names.size() == types.size());
  for (size_t i = 0; i != names.size(); i++) {
    cg << "  pub " << names[i] << ": ";
    if (Type(types[i]).hasTrait<CodegenOnlyPassByReferenceTypeTrait>())
      cg << "&'static ";
    cg << cg.getTypeName(types[i]) << ",\n";
  }
  cg << "}\n";
}
} // namespace detail

void RustLanguageSyntax::emitStructDef(CodegenEmitter& cg,
                                       mlir::Type ty,
                                       llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                       llvm::ArrayRef<mlir::Type> types) {
  cg << "#[derive(Copy,Clone,Debug)]\n";
  detail::emitStructDef(cg, ty, names, types);
}

void RustLanguageSyntax::emitStructConstruct(CodegenEmitter& cg,
                                             mlir::Type ty,
                                             llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                             llvm::ArrayRef<CodegenValue> values) {
  if (ty.hasTrait<CodegenOnlyPassByReferenceTypeTrait>())
    cg << "&";
  cg << cg.getTypeName(ty) << "{\n";
  assert(names.size() == values.size());

  for (size_t i = 0; i != names.size(); i++) {
    cg << "  " << names[i] << ": " << values[i] << ",\n";
  }
  cg << "}";
}

void RustLanguageSyntax::emitArrayDef(CodegenEmitter& cg,
                                      mlir::Type ty,
                                      mlir::Type elemType,
                                      size_t numElems) {
  cg << "pub type " << cg.getTypeName(ty) << " = [";
  if (elemType.hasTrait<CodegenOnlyPassByReferenceTypeTrait>())
    cg << "&'static ";
  cg << cg.getTypeName(elemType) << "; " << numElems << "];\n";
}

void RustLanguageSyntax::emitArrayConstruct(CodegenEmitter& cg,
                                            mlir::Type ty,
                                            mlir::Type elemType,
                                            llvm::ArrayRef<CodegenValue> values) {
  if (ty.hasTrait<CodegenOnlyPassByReferenceTypeTrait>())
    cg << "&";
  cg << "[";
  cg.interleaveComma(values, [&](auto value) { cg << value; });
  cg << "]";
}

void RustLanguageSyntax::emitMapConstruct(CodegenEmitter& cg,
                                          CodegenValue array,
                                          std::optional<CodegenValue> layout,
                                          llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                          mlir::Region& body) {
  if (layout)
    cg << "map_layout(";
  else
    cg << "map(";
  if (isReferenceType(array))
    cg << "*";
  cg << array;
  if (layout)
    cg << ", " << *layout;
  cg << ", |";
  cg.interleaveComma(argNames);
  cg << "| {\n";
  cg.emitRegion(body);
  cg << "\n})?";
}

void RustLanguageSyntax::emitReduceConstruct(CodegenEmitter& cg,
                                             CodegenValue array,
                                             CodegenValue init,
                                             std::optional<CodegenValue> layout,
                                             llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                             mlir::Region& body) {
  assert(argNames.size() == (layout ? 3 : 2));
  if (layout)
    cg << "reduce_layout(";
  else
    cg << "reduce(";
  if (isReferenceType(array))
    cg << "*";
  cg << array << ", ";
  if (isReferenceType(init))
    cg << "*";
  cg << init;
  if (layout)
    cg << ", " << *layout;
  cg << ", |";
  cg.interleaveComma(argNames);
  cg << "| {\n";
  cg.emitRegion(body);
  cg << "\n})?";
}

void RustLanguageSyntax::emitLayoutDef(CodegenEmitter& cg,
                                       mlir::Type ty,
                                       llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                       llvm::ArrayRef<mlir::Type> types) {
  // Layout structures define a visitor interface on top of a struct.
  detail::emitStructDef(cg, ty, names, types);

  auto tyName = cg.getTypeName(ty);

  cg << "impl risc0_zkp::layout::Component for " << tyName << " {\n";
  cg << "  fn ty_name(&self) -> &'static str { \"" << tyName << "\" }\n";
  cg << "  #[allow(unused_variables)]\n";
  cg << "  fn walk<V: risc0_zkp::layout::Visitor>(&self, v: &mut V) -> core::fmt::Result {\n";
  for (size_t i = 0; i != names.size(); ++i) {
    cg << "    v.visit_component(\"" << names[i] << "\", ";
    Type ty = types[i];
    if (!ty.hasTrait<CodegenOnlyPassByReferenceTypeTrait>()) {
      // visit_component always requires a reference, so add one
      // if it's not already a reference type.
      cg << "&";
    }
    cg << "self." << names[i] << ")?;\n";
  }
  cg << "    Ok(())\n";
  cg << "  }\n";
  cg << "}\n";
}

void RustLanguageSyntax::addItemsMacro(StringRef macroName) {
  itemsMacros.insert(canonIdent(macroName, IdentKind::Macro));
}

} // namespace zirgen::codegen
