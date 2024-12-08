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

void CppLanguageSyntax::emitConditional(CodegenEmitter& cg,
                                        CodegenValue condition,
                                        EmitPart emitThen) {
  cg << "if (to_size_t(" << condition << ")) {\n";
  cg << emitThen;
  cg << "}\n";
}

void CppLanguageSyntax::emitSwitchStatement(CodegenEmitter& cg,
                                            CodegenIdent<IdentKind::Var> resultName,
                                            mlir::Type resultType,
                                            llvm::ArrayRef<CodegenValue> conditions,
                                            llvm::ArrayRef<EmitArmPartFunc> emitArms) {
  cg << cg.getTypeName(resultType) << " " << resultName << ";\n";
  for (const auto& [cond, emitArm] : llvm::zip(conditions, emitArms)) {
    cg << "if (to_size_t(" << cond << ")) {\n";
    auto result = emitArm();
    cg << resultName << " = " << result << ";\n";
    cg << "} else ";
  }
  cg << "{\n";
  cg << "   assert(0 && \"Reached unreachable mux arm\");\n";
  cg << "}\n";
}

namespace {

void emitRawFuncDeclaration(CodegenEmitter& cg,
                            CodegenIdent<IdentKind::Func> funcName,
                            llvm::ArrayRef<std::string> contextArgDecls,
                            llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                            mlir::FunctionType funcType) {
  auto returnTypes = funcType.getResults();
  if (returnTypes.size() == 0) {
    cg << "void";
  } else if (returnTypes.size() == 1) {
    cg << cg.getTypeName(returnTypes[0]);
  } else {
    cg << "std::tuple<";
    cg.interleaveComma(returnTypes, [&](auto ty) { cg << cg.getTypeName(ty); });
    cg << ">";
  }

  cg << " " << funcName << "(";
  if (!contextArgDecls.empty()) {
    cg.interleaveComma(contextArgDecls, [&](auto contextArg) { cg << EmitPart(contextArg); });
    if (!argNames.empty())
      cg << ",";
  }
  cg.interleaveComma(zip(argNames, funcType.getInputs()), [&](auto vt) {
    auto argName = std::get<0>(vt);
    Type argType = std::get<1>(vt);

    if (argType.hasTrait<CodegenLayoutTypeTrait>())
      cg << "BoundLayout<" << cg.getTypeName(argType) << ">";
    else
      cg << cg.getTypeName(argType);
    cg << " " << argName;
  });

  cg << ")  ";
}

} // namespace

void CppLanguageSyntax::emitFuncDeclaration(CodegenEmitter& cg,
                                            CodegenIdent<IdentKind::Func> funcName,
                                            llvm::ArrayRef<std::string> contextArgDecls,
                                            llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                            mlir::FunctionType funcType) {
  cg << "extern ";
  emitRawFuncDeclaration(cg, funcName, contextArgDecls, argNames, funcType);
  cg << ";\n";
}

void CppLanguageSyntax::emitFuncDefinition(CodegenEmitter& cg,
                                           CodegenIdent<IdentKind::Func> funcName,
                                           llvm::ArrayRef<std::string> contextArgDecls,
                                           llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                           mlir::FunctionType funcType,
                                           mlir::Region* body) {
  emitRawFuncDeclaration(cg, funcName, contextArgDecls, argNames, funcType);
  cg << " {\n";
  cg.emitRegion(*body);
  cg << "}\n";
}

void CppLanguageSyntax::emitReturn(CodegenEmitter& cg, llvm::ArrayRef<CodegenValue> values) {

  cg << "return ";
  if (values.size() > 1) {
    cg << "std::make_tuple(";
  }
  cg.interleaveComma(values);
  if (values.size() > 1) {
    cg << ")";
  }
  cg << ";\n";
}

void CppLanguageSyntax::emitSaveResults(CodegenEmitter& cg,
                                        ArrayRef<CodegenIdent<IdentKind::Var>> names,
                                        llvm::ArrayRef<mlir::Type> types,
                                        EmitPart emitExpression) {
  if (names.empty()) {
    cg << emitExpression << ";\n";
  } else if (names.size() == 1) {
    if (Type(types[0]).hasTrait<CodegenLayoutTypeTrait>()) {
      cg << "BoundLayout<" << cg.getTypeName(types[0]) << ">";
    } else {
      cg << cg.getTypeName(types[0]);
    }
    cg << " " << names[0] << " = " << emitExpression << ";\n";
  } else {
    cg << "auto [";
    cg.interleaveComma(names);
    cg << "] = " << emitExpression << ";\n";
  }
}

void CppLanguageSyntax::emitConstDecl(CodegenEmitter& cg,
                                      CodegenIdent<IdentKind::Const> name,
                                      Type ty) {
  cg << "extern const " << cg.getTypeName(ty) << " " << name << ";\n";
}

void CppLanguageSyntax::emitSaveConst(CodegenEmitter& cg,
                                      CodegenIdent<IdentKind::Const> name,
                                      CodegenValue value) {
  cg << "constexpr " << cg.getTypeName(value.getType()) << " " << name << " = " << value << ";\n";
}

void CppLanguageSyntax::emitCall(CodegenEmitter& cg,
                                 CodegenIdent<IdentKind::Func> callee,
                                 llvm::ArrayRef<std::string> contextArgs,
                                 llvm::ArrayRef<CodegenValue> args) {
  cg << callee << "(";
  if (!contextArgs.empty()) {
    cg.interleaveComma(contextArgs, [&](auto contextArg) { cg << EmitPart(contextArg); });
    if (!args.empty())
      cg << ",";
  }
  cg.interleaveComma(args);
  cg << ")";
}

void CppLanguageSyntax::emitInvokeMacro(CodegenEmitter& cg,
                                        CodegenIdent<IdentKind::Macro> callee,
                                        llvm::ArrayRef<StringRef> contextArgs,
                                        llvm::ArrayRef<EmitPart> emitArgs) {
  cg << callee;
  if (contextArgs.empty() && emitArgs.empty())
    return;

  cg << "(";
  if (!contextArgs.empty()) {
    cg.interleaveComma(contextArgs, [&](auto contextArg) { cg << EmitPart(contextArg); });
    if (!emitArgs.empty())
      cg << ",";
  }
  cg.interleaveComma(emitArgs);
  cg << ")";
}

std::string CppLanguageSyntax::canonIdent(llvm::StringRef ident, IdentKind kind) {
  switch (kind) {
  case IdentKind::Var:
  case IdentKind::Field:
  case IdentKind::Func: {
    std::string str = convertToCamelFromSnakeCase(ident);
    if (!str.empty())
      str[0] = llvm::toLower(str[0]);
    return str;
  }
  case IdentKind::Type:
    return convertToCamelFromSnakeCase(ident, /*capitalizeFirst=*/true);
  case IdentKind::Const:
    return "k" + convertToCamelFromSnakeCase(ident, /*capitalizeFirst=*/true);
  case IdentKind::Macro:
    return llvm::StringRef(convertToSnakeFromCamelCase(ident)).upper();
  }
  throw(std::runtime_error("Unknown ident kind"));
}

void CppLanguageSyntax::emitStructDef(CodegenEmitter& cg,
                                      mlir::Type ty,
                                      llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                      llvm::ArrayRef<mlir::Type> types) {
  emitStructDefImpl(cg, ty, names, types, /*layout=*/false);
}

void CppLanguageSyntax::emitStructDefImpl(CodegenEmitter& cg,
                                          mlir::Type ty,
                                          llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                          llvm::ArrayRef<mlir::Type> types,
                                          bool layout) {
  cg << "struct " << cg.getTypeName(ty) << " {\n";
  assert(names.size() == types.size());
  for (size_t i = 0; i != names.size(); i++) {
    cg << "  ";
    Type subTy = types[i];
    if (subTy.hasTrait<CodegenLayoutTypeTrait>() && !layout) {
      cg << "BoundLayout<" << cg.getTypeName(types[i]) << ">";
    } else {
      cg << cg.getTypeName(types[i]);
    }
    cg << " " << names[i] << ";\n";
  }
  cg << "};\n";
}

void CppLanguageSyntax::emitStructConstruct(CodegenEmitter& cg,
                                            mlir::Type ty,
                                            llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                            llvm::ArrayRef<CodegenValue> values) {
  cg << cg.getTypeName(ty) << "{\n";
  assert(names.size() == values.size());

  cg.interleaveComma(zip(names, values), [&](auto zipped) {
    auto [name, value] = zipped;
    cg << "  ." << name << " = " << value;
  });
  cg << "}";
}

void CppLanguageSyntax::emitArrayDef(CodegenEmitter& cg,
                                     mlir::Type ty,
                                     mlir::Type elemType,
                                     size_t numElems) {
  cg << "using " << cg.getTypeName(ty) << " = std::array<" << cg.getTypeName(elemType) << ", "
     << numElems << ">;\n";
}

void CppLanguageSyntax::emitArrayConstruct(CodegenEmitter& cg,
                                           mlir::Type ty,
                                           mlir::Type elemType,
                                           llvm::ArrayRef<CodegenValue> values) {
  cg << cg.getTypeName(ty);
  cg << "{";
  cg.interleaveComma(values);
  cg << "}";
}

void CppLanguageSyntax::emitMapConstruct(CodegenEmitter& cg,
                                         CodegenValue array,
                                         std::optional<CodegenValue> layout,
                                         llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                         mlir::Region& body) {
  cg << "map(" << array << ", ";
  if (layout)
    cg << *layout << ", ";
  cg << "([&](";
  cg << cg.getTypeName(array.getType()) << "::value_type " << argNames[0];
  if (layout)
    cg << ", BoundLayout<" << cg.getTypeName(layout->getType()) << "::value_type> " << argNames[1];
  cg << ") {\n";
  cg.emitRegion(body);
  cg << "\n}))";
}

void CppLanguageSyntax::emitReduceConstruct(CodegenEmitter& cg,
                                            CodegenValue array,
                                            CodegenValue init,
                                            std::optional<CodegenValue> layout,
                                            llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                            mlir::Region& body) {
  cg << "reduce(" << array << ", " << init << ", ";
  if (layout) {
    cg << *layout << ", ";
  }
  cg << "([&](";
  cg << cg.getTypeName(init.getType()) << " " << argNames[0] << ", "
     << cg.getTypeName(array.getType()) << "::value_type " << argNames[1];
  if (layout) {
    cg << ", BoundLayout<" << cg.getTypeName(layout->getType()) << "::value_type> " << argNames[2];
  }
  cg << ") {\n";
  cg.emitRegion(body);
  cg << "\n}))";
}

void CppLanguageSyntax::emitLayoutDef(CodegenEmitter& cg,
                                      mlir::Type ty,
                                      llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                      llvm::ArrayRef<mlir::Type> types) {
  // In C++, layouts are just regular structures.
  emitStructDefImpl(cg, ty, names, types, /*layout=*/true);
}

// ----------------------------------------------------------------------
// CUDA variant of C++ that needs things slightly different than regular C++
void CudaLanguageSyntax::emitConstDecl(CodegenEmitter& cg,
                                       CodegenIdent<IdentKind::Const> name,
                                       Type ty) {
  cg << "extern __device__ const " << cg.getTypeName(ty) << " " << name << ";\n";
}

void CudaLanguageSyntax::emitSaveConst(CodegenEmitter& cg,
                                       CodegenIdent<IdentKind::Const> name,
                                       CodegenValue value) {
  // Make constants available on the device
  cg << "__device__ ";
  CppLanguageSyntax::emitSaveConst(cg, name, value);
}

void CudaLanguageSyntax::emitArrayDef(CodegenEmitter& cg,
                                      mlir::Type ty,
                                      mlir::Type elemType,
                                      size_t numElems) {
  cg << "using " << cg.getTypeName(ty) << " = ::cuda::std::array<" << cg.getTypeName(elemType)
     << "," << numElems << ">;\n";
}

void CudaLanguageSyntax::emitArrayConstruct(CodegenEmitter& cg,
                                            mlir::Type ty,
                                            mlir::Type elemType,
                                            llvm::ArrayRef<CodegenValue> values) {
  cg << cg.getTypeName(ty) << "{";
  cg.interleaveComma(values);
  cg << "}";
}

void CudaLanguageSyntax::emitFuncDefinition(CodegenEmitter& cg,
                                            CodegenIdent<IdentKind::Func> funcName,
                                            llvm::ArrayRef<std::string> contextArgDecls,
                                            llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                            mlir::FunctionType funcType,
                                            mlir::Region* body) {
  cg << "__device__ ";
  CppLanguageSyntax::emitFuncDefinition(cg, funcName, contextArgDecls, argNames, funcType, body);
}

void CudaLanguageSyntax::emitFuncDeclaration(CodegenEmitter& cg,
                                             CodegenIdent<IdentKind::Func> funcName,
                                             llvm::ArrayRef<std::string> contextArgDecls,
                                             llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                             mlir::FunctionType funcType) {
  cg << "extern __device__ ";
  emitRawFuncDeclaration(cg, funcName, contextArgDecls, argNames, funcType);
  cg << ";\n";
}

} // namespace zirgen::codegen
