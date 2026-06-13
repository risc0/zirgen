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

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "zirgen/Dialect/Zll/IR/Codegen.h"
#include "zirgen/compiler/codegen/Passes.h"
#include "zirgen/compiler/codegen/protocol_info_const.h"
#include "llvm/Support/ManagedStatic.h"

#include <memory>
#include <string>

namespace zirgen {
namespace recursion {
struct EncodeStats;
}

/// Abstract interface for writing Rust source output for a circuit. Obtain a
/// concrete instance via `createRustStreamEmitter`. The caller drives the
/// emission sequence by calling each method in the required order.
class RustStreamEmitter {
public:
  virtual ~RustStreamEmitter() = default;
  /// Emit a witness-generation step function.
  virtual void emitStepFunc(const std::string& name, mlir::func::FuncOp func) = 0;
  /// Emit one split of the validity polynomial; `idx` is the split index,
  /// `nsplit` is the total number of splits.
  virtual void
  emitPolyFunc(const std::string& fn, mlir::func::FuncOp func, size_t idx, size_t nsplit) = 0;
  /// Emit the extension-field validity polynomial function.
  virtual void emitPolyExtFunc(mlir::func::FuncOp func) = 0;
  /// Emit the tap set (column references used by the DEEP-ALI protocol).
  virtual void emitTaps(mlir::func::FuncOp func) = 0;
  /// Emit protocol metadata constants (hash type, field size, etc.).
  virtual void emitInfo(mlir::func::FuncOp func) = 0;
};

/// Abstract interface for writing GPU (CUDA/Metal) source output for a circuit.
/// Obtain a concrete instance via `createGpuStreamEmitter`.
class GpuStreamEmitter {
public:
  virtual ~GpuStreamEmitter() = default;
  /// Emit one split of the validity polynomial.
  /// When `declsOnly` is true, only emit declarations (for header files).
  virtual void
  emitPoly(mlir::func::FuncOp func, size_t idx, size_t nsplit, bool declsOnly = false) = 0;
  /// Emit a witness-generation step function.
  virtual void emitStepFunc(const std::string& name, mlir::func::FuncOp func) = 0;
};

/// Abstract interface for writing C++ source output for a circuit.
/// Obtain a concrete instance via `createCppStreamEmitter`.
class CppStreamEmitter {
public:
  virtual ~CppStreamEmitter() = default;
  /// Emit the validity polynomial function.
  virtual void emitPoly(mlir::func::FuncOp func) = 0;
  /// Emit the tap set.
  virtual void emitTaps(mlir::func::FuncOp func) = 0;
  /// Emit include/forward-declaration boilerplate.
  virtual void emitHeader(mlir::func::FuncOp func) = 0;
};

/// Per-stage configuration for `emitCode`.
struct StageOptions {
  /// Optional extra MLIR passes to run on this stage's module before emission.
  std::function<void(mlir::OpPassManager& opm)> addExtraPasses;

  /// If non-empty, write this stage's output to this filename instead of the
  /// stage name.
  std::string outputFile;
};

/// Top-level options for `emitCode`. Callers can customize individual stages
/// by populating the `stages` map keyed on stage name.
struct EmitCodeOptions {
  llvm::StringMap<StageOptions> stages;
};

namespace codegen {

/// `LanguageSyntax` implementation for Rust code generation. Handles
/// Rust-specific identifier casing, reference/clone semantics, lifetime
/// annotations, `match` statements, and struct/array construction syntax.
class RustLanguageSyntax : public LanguageSyntax {
public:
  /// Mark `macroName` as an "items macro" that is invoked with `{ }` instead
  /// of `( )`, allowing it to expand in statement position.
  void addItemsMacro(llvm::StringRef macroName);

private:
  LanguageKind getLanguageKind() override { return LanguageKind::Rust; }
  std::string canonIdent(llvm::StringRef ident, IdentKind idt) override;
  void emitClone(CodegenEmitter& cg, CodegenIdent<IdentKind::Var> value) override;
  void emitTakeReference(CodegenEmitter& cg, EmitPart emitTarget) override;

  void emitConditional(CodegenEmitter& cg, CodegenValue condition, EmitPart emitThen) override;
  void emitSwitchStatement(CodegenEmitter& cg,
                           CodegenIdent<IdentKind::Var> resultNames,
                           mlir::Type resultType,
                           llvm::ArrayRef<CodegenValue> conditions,
                           llvm::ArrayRef<EmitArmPartFunc> emitArms) override;

  void emitFuncDefinition(CodegenEmitter& cg,
                          CodegenIdent<IdentKind::Func> funcName,
                          llvm::ArrayRef<std::string> contextArgs,
                          llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                          mlir::FunctionType funcType,
                          mlir::Region* body) override;
  void emitFuncDeclaration(CodegenEmitter& cg,
                           CodegenIdent<IdentKind::Func> funcName,
                           llvm::ArrayRef<std::string> contextArgs,
                           llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                           mlir::FunctionType funcType) override;

  void emitReturn(CodegenEmitter& cg, llvm::ArrayRef<CodegenValue> values) override;

  void emitSaveResults(CodegenEmitter& cg,
                       llvm::ArrayRef<CodegenIdent<IdentKind::Var>> names,
                       llvm::ArrayRef<mlir::Type> types,
                       EmitPart emitExpression) override;

  void emitSaveConst(CodegenEmitter& cg,
                     CodegenIdent<IdentKind::Const> name,
                     CodegenValue value) override;
  void
  emitConstDecl(CodegenEmitter& cg, CodegenIdent<IdentKind::Const> name, mlir::Type type) override;

  void emitCall(CodegenEmitter& cg,
                CodegenIdent<IdentKind::Func> callee,
                llvm::ArrayRef<std::string> contextArgs,
                llvm::ArrayRef<CodegenValue> args) override;

  void emitInvokeMacro(CodegenEmitter& cg,
                       CodegenIdent<IdentKind::Macro> callee,
                       llvm::ArrayRef<llvm::StringRef> contextArgs,
                       llvm::ArrayRef<EmitPart> emitArgs) override;

  void emitStructDef(CodegenEmitter& cg,
                     mlir::Type ty,
                     llvm::ArrayRef<CodegenIdent<IdentKind::Field>> fields,
                     llvm::ArrayRef<mlir::Type> types) override;
  void emitStructConstruct(CodegenEmitter& cg,
                           mlir::Type ty,
                           llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                           llvm::ArrayRef<CodegenValue> values) override;
  void
  emitArrayDef(CodegenEmitter& cg, mlir::Type ty, mlir::Type elemType, size_t numElems) override;
  void emitArrayConstruct(CodegenEmitter& cg,
                          mlir::Type ty,
                          mlir::Type elemType,
                          llvm::ArrayRef<CodegenValue> values) override;
  void emitMapConstruct(CodegenEmitter& cg,
                        CodegenValue array,
                        std::optional<CodegenValue> layout,
                        llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                        mlir::Region& body) override;
  void emitReduceConstruct(CodegenEmitter& cg,
                           CodegenValue array,
                           CodegenValue init,
                           std::optional<CodegenValue> layout,
                           llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                           mlir::Region& body) override;
  void emitLayoutDef(CodegenEmitter& cg,
                     mlir::Type ty,
                     llvm::ArrayRef<CodegenIdent<IdentKind::Field>> fields,
                     llvm::ArrayRef<mlir::Type> types) override;

  llvm::StringSet<> itemsMacros;
  llvm::DenseMap<mlir::Type, bool> typesNeedLifetime;

private:
  void emitStructDefImpl(CodegenEmitter& cg,
                         mlir::Type ty,
                         llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                         llvm::ArrayRef<mlir::Type> types,
                         bool layout);
  void emitValueWithReferenceIfNeeded(CodegenEmitter& cg, CodegenValue value);
  bool typeNeedsLifetime(mlir::Type ty);
};

/// `LanguageSyntax` implementation for C++ code generation.
struct CppLanguageSyntax : public LanguageSyntax {
  LanguageKind getLanguageKind() override { return LanguageKind::Cpp; }

  std::string canonIdent(llvm::StringRef ident, IdentKind idt) override;

  void emitConditional(CodegenEmitter& cg, CodegenValue condition, EmitPart emitThen) override;
  void emitSwitchStatement(CodegenEmitter& cg,
                           CodegenIdent<IdentKind::Var> resultName,
                           mlir::Type resultType,
                           llvm::ArrayRef<CodegenValue> conditions,
                           llvm::ArrayRef<EmitArmPartFunc> emitArms) override;

  void emitFuncDefinition(CodegenEmitter& cg,
                          CodegenIdent<IdentKind::Func> funcName,
                          llvm::ArrayRef<std::string> contextArgs,
                          llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                          mlir::FunctionType funcType,
                          mlir::Region* body) override;
  void emitFuncDeclaration(CodegenEmitter& cg,
                           CodegenIdent<IdentKind::Func> funcName,
                           llvm::ArrayRef<std::string> contextArgs,
                           llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                           mlir::FunctionType funcType) override;

  void emitReturn(CodegenEmitter& cg, llvm::ArrayRef<CodegenValue> values) override;

  void emitSaveResults(CodegenEmitter& cg,
                       llvm::ArrayRef<CodegenIdent<IdentKind::Var>> names,
                       llvm::ArrayRef<mlir::Type> types,
                       EmitPart emitExpression) override;

  void emitSaveConst(CodegenEmitter& cg,
                     CodegenIdent<IdentKind::Const> name,
                     CodegenValue value) override;
  void
  emitConstDecl(CodegenEmitter& cg, CodegenIdent<IdentKind::Const> name, mlir::Type type) override;

  void emitCall(CodegenEmitter& cg,
                CodegenIdent<IdentKind::Func> callee,
                llvm::ArrayRef<std::string> contextArgs,
                llvm::ArrayRef<CodegenValue> args) override;

  void emitInvokeMacro(CodegenEmitter& cg,
                       CodegenIdent<IdentKind::Macro> callee,
                       llvm::ArrayRef<llvm::StringRef> contextArgs,
                       llvm::ArrayRef<EmitPart> emitArgs) override;

  void emitStructDef(CodegenEmitter& cg,
                     mlir::Type ty,
                     llvm::ArrayRef<CodegenIdent<IdentKind::Field>> fields,
                     llvm::ArrayRef<mlir::Type> types) override;
  void emitStructConstruct(CodegenEmitter& cg,
                           mlir::Type ty,
                           llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                           llvm::ArrayRef<CodegenValue> values) override;
  void
  emitArrayDef(CodegenEmitter& cg, mlir::Type ty, mlir::Type elemType, size_t numElems) override;
  void emitArrayConstruct(CodegenEmitter& cg,
                          mlir::Type ty,
                          mlir::Type elemType,
                          llvm::ArrayRef<CodegenValue> values) override;
  void emitMapConstruct(CodegenEmitter& cg,
                        CodegenValue array,
                        std::optional<CodegenValue> layout,
                        llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                        mlir::Region& body) override;
  void emitReduceConstruct(CodegenEmitter& cg,
                           CodegenValue array,
                           CodegenValue init,
                           std::optional<CodegenValue> layout,
                           llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                           mlir::Region& body) override;
  void emitLayoutDef(CodegenEmitter& cg,
                     mlir::Type ty,
                     llvm::ArrayRef<CodegenIdent<IdentKind::Field>> fields,
                     llvm::ArrayRef<mlir::Type> types) override;

private:
  void emitStructDefImpl(CodegenEmitter& cg,
                         mlir::Type ty,
                         llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                         llvm::ArrayRef<mlir::Type> types,
                         bool layout);
};

/// `LanguageSyntax` implementation for CUDA C++ code generation. Extends
/// `CppLanguageSyntax` to emit `__device__ __forceinline__` qualifiers and
/// CUDA-compatible constant/array representations.
struct CudaLanguageSyntax : public CppLanguageSyntax {
  void
  emitConstDecl(CodegenEmitter& cg, CodegenIdent<IdentKind::Const> name, mlir::Type type) override;
  void emitSaveConst(CodegenEmitter& cg,
                     CodegenIdent<IdentKind::Const> name,
                     CodegenValue value) override;
  void
  emitArrayDef(CodegenEmitter& cg, mlir::Type ty, mlir::Type elemType, size_t numElems) override;
  void emitArrayConstruct(CodegenEmitter& cg,
                          mlir::Type ty,
                          mlir::Type elemType,
                          llvm::ArrayRef<CodegenValue> values) override;
  void emitFuncDefinition(CodegenEmitter& cg,
                          CodegenIdent<IdentKind::Func> funcName,
                          llvm::ArrayRef<std::string> contextArgs,
                          llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                          mlir::FunctionType funcType,
                          mlir::Region* body) override;
  void emitFuncDeclaration(CodegenEmitter& cg,
                           CodegenIdent<IdentKind::Func> funcName,
                           llvm::ArrayRef<std::string> contextArgs,
                           llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                           mlir::FunctionType funcType) override;
};

/// Return a fully-configured `CodegenOptions` for emitting Rust source,
/// including dialect-specific op lowering for all dialects used by zirgen.
CodegenOptions getRustCodegenOpts();
/// Return a fully-configured `CodegenOptions` for emitting C++ source.
CodegenOptions getCppCodegenOpts();
/// Return a fully-configured `CodegenOptions` for emitting CUDA C++ source.
CodegenOptions getCudaCodegenOpts();

} // namespace codegen

/// Create a Rust stream emitter writing to `ofs`.
std::unique_ptr<RustStreamEmitter> createRustStreamEmitter(llvm::raw_ostream& ofs);
/// Create a GPU (CUDA/Metal) stream emitter writing to `ofs`.
/// `suffix` selects the target language (e.g. "cu" or "metal").
std::unique_ptr<GpuStreamEmitter> createGpuStreamEmitter(llvm::raw_ostream& ofs,
                                                         const std::string& suffix);
/// Create a C++ stream emitter writing to `ofs`.
std::unique_ptr<CppStreamEmitter> createCppStreamEmitter(llvm::raw_ostream& ofs);

/// Lower `module` to source files in the output directory configured via
/// `codegenCLOptions`. Stage outputs are controlled by `opts.stages`.
void emitCode(mlir::ModuleOp module, const EmitCodeOptions& opts = {});
/// Emit zirgen polynomial source files into `outputDir`.
void emitCodeZirgenPoly(mlir::ModuleOp module, llvm::StringRef outputDir);
/// Encode a recursion witness and emit source to `path`.
/// If `stats` is non-null it is populated with encoding statistics.
void emitRecursion(const std::string& path,
                   mlir::func::FuncOp func,
                   recursion::EncodeStats* stats = nullptr);

/// Tracks variable name assignments for a single output file during lowering.
/// Used to map MLIR `Value`s to their generated source names.
struct FileContext {
  llvm::DenseMap<mlir::Value, std::string> vars;
  size_t next = 0;

  /// Look up the generated name for an already-defined value.
  std::string use(mlir::Value value) const {
    auto it = vars.find(value);
    if (it == vars.end()) {
      llvm::errs() << "Missing use: " << value << "\n";
      throw std::runtime_error("Missing use");
    }
    return it->second;
  }

  /// Assign a fresh generated name to `value` and return it.
  std::string def(mlir::Value value, const std::string& prefix = "x") {
    std::string name = prefix + std::to_string(next++);
    vars[value] = name;
    return name;
  }
};

/// Escape a string literal for inclusion in generated source.
std::string escapeString(llvm::StringRef str);

/// Command-line options shared by all circuit codegen binaries.
struct CodegenCLOptions {
  llvm::cl::opt<std::string> outputDir{"output-dir",
                                       llvm::cl::desc("Output directory"),
                                       llvm::cl::value_desc("dir"),
                                       llvm::cl::Required};
  llvm::cl::opt<size_t> validitySplitCount{
      "validity-split-count",
      llvm::cl::desc(
          "Split up validity polynomial into this many files to allow for parallel compilation"),
      llvm::cl::value_desc("numParts"),
      llvm::cl::init(1)};
};

extern llvm::ManagedStatic<CodegenCLOptions> codegenCLOptions;
/// Register codegen command-line options with LLVM's option parser.
void registerCodegenCLOptions();

} // namespace zirgen
