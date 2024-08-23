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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringSet.h"

namespace zirgen {

namespace codegen {

enum class IdentKind {
  Var,
  Type,
  Const,
  Field,
  Func,
  Macro,
};

enum class LanguageKind { Rust, Cpp };

class CodegenEmitter;

// A value that can be emitted: either a mlir::Value representing the output from
// a computation or an attribute and an associated type.
class CodegenValue {
public:
  CodegenValue(mlir::Value value) : type(value.getType()), value(value) {}
  CodegenValue(mlir::TypedAttr attr) : type(attr.getType()), attr(attr) {}
  CodegenValue(mlir::Type type, mlir::Attribute attr) : type(type), attr(attr) {}

  friend class CodegenEmitter;

  mlir::Type getType() const { return type; }
  mlir::Value getValue() const { return value; }

  // Produce a version of this value that is owned, i.e. is a copy or
  // can be moved out of its current storage.
  CodegenValue owned() const {
    CodegenValue ownedVal = *this;
    ownedVal.needOwned = true;
    return ownedVal;
  }

private:
  mlir::Type type;

  // Populated if this is a value
  mlir::Value value;

  // Populated if this is a literal constant.
  mlir::Attribute attr;

  // True if this expression needs to emit an owned value, false if if the target
  // expression can be implicily copied.
  bool needOwned = false;
};

// An uncanonicalized name of the given kind.  It will be
// canonicalized when written out to generated code.
template <IdentKind kind> class CodegenIdent {
public:
  CodegenIdent() = default;
  CodegenIdent(mlir::StringAttr strAttr, bool canonicalized = false)
      : strAttr(strAttr), canonicalized(canonicalized) {}

  llvm::StringRef strref() const { return strAttr.strref(); };
  std::string str() const { return strAttr.str(); };
  mlir::StringAttr getAttr() const { return strAttr; };
  bool isCanonicalized() const { return canonicalized; }

  explicit operator bool() const { return strAttr ? true : false; }

private:
  mlir::StringAttr strAttr;
  bool canonicalized;
};

struct EmitPart;
// Arm emitter returns CodegenValues for any values returned by the arm.
using EmitArmPartFunc = std::function<CodegenValue()>;

struct LanguageSyntax {
protected:
  LanguageSyntax() = default;

  llvm::SmallVector<std::string> contextArgDecls;

public:
  // Adds the given argument as a context arg to any functions generated, with the provided
  // definition.
  void addContextArgument(llvm::StringRef argDecl) { contextArgDecls.push_back(argDecl.str()); }

  virtual LanguageKind getLanguageKind() = 0;

  virtual std::string canonIdent(llvm::StringRef ident, IdentKind idt) = 0;

  virtual void emitClone(CodegenEmitter& cg, CodegenIdent<IdentKind::Var> value);
  virtual void emitTakeReference(CodegenEmitter& cg, EmitPart emitTarget);

  virtual void fallbackEmitLiteral(CodegenEmitter& cg, mlir::Type ty, mlir::Attribute value) = 0;
  virtual void emitFuncDefinition(CodegenEmitter& cg,
                                  CodegenIdent<IdentKind::Func> funcName,
                                  llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                  mlir::FunctionType funcType,
                                  mlir::Region* body) = 0;

  virtual void emitConditional(CodegenEmitter& cg, CodegenValue condition, EmitPart emitThen) = 0;

  virtual void emitSwitchStatement(CodegenEmitter& cg,
                                   CodegenIdent<IdentKind::Var> resultName,
                                   mlir::Type resultType,
                                   llvm::ArrayRef<CodegenValue> conditions,
                                   llvm::ArrayRef<EmitArmPartFunc> emitArms) = 0;

  virtual void emitReturn(CodegenEmitter& cg, llvm::ArrayRef<CodegenValue> values) = 0;

  virtual void emitSaveResults(CodegenEmitter& cg,
                               llvm::ArrayRef<CodegenIdent<IdentKind::Var>> names,
                               llvm::ArrayRef<mlir::Type> types,
                               EmitPart emitExpression) = 0;

  virtual void
  emitSaveConst(CodegenEmitter& cg, CodegenIdent<IdentKind::Const> name, CodegenValue value) = 0;

  virtual void emitCall(CodegenEmitter& cg,
                        CodegenIdent<IdentKind::Func> callee,
                        llvm::ArrayRef<llvm::StringRef> contextArgs,
                        llvm::ArrayRef<CodegenValue> args) = 0;

  virtual void emitInvokeMacro(CodegenEmitter& cg,
                               CodegenIdent<IdentKind::Macro> callee,
                               llvm::ArrayRef<llvm::StringRef> contextArgs,
                               llvm::ArrayRef<EmitPart> emitArgs) = 0;

  virtual void emitStructDef(CodegenEmitter& cg,
                             mlir::Type ty,
                             llvm::ArrayRef<CodegenIdent<IdentKind::Field>> fields,
                             llvm::ArrayRef<mlir::Type> types) {
    llvm::errs() << "Structure definitions are not available for this language syntax.\n";
    abort();
  }
  virtual void emitStructConstruct(CodegenEmitter& cg,
                                   mlir::Type ty,
                                   llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                   llvm::ArrayRef<CodegenValue> values) {
    llvm::errs() << "Structure constructions are not available for this language syntax.\n";
    abort();
  }
  virtual void
  emitArrayDef(CodegenEmitter& cg, mlir::Type ty, mlir::Type elemType, size_t numElems) {
    llvm::errs() << "Array definitions are not available for this language syntax.\n";
    abort();
  }
  virtual void emitArrayConstruct(CodegenEmitter& cg,
                                  mlir::Type ty,
                                  mlir::Type elemType,
                                  llvm::ArrayRef<CodegenValue> values) {
    llvm::errs() << "Array constructions are not available for this language syntax.\n";
    abort();
  }

  virtual void emitMapConstruct(CodegenEmitter& cg,
                                CodegenValue array,
                                std::optional<CodegenValue> layout,
                                llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                mlir::Region& body) {
    llvm::errs() << "Map constructions are not available for this language syntax.\n";
    abort();
  }

  virtual void emitReduceConstruct(CodegenEmitter& cg,
                                   CodegenValue array,
                                   CodegenValue init,
                                   std::optional<CodegenValue> layout,
                                   llvm::ArrayRef<CodegenIdent<IdentKind::Var>> argNames,
                                   mlir::Region& body) {
    llvm::errs() << "Reduce constructions are not available for this language syntax.\n";
    abort();
  }

  virtual void emitLayoutDef(CodegenEmitter& cg,
                             mlir::Type ty,
                             llvm::ArrayRef<CodegenIdent<IdentKind::Field>> fields,
                             llvm::ArrayRef<mlir::Type> types) {
    llvm::errs() << "Layouts are not available for this language syntax.\n";
    abort();
  }

  virtual ~LanguageSyntax() {}
};

struct CodegenOptions {
  LanguageSyntax* lang = nullptr;

  // If true, generate compatible layout code for zkp layout library.
  // TODO: Migrate to macro use so we don't have to have special cases.
  bool zkpLayoutCompat = false;
};

// Manages emitting generated code.
class CodegenEmitter {
public:
  CodegenEmitter(CodegenOptions opts, llvm::raw_ostream* os, mlir::MLIRContext* ctx)
      : opts(opts), outStream(os), ctx(ctx) {}

  void emitModule(mlir::ModuleOp op);
  void emitFunc(mlir::FunctionOpInterface op);
  void emitRegion(mlir::Region& region);
  void emitBlock(mlir::Block& block);

  // Emits a conditional statement.
  void emitConditional(CodegenValue condition, mlir::Region& region);

  // Emit a function call invocation.
  void emitFuncCall(CodegenIdent<IdentKind::Func> callee, llvm::ArrayRef<CodegenValue> args);
  void emitFuncCall(CodegenIdent<IdentKind::Func> callee,
                    llvm::ArrayRef<llvm::StringRef> contextArgs,
                    llvm::ArrayRef<CodegenValue> args);

  // Emits an expression executing an infix operation .
  void emitInfix(llvm::StringRef infixOp, CodegenValue lhs, CodegenValue rhs);

  // Emits a named definition for a constant value.
  void emitConstDef(CodegenIdent<IdentKind::Const> name, CodegenValue value);

  // Emits a named type definition for the specified structure
  void emitStructDef(mlir::Type ty,
                     llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                     llvm::ArrayRef<mlir::Type> types);

  // Emits a named type definition for the specified layout structure.
  void emitLayoutDef(mlir::Type ty,
                     llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                     llvm::ArrayRef<mlir::Type> types);

  // Emits an expression generating a constructed structure.
  void emitStructConstruct(mlir::Type ty,
                           llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                           llvm::ArrayRef<CodegenValue> values);

  // Emits a named type definition for the given array.
  void emitArrayDef(mlir::Type ty, mlir::Type elemType, size_t numElem);

  // Emits an expression generating a constructed array.
  void emitArrayConstruct(mlir::Type ty, mlir::Type elemType, llvm::ArrayRef<CodegenValue> elems);

  // Emits an expression which evaluates a map over an array.
  void emitMapConstruct(CodegenValue array, std::optional<CodegenValue> layout, mlir::Region& body);

  // Emits an expression which evaluates a reduction over an array.
  void emitReduceConstruct(CodegenValue array,
                           CodegenValue init,
                           std::optional<CodegenValue> layout,
                           mlir::Region& body);

  // Emits a block selection based on which of the selector values is nonzero, and
  // records the results as associated with the given values.
  void emitSwitchStatement(mlir::Value result,
                           llvm::ArrayRef<CodegenValue> selector,
                           llvm::ArrayRef<mlir::Block*> arms);

  // Emits a macro invokation (e.g. #define)
  template <typename... T> void emitInvokeMacroV(CodegenIdent<IdentKind::Macro> name, T... arg);

  void emitInvokeMacro(CodegenIdent<IdentKind::Macro> name,
                       llvm::ArrayRef<llvm::StringRef> contextArgs,
                       llvm::ArrayRef<EmitPart> emitArgs);
  void emitInvokeMacro(CodegenIdent<IdentKind::Macro> name, llvm::ArrayRef<EmitPart> emitArgs);

  LanguageKind getLanguageKind() { return getLang()->getLanguageKind(); }
  const CodegenOptions& getOpts() { return opts; }
  LanguageSyntax* getLang() { return opts.lang; }

  CodegenIdent<IdentKind::Type> getTypeName(mlir::Type ty);

  // Inside of a block, emit code to execute `op' if appropriate and save its results.
  // Sometimes this operation might be skiped, for instance:
  // * It is ConstantLike and Pure and gets inlined at point of use
  // * It has the CodegenSkip trait.
  void emitStatement(mlir::Operation* op);

  // Saves the results generated by `expr' to the given values to be
  // available to be referenced later.
  void emitSaveResults(mlir::ValueRange results, EmitPart expr);

  // Emits a comment describing the given location
  void emitLoc(mlir::Location loc);

  // Emit code to execute 'op'; this code should return the same
  // number of values as op has results, idiomatically for the target language.
  // For instance, in C++, a 0-result operation is expected to return "void",
  // a 1-result operation is expected to return a single type, and
  // a 2 or more result operation should return a tuple-like type.
  void emitExpr(mlir::Operation* op);

  // Guesses the Type of the given attribute.  This isn't reliable,
  // and an explicit type should be specified when possible.
  CodegenValue guessAttributeType(mlir::IntegerAttr attr) {
    return CodegenValue(mlir::IndexType::get(attr.getContext()), attr);
  }

  void emitEscapedString(llvm::StringRef str);

  void emitTakeReference(EmitPart emitTarget);

  // Emits a value, either by referencing a variable containing a
  // previously calculated result, or by inlining the calculation/materialization.
  void emitValue(CodegenValue val);

  // Emits type definitions for any types used within op.
  void emitTypeDefs(mlir::Operation* op);

  CodegenEmitter& operator<<(EmitPart emitPart);

  llvm::raw_ostream* getOutputStream() const { return outStream; }

  mlir::StringAttr getStringAttr(llvm::StringRef str);

  // llvm::interleave turns the separator into into a StringRef, and
  // we'd prefer not to implicitly let StringRefs be emitted without
  // explicitly specifying what they are.
  template <typename Container,
            typename UnaryFunctor,
            typename T = llvm::detail::ValueOfRange<Container>>
  void interleaveComma(const Container& c, UnaryFunctor each_fn);
  template <typename Container, typename T = llvm::detail::ValueOfRange<Container>>
  void interleaveComma(const Container& c);

private:
  friend struct EmitPart;

  void emitLiteral(mlir::Type ty, mlir::Attribute value);

  CodegenIdent<IdentKind::Var>
  getNewValueName(mlir::Value val, llvm::StringRef namePrefix = "x", bool owned = true);
  void resetValueNumbering();

  std::string canonIdent(llvm::StringRef ident, IdentKind idt);

  void emitTypeDefs(mlir::TypeRange vals);

  // Returns true if emitting the given operation would generate less than n depth of inlining.
  bool inlineDepthLessThan(mlir::Operation* op, size_t n);
  bool shouldInlineConstant(mlir::Operation* op);

  CodegenOptions opts;

  size_t nextVarId = 0;
  struct VarInfo {
    CodegenIdent<IdentKind::Var> ident;
    // Number of uses needed of this variable that we haven't
    // processed yet that might potentially need an owned copy.
    size_t usesRemaining = 0;
  };
  llvm::DenseMap<mlir::Value, VarInfo> varNames;
  llvm::DenseSet<mlir::Type> types;
  llvm::DenseMap<mlir::StringAttr, mlir::Type> typeNames;
  llvm::raw_ostream* outStream = nullptr;
  mlir::MLIRContext* ctx = nullptr;

  llvm::DenseSet<mlir::Location> currentLocations;
};

// A piece of generated code that can be emitted through
// CodegenEmitter.  This class allows implicit construction with a
// number of types which we know are safe to codegen,
// specifically:
//
// * functions, either with a CodegenEmitter argument or not
// * Integers
// * String literals
// * CodegenValues
// * CodegenIdents
//
// It also allows explicit construction from a StringRef.  We don't allow
// implicit StringRef construction so that we don't accidentally emit something
// that should be a CodegenIdent.
struct EmitPart {
public:
  EmitPart() = delete;

  // Any callable that's convertible to a void() void(CodegenEmitter&)
  template <typename T,
            std::enable_if_t<std::is_convertible_v<T, std::function<void()>>, bool> = true>
  EmitPart(const T& f) : emitFunc([f](CodegenEmitter& cg) { f(); }) {}
  template <
      typename T,
      std::enable_if_t<std::is_convertible_v<T, std::function<void(CodegenEmitter&)>>, bool> = true>
  EmitPart(const T& f) : emitFunc(f) {}

  // Implicit emissions for simple types.  We support fewer types than e.g. llvm::raw_ostream
  // so that we don't accidentally emit non-code strings.
  template <IdentKind kind>
  EmitPart(CodegenIdent<kind> ident)
      : emitFunc([ident](CodegenEmitter& cg) {
        std::string identStr =
            ident.isCanonicalized() ? ident.str() : cg.canonIdent(ident.strref(), kind);
        *cg.getOutputStream() << identStr;
      }) {}

  // Any integer type.
  template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
  EmitPart(T intVal)
      : emitFunc([intVal](CodegenEmitter& cg) { *cg.getOutputStream() << intVal; }) {}

  // String literals
  template <size_t N>
  EmitPart(const char (&str)[N])
      : emitFunc([str](CodegenEmitter& cg) { *cg.getOutputStream() << str; }){};

  // References to a generated value.
  EmitPart(CodegenValue val) : emitFunc([val](CodegenEmitter& cg) { cg.emitValue(val); }) {}

  // Anything else that's implicitly convertible to CodegenValue
  template <typename T, std::enable_if_t<std::is_convertible_v<T, CodegenValue>, bool> = true>
  EmitPart(const T& codegenValue) : EmitPart(CodegenValue(codegenValue)) {}

  // StringRefs must be explicitly converted so we don't accidentally
  // skip canonicalizing identifiers.
  explicit EmitPart(llvm::StringRef str)
      : emitFunc([str](CodegenEmitter& cg) { *cg.getOutputStream() << str; }){};

  void emit(CodegenEmitter& cg) { emitFunc(cg); }

private:
  std::function<void(CodegenEmitter&)> emitFunc;
};

template <typename... T>
void CodegenEmitter::emitInvokeMacroV(CodegenIdent<IdentKind::Macro> name, T... arg) {
  emitInvokeMacro(name, std::initializer_list<EmitPart>({arg...}));
}

inline CodegenEmitter& CodegenEmitter::operator<<(EmitPart emitPart) {
  emitPart.emit(*this);
  return *this;
}

template <typename Container, typename UnaryFunctor, typename T>
void CodegenEmitter::interleaveComma(const Container& c, UnaryFunctor each_fn) {
  llvm::interleave(
      c, *getOutputStream(), [&](const T& elem) { each_fn(elem); }, ", ");
}
template <typename Container, typename T> void CodegenEmitter::interleaveComma(const Container& c) {
  llvm::interleave(
      c,
      *getOutputStream(),
      [&](const T& elem) {
        EmitPart emitPart = elem;
        (*this) << emitPart;
      },
      ", ");
}

mlir::LogicalResult
translateCodegen(mlir::Operation* op, CodegenOptions opts, llvm::raw_ostream& os);

} // namespace codegen
} // namespace zirgen
