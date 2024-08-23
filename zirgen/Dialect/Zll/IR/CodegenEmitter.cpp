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

#include <cassert>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;
using namespace zirgen::Zll;

namespace cl = llvm::cl;
static cl::opt<size_t> inlineDepth("codegen-inline-depth",
                                   cl::desc("Maximum depth of generated calls to inline instead of "
                                            "assigning to a variable for readability"),
                                   cl::init(5));

namespace zirgen::codegen {

void CodegenEmitter::emitModule(mlir::ModuleOp moduleOp) {
  emitTypeDefs(moduleOp);

  for (auto& op : *moduleOp.getBody()) {
    if (op.hasTrait<CodegenSkipTrait>())
      continue;

    TypeSwitch<Operation*>(&op)
        .Case<ModuleOp>([&](ModuleOp op) { emitModule(op); })
        .Case<FunctionOpInterface>([&](FunctionOpInterface op) { emitFunc(op); })
        .Case<CodegenGlobalOpInterface>([&](CodegenGlobalOpInterface op) { op.emitGlobal(*this); })
        .Default([&](auto op) { emitStatement(op); });
  }
}

std::string CodegenEmitter::canonIdent(llvm::StringRef ident, IdentKind idt) {
  assert(!ident.empty());

  std::string id;
  id.reserve(ident.size());

  if (isdigit(ident[0])) {
    // Prefix digit with underscore.
    id.push_back('_');
  }
  for (char ch : ident) {
    // Special characters aren't supported by any of our output languages, so replace with
    // underscores.
    if (ch == '$' || ch == '@' || ch == ' ' || ch == ':' || ch == '<' || ch == '>' || ch == ',') {
      id.push_back('_');
    } else {
      id.push_back(ch);
    }
  }
  // Apply language-specific canonicalization.
  return opts.lang->canonIdent(id, idt);
}

void CodegenEmitter::resetValueNumbering() {
  nextVarId = 0;
}

void CodegenEmitter::emitFunc(FunctionOpInterface op) {
  if (op->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    resetValueNumbering();
  }
  auto body = op.getCallableRegion();

  emitTypeDefs(op);

  // Pick up any special names of arguments.
  DenseMap<Value, StringRef> argValueNames;
  if (auto opAsm = dyn_cast<OpAsmOpInterface>(op.getOperation())) {
    opAsm.getAsmBlockArgumentNames(*body,
                                   [&](Value v, StringRef name) { argValueNames[v] = name; });
  }

  llvm::SmallVector<CodegenIdent<IdentKind::Var>> argNames;
  for (auto arg : op.getArguments()) {
    StringRef baseName = argValueNames.lookup(arg);
    if (baseName.empty())
      baseName = "arg";
    argNames.push_back(getNewValueName(arg, baseName, /*owned=*/false));
  }

  opts.lang->emitFuncDefinition(
      *this, op.getNameAttr(), argNames, llvm::cast<FunctionType>(op.getFunctionType()), body);

  if (op->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    resetValueNumbering();
  }
}

// Operator API; called from operators to output their parts.
void CodegenEmitter::emitRegion(mlir::Region& region) {
  for (Block& block : region) {
    emitBlock(block);
  }
}

void CodegenEmitter::emitBlock(mlir::Block& block) {
  for (Operation& op : block) {
    emitStatement(&op);
  }
}

bool CodegenEmitter::inlineDepthLessThan(Operation* op, size_t n) {
  if (!n)
    return false;

  for (auto operand : op->getOperands()) {
    if (varNames.contains(operand))
      return true;

    Operation* definer = operand.getDefiningOp();
    if (!inlineDepthLessThan(definer, n - 1))
      return false;
  }
  return true;
}

bool CodegenEmitter::shouldInlineConstant(Operation* op) {
  if (op->hasTrait<CodegenNeverInlineOpTrait>())
    return false;

  if (op->hasTrait<CodegenAlwaysInlineOpTrait>())
    return true;

  if (!isPure(op) || op->getNumResults() != 1)
    return false;

  // Rust only allows mutable references to be borrowed once at a
  // time.
  if (llvm::any_of(op->getOperandTypes(), [](Type operandType) {
        return operandType.hasTrait<CodegenPassByMutRefTypeTrait>();
      }))
    return false;

  if (op->hasOneUse() && inlineDepthLessThan(op, inlineDepth)) {
    return true;
  }

  if (!op->hasTrait<OpTrait::ConstantLike>())
    return false;

  // If walking this type finds any types other than the type itself,
  // it's likely complicated, so we want to put it in its own
  // definition instead of of inlining it everywhere.
  Type type = op->getResult(0).getType();
  auto walkResult = type.walk([&](Type t) {
    if (t != type)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return false;

  return true;
}

namespace {

Location stripColumn(Location loc) {
  if (auto fileLineCol = llvm::dyn_cast<FileLineColLoc>(loc)) {
    // Strip out the column number since we don't display it anyways
    return FileLineColLoc::get(fileLineCol.getFilename(), fileLineCol.getLine(), 0);
  }
  return loc;
}

void getCallStack(SmallVector<Location>& calls, Location loc) {
  if (auto callSite = llvm::dyn_cast<CallSiteLoc>(loc)) {
    getCallStack(calls, callSite.getCallee());
    getCallStack(calls, callSite.getCaller());
  } else if (auto nameLoc = llvm::dyn_cast<NameLoc>(loc)) {
    calls.push_back(NameLoc::get(nameLoc.getName(), stripColumn(nameLoc.getChildLoc())));
  } else if (llvm::isa<FileLineColLoc>(loc)) {
    calls.push_back(stripColumn(loc));
  } else if (!llvm::isa<UnknownLoc>(loc)) {
    llvm::errs() << "UNKNOWN location " << loc << "\n";
    calls.push_back(loc);
  }
}

} // namespace

void CodegenEmitter::emitLoc(Location loc) {
  SmallVector<Location> calls;
  getCallStack(calls, loc);

  if (calls.empty())
    return;

  for (Location callLoc : calls) {
    if (!currentLocations.contains(callLoc)) {
      *this << "// " << EmitPart(getLocString(callLoc)) << "\n";
    }
  }

  currentLocations.clear();
  currentLocations.insert(calls.begin(), calls.end());
}

void CodegenEmitter::emitStatement(Operation* op) {
  if (op->hasTrait<CodegenSkipTrait>())
    return;

  // If it handles this case specially, let it do its thing.
  if (auto statementOp = dyn_cast<CodegenStatementOpInterface>(op)) {
    statementOp.emitStatement(*this);
    return;
  }

  if (shouldInlineConstant(op))
    // Don't assign to a variable; instead, inline at point of use.
    return;

  if (op->hasTrait<OpTrait::ReturnLike>()) {
    opts.lang->emitReturn(
        *this,
        llvm::to_vector_of<CodegenValue>(llvm::map_range(
            op->getOperands(), [&](auto operand) { return CodegenValue(operand).owned(); })));
    return;
  }

  emitLoc(op->getLoc());

  emitSaveResults(op->getResults(), [&]() { emitExpr(op); });
}

void CodegenEmitter::emitSaveResults(ValueRange results, EmitPart expr) {
  auto newNames =
      llvm::to_vector(llvm::map_range(results, [&](Value v) { return getNewValueName(v); }));
  opts.lang->emitSaveResults(*this,
                             llvm::to_vector_of<CodegenIdent<IdentKind::Var>>(newNames),
                             llvm::to_vector(results.getTypes()),
                             expr);
}

void CodegenEmitter::emitValue(CodegenValue val) {
  if (val.value) {
    // If this comes from a mlir::Value, either reference the
    // previously emitted operation or emit it inline.
    if (varNames.contains(val.value)) {
      VarInfo& varInfo = varNames.getOrInsertDefault(val.value);

      if (varInfo.usesRemaining) {
        varInfo.usesRemaining--;
        if (val.needOwned && varInfo.usesRemaining) {
          // Clone instead of move if we're still going to need this value later.
          opts.lang->emitClone(*this, varInfo.ident);
          return;
        }
      }
      *this << varInfo.ident;
      return;
    }

    // We don't have this value stored in a variable; emit it inline.
    auto op = val.value.getDefiningOp();
    assert(op->hasTrait<CodegenAlwaysInlineOpTrait>() || (isPure(op) && op->getNumResults() == 1));
    assert(!op->hasTrait<CodegenNeverInlineOpTrait>());
    emitExpr(op);

  } else {
    // Otherwise, this is a constant and we can emit it as a literal.
    assert(val.attr);

    emitLiteral(val.getType(), val.attr);
  }
}

CodegenIdent<IdentKind::Var>
CodegenEmitter::getNewValueName(mlir::Value val, llvm::StringRef namePrefix, bool owned) {
  assert(val);
  CodegenIdent<IdentKind::Var> name =
      getStringAttr(llvm::formatv("{0}{1}", namePrefix, nextVarId++).str());

  VarInfo varInfo = {.ident = name};

  if (val.getType().hasTrait<CodegenNeedsCloneTypeTrait>()) {
    if (owned) {
      varInfo.usesRemaining = llvm::range_size(val.getUses());
    } else {
      // We didn't own this value to start with (e.g. an argument passed in as a reference),
      // so we can never std::move from it.
      varInfo.usesRemaining = std::numeric_limits<size_t>::max();
    }
  }

  bool didEmplace = varNames.try_emplace(val, varInfo).second;
  assert(didEmplace);
  return name;
}

void CodegenEmitter::emitExpr(Operation* op) {
  if (auto codegenOp = dyn_cast<CodegenExprOpInterface>(op)) {
    codegenOp.emitExpr(*this);
    return;
  }

  // Check if it's a function call to a callable, and if we have the target for it.
  if (auto callOp = dyn_cast<CallOpInterface>(op)) {
    if (auto callee = callOp.getCallableForCallee()) {
      if (auto calleeName =
              dyn_cast_if_present<FlatSymbolRefAttr>(dyn_cast<SymbolRefAttr>(callee))) {
        emitFuncCall(getStringAttr(calleeName.getValue()),
                     llvm::to_vector_of<CodegenValue>(op->getOperands()));
        return;
      }
    }
  }

  // Check for an op that supplies a constant.
  if (op->getNumResults() == 1) {
    auto result = op->getResult(0);
    Attribute constValue;
    if (matchPattern(result, m_Constant(&constValue))) {
      emitLiteral(result.getType(), constValue);
      return;
    }
  }

  // If all else fails and we have no internal reigons, emit as a function call.
  if (op->getRegions().empty()) {
    emitFuncCall(getStringAttr(op->getName().stripDialect()),
                 llvm::to_vector_of<CodegenValue>(op->getOperands()));
    return;
  }

  llvm::errs() << "Don't know how to codegen operation " << *op << "\n";
  throw(std::runtime_error("Attempted to translate unknown operation"));
}

void CodegenEmitter::emitLiteral(mlir::Type ty, mlir::Attribute value) {
  if (auto codegenType = dyn_cast<CodegenTypeInterface>(ty)) {
    if (succeeded(codegenType.emitLiteral(*this, value)))
      return;
  }
  opts.lang->fallbackEmitLiteral(*this, ty, value);
}

void CodegenEmitter::emitConstDef(CodegenIdent<IdentKind::Const> name, CodegenValue value) {
  opts.lang->emitSaveConst(*this, name, value);
}

void CodegenEmitter::emitConditional(CodegenValue condition, mlir::Region& region) {
  opts.lang->emitConditional(*this, condition, [&]() { emitRegion(region); });
}

void CodegenEmitter::emitFuncCall(CodegenIdent<IdentKind::Func> callee,
                                  llvm::ArrayRef<CodegenValue> args) {
  opts.lang->emitCall(*this, callee, /*contextArgs=*/{}, args);
}

void CodegenEmitter::emitFuncCall(CodegenIdent<IdentKind::Func> callee,
                                  llvm::ArrayRef<llvm::StringRef> contextArgs,
                                  llvm::ArrayRef<CodegenValue> args) {
  opts.lang->emitCall(*this, callee, contextArgs, args);
}

void CodegenEmitter::emitStructDef(mlir::Type ty,
                                   llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                   llvm::ArrayRef<mlir::Type> types) {
  opts.lang->emitStructDef(*this, ty, names, types);
}

void CodegenEmitter::emitLayoutDef(mlir::Type ty,
                                   llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                   llvm::ArrayRef<mlir::Type> types) {
  opts.lang->emitLayoutDef(*this, ty, names, types);
}

void CodegenEmitter::emitStructConstruct(mlir::Type ty,
                                         llvm::ArrayRef<CodegenIdent<IdentKind::Field>> names,
                                         llvm::ArrayRef<CodegenValue> values) {
  opts.lang->emitStructConstruct(*this, ty, names, values);
}

void CodegenEmitter::emitArrayDef(mlir::Type ty, mlir::Type elemType, size_t numElem) {
  opts.lang->emitArrayDef(*this, ty, elemType, numElem);
}

void CodegenEmitter::emitArrayConstruct(mlir::Type ty,
                                        mlir::Type elemType,
                                        llvm::ArrayRef<CodegenValue> elems) {
  opts.lang->emitArrayConstruct(*this, ty, elemType, elems);
}

void CodegenEmitter::emitMapConstruct(CodegenValue array,
                                      std::optional<CodegenValue> layout,
                                      mlir::Region& body) {
  auto argNames = llvm::to_vector_of<CodegenIdent<IdentKind::Var>>(
      llvm::map_range(body.getArguments(), [&](auto arg) { return getNewValueName(arg); }));
  opts.lang->emitMapConstruct(*this, array, layout, argNames, body);
}

void CodegenEmitter::emitReduceConstruct(CodegenValue array,
                                         CodegenValue init,
                                         std::optional<CodegenValue> layout,
                                         mlir::Region& body) {
  auto argNames = llvm::to_vector_of<CodegenIdent<IdentKind::Var>>(
      llvm::map_range(body.getArguments(), [&](auto arg) { return getNewValueName(arg); }));
  opts.lang->emitReduceConstruct(*this, array, init, layout, argNames, body);
}

void CodegenEmitter::emitSwitchStatement(mlir::Value result,
                                         llvm::ArrayRef<CodegenValue> selector,
                                         llvm::ArrayRef<mlir::Block*> arms) {
  SmallVector<EmitArmPartFunc> emitArms;
  for (Block* arm : arms) {
    emitArms.push_back([=]() {
      for (auto& op : arm->without_terminator())
        emitStatement(&op);
      Operation* termOp = arm->getTerminator();
      assert(termOp->hasTrait<OpTrait::ReturnLike>());
      assert(termOp->getOperands().size() == 1);
      return CodegenValue(termOp->getOperands()[0]);
    });
  }
  opts.lang->emitSwitchStatement(
      *this, getNewValueName(result), result.getType(), selector, emitArms);
}

void CodegenEmitter::emitTypeDefs(Operation* op) {
  op->walk([&](Operation* subOp) {
    emitTypeDefs(subOp->getOperandTypes());
    emitTypeDefs(subOp->getResultTypes());
    for (auto attr : subOp->getAttrs()) {
      if (auto tyAttr = dyn_cast<TypeAttr>(attr.getValue()))
        emitTypeDefs(tyAttr.getValue());
    }
  });
}

void CodegenEmitter::emitTypeDefs(TypeRange tys) {
  for (auto genericTy : tys) {
    genericTy.walk([&](CodegenTypeInterface ty) {
      if (types.count(ty)) {
        return;
      }

      auto name = getTypeName(ty);
      if (auto oldTy = typeNames.lookup(name.getAttr())) {
        if (!ty.allowDuplicateTypeNames()) {
          llvm::errs() << "Duplicate type name " << name.strref() << ":\n"
                       << "New:" << ty << "\n"
                       << "Old:" << oldTy << "\n";
        }
      } else {
        typeNames[name.getAttr()] = ty;
      }

      ty.emitTypeDefinition(*this);
      types.insert(ty);
    });
  }
}

CodegenIdent<IdentKind::Type> CodegenEmitter::getTypeName(Type ty) {
  if (auto codegenType = dyn_cast<CodegenTypeInterface>(ty)) {
    // Store in a StringAttr to persist the storage of this generated type name.
    return codegenType.getTypeName(*this);
  } else {
    std::string typeName;
    llvm::raw_string_ostream stream(typeName);

    ty.print(stream);

    return getStringAttr(typeName);
  }
}

void CodegenEmitter::emitEscapedString(llvm::StringRef str) {
  *this << "\"";
  printEscapedString(str, *outStream);
  *this << "\"";
}

void CodegenEmitter::emitTakeReference(EmitPart emitTarget) {
  opts.lang->emitTakeReference(*this, emitTarget);
}

void CodegenEmitter::emitInfix(llvm::StringRef infixOp, CodegenValue lhs, CodegenValue rhs) {
  *this << "(" << lhs << " " << EmitPart(infixOp) << " " << rhs << ")";
}

StringAttr CodegenEmitter::getStringAttr(llvm::StringRef str) {
  return StringAttr::get(ctx, str);
}

void CodegenEmitter::emitInvokeMacro(CodegenIdent<IdentKind::Macro> name,
                                     llvm::ArrayRef<EmitPart> emitArgs) {
  opts.lang->emitInvokeMacro(*this, name, /*contextArgs=*/{}, emitArgs);
}

void CodegenEmitter::emitInvokeMacro(CodegenIdent<IdentKind::Macro> name,
                                     llvm::ArrayRef<llvm::StringRef> contextArgs,
                                     llvm::ArrayRef<EmitPart> emitArgs) {
  opts.lang->emitInvokeMacro(*this, name, contextArgs, emitArgs);
}

void LanguageSyntax::emitClone(CodegenEmitter& cg, CodegenIdent<IdentKind::Var> value) {
  // Do nothing special by default.
  cg << value;
}

void LanguageSyntax::emitTakeReference(CodegenEmitter& cg, EmitPart emitTarget) {
  // Do nothing special by default.
  cg << emitTarget;
}

LogicalResult translateCodegen(Operation* op, CodegenOptions opts, llvm::raw_ostream& os) {
  CodegenEmitter emitter(opts, &os, op->getContext());
  TypeSwitch<Operation*>(op)
      .Case<ModuleOp>([&](ModuleOp op) { emitter.emitModule(op); })
      .Case<FunctionOpInterface>([&](FunctionOpInterface op) { emitter.emitFunc(op); })
      .Default([](auto op) {
        llvm::errs() << "Don't know how to translate " << *op << " for codegen\n";
        throw(std::runtime_error("Attempted to translate unknown operation"));
      });
  return success();
}

namespace {

template <typename OpType, char infixOp>
struct InfixInterface
    : public CodegenExprOpInterface::ExternalModel<InfixInterface<OpType, infixOp>, OpType> {
  void emitExpr(Operation* op, zirgen::codegen::CodegenEmitter& cg) const {
    static const char infixStr[] = {infixOp, '\0'};
    assert(op->getNumOperands() == 2);
    assert(op->getNumResults() == 1);
    cg.emitInfix(infixStr, op->getOperand(0), op->getOperand(1));
  }
};

} // namespace

} // namespace zirgen::codegen
