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

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

#ifndef ZLL_INTERFACES
#define ZLL_INTERFACES

class ZirgenNativeOpTrait<string name, list<Trait> traits = [],
                    code extraOpDeclaration = [{}],
                    code extraOpDefinition = [{}]> :
      NativeOpTrait<name, traits, extraOpDeclaration, extraOpDefinition> {
  let cppNamespace = "::zirgen::Zll";
}

class ZirgenNativeTypeTrait<string name,
                    code extraTypeDeclaration = [{}],
                    code extraTypeDefinition = [{}]> :
      NativeTypeTrait<name, extraTypeDeclaration, extraTypeDefinition> {
  let cppNamespace = "::zirgen::Zll";
}


def EvalOp : OpInterface<"EvalOp"> {
  let cppNamespace = "::zirgen::Zll";
  let methods = [
    InterfaceMethod<"Generate a function to evaluate this operation in the interpreter",
      "zirgen::Zll::OpEvaluator*", "getOpEvaluator",
      (ins "::zirgen::Zll::Interpreter*":$interp,
           "llvm::ArrayRef<::zirgen::Zll::InterpVal*>":$outputs,
           "llvm::ArrayRef<const ::zirgen::Zll::InterpVal*>":$inputs)>];
}

// Provides an EvalOp interface using an EvalAdaptor which prebinds values to InterpVals
// so they don't have to be looked up each time the operation is executed.
def EvalOpAdaptor : TraitList<[
   EvalOp,
   ZirgenNativeOpTrait<
      /*name=*/"EvalOpAdaptor",
      /*traits=*/[],
      /*extraOpDeclaration=*/[{
        using EvalAdaptor = GenericAdaptor<::llvm::ArrayRef<const zirgen::Zll::InterpVal*>>;
        ::mlir::LogicalResult
        evaluate(::zirgen::Zll::Interpreter& interp, llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                 EvalAdaptor& adaptor);
        zirgen::Zll::OpEvaluator*
        getOpEvaluator(
                          ::zirgen::Zll::Interpreter* interp,
                          llvm::ArrayRef<zirgen::Zll::InterpVal*> outputs,
                          llvm::ArrayRef<const zirgen::Zll::InterpVal*> operands);
      }],
      /*extraOpDefinition=*/[{
         zirgen::Zll::OpEvaluator* $cppClass::getOpEvaluator(
                          ::zirgen::Zll::Interpreter* interp,
                          llvm::ArrayRef<zirgen::Zll::InterpVal*> outputs,
                          llvm::ArrayRef<const zirgen::Zll::InterpVal*> operands) {
          return interp->getEvaluatorForEvalOp(*this, outputs, operands);
        }
      }]>]>;

// Extends EvalOpAdaptor to also provide an ExtensionField based on
// the output type of this operation.
def EvalOpFieldAdaptor : TraitList<[
   EvalOp,
   ZirgenNativeOpTrait<
      /*name=*/"EvalOpFieldAdaptor",
      /*traits=*/[],
      /*extraOpDeclaration=*/[{
        using EvalAdaptor = GenericAdaptor<::llvm::ArrayRef<const zirgen::Zll::InterpVal*>>;
        ::mlir::LogicalResult
        evaluate(::zirgen::Zll::Interpreter& interp, llvm::ArrayRef<zirgen::Zll::InterpVal*> outs,
                 EvalAdaptor& adaptor, ExtensionField& field);
        zirgen::Zll::OpEvaluator*
        getOpEvaluator(
                          ::zirgen::Zll::Interpreter* interp,
                          llvm::ArrayRef<zirgen::Zll::InterpVal*> outputs,
                          llvm::ArrayRef<const zirgen::Zll::InterpVal*> operands);
      }],
      /*extraOpDefinition=*/[{
         zirgen::Zll::OpEvaluator* $cppClass::getOpEvaluator(
                          ::zirgen::Zll::Interpreter* interp,
                          llvm::ArrayRef<zirgen::Zll::InterpVal*> outputs,
                          llvm::ArrayRef<const zirgen::Zll::InterpVal*> operands) {
          return interp->getEvaluatorForEvalOp(*this, outputs, operands,
                          llvm::cast<ValType>(getOperation()->getResult(0).getType()).getExtensionField());
        }
      }]>]>;

def PolyOp : OpInterface<"PolyOp"> {
  let cppNamespace = "::zirgen::Zll";
}

def ReduceOpInterface : OpInterface<"ReduceOp"> {
  let description = "An operation that cares about reductions when lowering to a large field.";
  let cppNamespace = "::zirgen::Zll";
  let methods = [
    InterfaceMethod<"Update the ranges of this ops outputs from its inputs ranges, or return false if invalid",
      "bool", "updateRanges", (ins "::mlir::DenseMap<::mlir::Value, ::zirgen::Zll::BigIntRange>&":$ranges)
    >,
  ];
}

def IsReduce : DeclareOpInterfaceMethods<ReduceOpInterface>;

// Interface for operations which can codegen in the global context, e.g. a function
// or global constant.
def CodegenGlobalOpInterface : OpInterface<"CodegenGlobalOpInterface"> {
  let description = "An operation that can get codegenned in the global namespace, like a function or constant.";
  let cppNamespace = "::zirgen::Zll";
  let methods = [
    InterfaceMethod<"Write out this operation definition.",
      "void", "emitGlobal", (ins "zirgen::codegen::CodegenEmitter&": $cgs)>,
    InterfaceMethod<"Write out a forward declaration for this operation.",
      "void", "emitGlobalDecl", (ins "zirgen::codegen::CodegenEmitter&": $cgs)>
  ];
}

// Interface for operations which can codegen as a complete statement, and do not need
// any result handling (e.g. having their results assigned to variables).
def CodegenStatementOpInterface : OpInterface<"CodegenStatementOpInterface"> {
  let description = "An operation which has special handling to emit itself within a block.";
  let cppNamespace = "::zirgen::Zll";
  let methods = [
    InterfaceMethod<"Output this statement.",
      "void", "emitStatement", (ins "zirgen::codegen::CodegenEmitter&": $cg)>
   ];
}

// Interface for operations which codegen as an expression evaluation.  The generated
// expression should return results idiomatically to the target language;
// see CodegenEmitter::emitExpr for details.
def CodegenExprOpInterface : OpInterface<"CodegenExprOpInterface"> {
  let description = "An operation that has special handling to codegen in an expression context";
  let cppNamespace = "::zirgen::Zll";
  let methods = [
    InterfaceMethod<"Output an expression to calculate this operation's result(s)",
      "void", "emitExpr", (ins "zirgen::codegen::CodegenEmitter&": $cg)>
   ];
}

// An operation that gets codegenned as an infix operator, where infixOp is the
// operator.
class CodegenInfixOp<string infixOp> : TraitList<[
   CodegenExprOpInterface,
   PredOpTrait<"Has 2 operands", CPred<"$_op.getNumOperands() == 2">>,
   PredOpTrait<"Has a single result", CPred<"$_op.getNumResults() == 1">>,
   ZirgenNativeOpTrait<
      /*name=*/"CodegenInfixOpTrait",
      /*traits=*/[],
      /*extraOpDeclaration=*/[{
        void emitExpr(zirgen::codegen::CodegenEmitter& cg);
      }],
      /*extraOpDefinition=*/[{
        void $cppClass::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
          cg.emitInfix(}] # "\"" # infixOp # "\"" # [{ , getOperand(0), getOperand(1));
        }
      }]>
   ]>;


// An operation that codegens as a function call, but has additional
// properties that should be included in the function call arguments.
//
// Properties should be specified in CamelCase.
//
// TODO: Is there a way to convert the properties to camel case so we
// don't have to burden callers?
class CodegenOpWithProperties<list<string> properties> : TraitList<[
  CodegenExprOpInterface,
  ZirgenNativeOpTrait<
      /*name=*/"CodegenOpWithPropertiesTrait",
      /*traits=*/[],
      /*extraOpDeclaration=*/[{
        void emitExpr(zirgen::codegen::CodegenEmitter& cg);
      }],
      /*extraOpDefinition=*/[{
        void $cppClass::emitExpr(zirgen::codegen::CodegenEmitter& cg) {
          llvm::SmallVector<zirgen::codegen::CodegenValue> args = getOperation()->getOperands();
          }] # !interleave(!foreach(prop, properties, [{
             args.push_back(cg.guessAttributeType( }] # "get" # prop # "Attr" # [{ ()));
          }]), "") # [{
          cg.emitFuncCall(cg.getStringAttr((*this)->getName().stripDialect()), args);
        }
      }]>
  ]>;

// Skip this operation during codegen.
def CodegenSkip : ZirgenNativeOpTrait<"CodegenSkipTrait">;

// A codegen type that can be emitted.  By default, types are named as
// the type mnemonic, and no definitions are emitted.
def CodegenTypeInterface : TypeInterface<"CodegenTypeInterface"> {
  let description = "A type which has special handling when emitted in generated code";
  let cppNamespace = "::zirgen::Zll";
  let methods = [
    InterfaceMethod<"Emit the name of this type suitable for use in a declaration",
      "zirgen::codegen::CodegenIdent<zirgen::codegen::IdentKind::Type>",
      "getTypeName", (ins "zirgen::codegen::CodegenEmitter&": $cg),
      "",
      /*default=*/ "return cg.getStringAttr($_type.getMnemonic());">,
    InterfaceMethod<"Emit a type definition (e.g. `typedef') for this type if necessary",
      "void", "emitTypeDefinition", (ins "zirgen::codegen::CodegenEmitter&": $cg),
      "",
      /*default=*/ "return;">,
    InterfaceMethod<"Emit a constant literal value of this type",
      "mlir::LogicalResult", "emitLiteral",
      (ins "zirgen::codegen::CodegenEmitter&": $cg, "mlir::Attribute": $value),
      "",
      /*default=*/ "return mlir::failure();">,
    InterfaceMethod<"Allow multiple instances of this type to flatten to the same name",
      "bool", "allowDuplicateTypeNames",
      (ins),
      "",
      /*default=*/ "return false;">
   ];
}

// Always inline this operation at point of use instead of assigning
// its results to a local variable.
def CodegenAlwaysInlineOp : ZirgenNativeOpTrait<"CodegenAlwaysInlineOpTrait">;

// Never inline this operation at point of use.
def CodegenNeverInlineOp : ZirgenNativeOpTrait<"CodegenNeverInlineOpTrait">;

// True if this type is lacking copy semantics and needs an explicit
// call to e.g. .clone()
def CodegenNeedsCloneType : ZirgenNativeTypeTrait<"CodegenNeedsCloneTypeTrait">;

// True if we only pass this type around by reference, and never
// construct or copy it except in special cases.
def CodegenOnlyPassByReferenceType : ZirgenNativeTypeTrait<"CodegenOnlyPassByReferenceTypeTrait">;

// True if this type needs to be passed as a borrowed mutable in rust
// (C++ needs less special handling for these types).
def CodegenPassByMutRefType : ZirgenNativeTypeTrait<"CodegenPassByMutRefTypeTrait">;

// True if this type needs to be handled specially as a layout type.
def CodegenLayoutType : ZirgenNativeTypeTrait<"CodegenLayoutTypeTrait">;

#endif /* ZLL_INTERFACES */
