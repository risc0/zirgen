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
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "zirgen/Dialect/ZStruct/IR/Attrs.td"
include "zirgen/Dialect/ZStruct/IR/Dialect.td"
include "zirgen/Dialect/ZStruct/IR/Types.td"

class ZStructOp<string mnemonic, list<Trait> traits = []>
    : Op<ZStructDialect, mnemonic, traits> {
}

def LookupOp : ZStructOp<"lookup", [Pure,
                                    InferTypeOpAdaptor,
                                    DeclareOpInterfaceMethods<CodegenExprOpInterface>,
                                    DeclareOpInterfaceMethods<PolyOp>,
                                    CodegenAlwaysInlineOp]> {
  let summary = "Look up reference to struct or union member";
  let assemblyFormat = [{ $base `[` $member `]` `:` functional-type($base, $out) attr-dict }];
  let arguments = (ins AnyTypeOf<[StructType,LayoutType,UnionType]>:$base, StrAttr:$member);
  let results = (outs ZirType:$out);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def SubscriptOp : ZStructOp<"subscript", [NoMemoryEffect, // but not speculatable, sicne the index might not be in range.
                                          InferTypeOpAdaptor,
                                          DeclareOpInterfaceMethods<CodegenExprOpInterface>,
                                          CodegenAlwaysInlineOp,
                                          DeclareOpInterfaceMethods<PolyOp>]> {
  let summary = "Select array element by index";
  let assemblyFormat = [{ $base `[` type($index) $index `]` `:` functional-type($base, $out) attr-dict }];
  // TODO: Pick between argument types for the index and make everything
  // that uses it consistent.
  let arguments = (ins AnyTypeOf<[ArrayType,LayoutArrayType]>:$base, AnyTypeOf<[Index,Val]>:$index);
  let results = (outs ZirType:$out);
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    mlir::IntegerAttr getIndexAsAttr();
    size_t getIndexUpperBound();
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def LoadOp : ZStructOp<"load", [EvalOpAdaptor,
                                MemoryEffects<[MemRead]>,
                                DeclareOpInterfaceMethods<CodegenExprOpInterface>,
                                CodegenAlwaysInlineOp,
                                InferTypeOpAdaptor,
                                DeclareOpInterfaceMethods<PolyOp>]> {
  let summary = "Load value from a Ref";
  let description = [{
     With respect to extension elements, `ref' specifies how many elements should be read
     from the layout, and `out' specifies the type that the value should be coalesced to.
  }];
  let assemblyFormat = [{
      $ref `back` $distance `:` functional-type($ref, $out) attr-dict-with-keyword
  }];
  let arguments = (ins Ref:$ref, Index:$distance);
  let results = (outs Val:$out);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def StoreOp : ZStructOp<"store", [EvalOpAdaptor, MemoryEffects<[MemWrite]>, DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Store new value into a member reference";
  let assemblyFormat = [{
      $ref `,` $val `:` functional-type($val, $ref) attr-dict
  }];
  let arguments = (ins Ref:$ref, Val:$val);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def PackOp : ZStructOp<"pack", [Pure, DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Packs the operands into the members of a record";
  let arguments = (ins Variadic<ZirType>:$members);
  let results = (outs StructType:$out);
  let assemblyFormat = [{
     `(` ($members^ `:` type($members))? `)` `:` type($out) attr-dict-with-keyword
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def SwitchOp: ZStructOp<"switch", [RecursiveMemoryEffects, EvalOpAdaptor,
                                   DeclareOpInterfaceMethods<CodegenStatementOpInterface>]> {
  let summary = "Selects a mux arm and executes it";
  let description = [{
    `arguments' should be an array of selectors, exactly one of which should be non-zero.

    This operation executes the arm corresponding to the selector that's non-zero.
  }];
  let arguments = (ins Variadic<Val>:$selector);
  let regions = (region VariadicRegion<SizedRegion<1>>:$arms);
  let results = (outs ZirType:$out);
  let assemblyFormat = [{
    $selector `:` type($selector) `:` type($out) `->` $arms attr-dict
  }];
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def YieldOp : ZStructOp<"yield", [Terminator, ReturnLike, EvalOpAdaptor]> {
  let summary = "A terminator which returns a result from a block";
  let arguments = (ins ZirType:$value);
  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];
}

def ArrayOp: ZStructOp<"array", [Pure, SameTypeOperands, InferTypeOpAdaptor, DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Create an array from a list of values";
  let arguments = (ins Variadic<ZirType>:$elements);
  let results = (outs ArrayType:$out);
  let assemblyFormat = [{
    `[` $elements `:` type($elements) `]` attr-dict
  }];
  let hasFolder = 1;
}

def MapOp : ZStructOp<"map", [DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Apply function to elements of array";
  let arguments = (ins ArrayLikeTypeInterface:$array, Optional<LayoutArrayType>:$layout);
  let results = (outs ArrayType:$out);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $array `:` type($array) ( `,` $layout^  `:` type($layout) )? `->` type($out) $body attr-dict
  }];
  let hasRegionVerifier = 1;
}

def ReduceOp: ZStructOp<"reduce", [DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Reduce array elements";
  let arguments = (ins ArrayType:$array, ZirType:$init, Optional<LayoutArrayType>:$layout);
  let results = (outs ZirType:$out);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $array `:` type($array) `,` $init `:` type($init) (`,` $layout^ `:` type($layout))? `->` type($out) $body attr-dict
  }];
  let hasRegionVerifier = 1;
}

def GlobalConstOp : ZStructOp<"global_const", [DeclareOpInterfaceMethods<CodegenGlobalOpInterface>]> {
  let summary = "Defines a global constant symbol and its value as an attribute.";
  let arguments = (ins
      SymbolNameAttr:$sym_name,
      OptionalAttr<StrAttr>:$sym_visibility,
      TypeAttr:$type,
      AnyAttr:$constant);
  let builders = [
    OpBuilder<
      (ins "mlir::StringRef":$name, "mlir::Type":$type, "mlir::Attribute":$constant), [{
        $_state.addAttribute("sym_name", $_builder.getStringAttr(name));
        $_state.addAttribute("sym_visibility", $_builder.getStringAttr("public"));
        $_state.addAttribute("type", mlir::TypeAttr::get(type));
        $_state.addAttribute("constant", constant);
      }]
    >
  ];
  let assemblyFormat = [{ $sym_name `:` $type `=` $constant attr-dict-with-keyword }];
  let extraClassDeclaration = [{
    bool isOptionalSymbol() { return true; }
  }];
}

def BindLayoutOp : ZStructOp<"bind_layout", [
    EvalOpAdaptor,
    Pure,
    CodegenNeverInlineOp,
    DeclareOpInterfaceMethods<CodegenExprOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Binds a layout from a global constant to a buffer";
  let description = [{
    This interprets the given buffer with respect to the given layout and allows symbolic access
    to the registers contained therein.
  }];
  let arguments = (ins
      AnyAttrOf<[FlatSymbolRefAttr,StructAttr,RefAttr,ArrayAttr]>:$layout,
      Buffer:$buffer);
  let assemblyFormat = [{ $layout `->` type($out) `=` $buffer `:` type($buffer) attr-dict }];
  let results = (outs IsLayoutType:$out);
  let builders = [
      OpBuilder<
          (ins "GlobalConstOp":$constOp, "mlir::Value":$buffer), [{
              build($_builder, $_state, constOp.getType(), mlir::SymbolTable::getSymbolName(constOp), buffer);
          }]>];
}

def GetBufferOp : ZStructOp<"get_buffer", [
    Pure,
    EvalOpAdaptor,
    // Always inline this; otherwise we have to represent buffer types in generated code.
    CodegenAlwaysInlineOp,
    DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
    let summary = "Gets a buffer from the current context by name";
    let arguments = (ins StrAttr:$name);
    let assemblyFormat = [{ $name `:` type($out) attr-dict }];
    let results = (outs Buffer:$out);
}

def AliasLayoutOp : ZStructOp<"alias_layout", [SameTypeOperands, EvalOpAdaptor]> {
  let summary = "An operation that signals that the given layouts must alias each other";
  let description = [{
    This operation is used by the GenerateLayoutPass to ensure the two layout
    operands refer to the same buffer offsets. This should statically guarantee
    that the layouts alias each other, but it also functions as a runtime check
    in the interpreter.
  }];
  let arguments = (ins IsLayoutType:$lhs, IsLayoutType:$rhs);
}
