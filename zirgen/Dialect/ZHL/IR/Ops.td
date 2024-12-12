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
include "zirgen/Dialect/ZHL/IR/Dialect.td"
include "zirgen/Dialect/ZHL/IR/Types.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def ReturnsExpr : DeclareOpInterfaceMethods<InferTypeOpInterface>;

class ZhlOp<string mnemonic, list<Trait> traits = []>
    : Op<ZhlDialect, mnemonic, traits> {
}

def TypeParamOp : ZhlOp<"generic", [ReturnsExpr]> {
  let summary = "Generic type template parameter value";
  let arguments = (ins StrAttr:$name, UI32Attr:$index, Expr:$type);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $name `(` $index `)` `:` $type attr-dict
  }];
}

def ConstructorParamOp : ZhlOp<"parameter", [ReturnsExpr]> {
  let summary = "Constructor function parameter value";
  let arguments = (ins StrAttr:$name, UI32Attr:$index, Expr:$type, UnitAttr:$variadic);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $name `(` $index `)` `:` $type attr-dict
  }];
}

def ComponentOp : ZhlOp<"component"> {
  let summary = "Component declaration";
  let arguments = (ins SymbolNameAttr:$name, UnitAttr:$function, UnitAttr:$generic);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $name attr-dict-with-keyword $body
  }];
}

def ExternOp : ZhlOp<"extern"> {
  let summary = "Execute an function external to the DSL";
  let arguments = (ins StrAttr:$name, Expr:$returnType, Variadic<Expr>:$args);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
   $name attr-dict-with-keyword  `(` $args `)` `:` $returnType
  }];
}

def LiteralOp : ZhlOp<"literal", [ReturnsExpr]> {
  let summary = "Numeric constant";
  let arguments = (ins UI64Attr:$value);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $value attr-dict
  }];
}

def StringOp : ZhlOp<"string", [ReturnsExpr]> {
  let summary = "string constant";
  let arguments = (ins StrAttr:$value);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $value attr-dict
  }];
}

def GlobalOp : ZhlOp<"global", [ReturnsExpr]> {
  let summary = "Reference to implicit global symbol";
  let arguments = (ins StrAttr:$name);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $name attr-dict
  }];
}

def LookupOp: ZhlOp<"lookup", [ReturnsExpr]> {
  let summary = "Member of component";
  let arguments = (ins Expr:$component, StrAttr:$member);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $component `,` $member attr-dict
  }];
}

def SubscriptOp: ZhlOp<"subscript", [ReturnsExpr]> {
  let summary = "Element of array";
  let arguments = (ins Expr:$array, Expr:$element);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $array `[` $element `]` attr-dict
  }];
}

def SpecializeOp: ZhlOp<"specialize", [ReturnsExpr]> {
  let summary = "Apply type parameers";
  let arguments = (ins Expr:$type, Variadic<Expr>:$args);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $type `<` $args `>` attr-dict
  }];
}

def ConstructOp: ZhlOp<"construct", [ReturnsExpr]> {
  let summary = "Construct component";
  let arguments = (ins Expr:$type, Variadic<Expr>:$args);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $type `(` $args `)` attr-dict
  }];
}

def ConstructGlobalOp: ZhlOp<"construct_global"> {
  let summary = "Constructs a global component";
  let arguments = (ins StrAttr:$name, Expr:$constructType, Variadic<Expr>:$args);
  let assemblyFormat = [{
    $name `:` `` `=` $constructType `(` $args `)` attr-dict
  }];
}

def GetGlobalOp: ZhlOp<"get_global", [ReturnsExpr]> {
  let summary = "Returns the value of a global component";
  let arguments = (ins StrAttr:$name, Expr:$constructType);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $name `:` $constructType attr-dict
  }];
}

def BlockOp: ZhlOp<"block", [ReturnsExpr]> {
  let summary = "Block scope with statements";
  let regions = (region SizedRegion<1>:$inner);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    attr-dict-with-keyword $inner
  }];
}

def MapOp : ZhlOp<"map", [ReturnsExpr]> {
  let summary = "Apply function to elements of array";
  let arguments = (ins Expr:$array);
  let results = (outs Expr:$out);
  let regions = (region SizedRegion<1>:$function);
  let assemblyFormat = [{
    $array attr-dict-with-keyword $function
  }];
}

def ReduceOp: ZhlOp<"reduce", [ReturnsExpr]> {
  let summary = "Reduce array elements";
  let arguments = (ins Expr:$array, Expr:$init, Expr:$type);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $array `,` $init `,` $type attr-dict
  }];
}

def SwitchOp: ZhlOp<"switch", [ReturnsExpr]> {
  let summary = "Evaluate expression array based on selector value";
  let arguments = (ins Expr:$selector);
  let regions = (region VariadicRegion<SizedRegion<1>>:$cases);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $selector `->` $cases attr-dict
  }];
}

def RangeOp: ZhlOp<"range", [ReturnsExpr]> {
  let summary = "Create an array spanning a range";
  let arguments = (ins Expr:$start, Expr:$end);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $start `,` $end attr-dict
  }];
}

def BackOp: ZhlOp<"back", [ReturnsExpr]> {
  let summary = "Retrieve value from previous iteration";
  let arguments = (ins Expr:$distance, Expr:$target);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $distance `,` $target attr-dict-with-keyword
  }];
}

def ArrayOp: ZhlOp<"array", [ReturnsExpr]> {
  let summary = "Create an array from a list of values";
  let arguments = (ins Variadic<Expr>:$elements);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    `[` $elements `]` attr-dict
  }];
}

def DefinitionOp: ZhlOp<"define"> {
  let summary = "define member";
  let arguments = (ins Expr:$declaration, Expr:$definition);
  let assemblyFormat = [{
    $declaration `=` $definition attr-dict
  }];
}

def DeclarationOp: ZhlOp<"declare", [ReturnsExpr]> {
  let summary = "declare member";
  let arguments = (ins StrAttr:$member, BoolAttr:$isPublic, Optional<Expr>:$typeExp);
  let results = (outs Expr:$out);
  let assemblyFormat = [{
    $member (`:` $typeExp^)? attr-dict
  }];
}

def ConstraintOp: ZhlOp<"constrain"> {
  let summary = "Declare constraint";
  let arguments = (ins Expr:$lhs, Expr:$rhs);
  let assemblyFormat = [{
    $lhs `=` $rhs attr-dict
  }];
}

def DirectiveOp: ZhlOp<"directive"> {
  let summary = "Compiler directive";
  let arguments = (ins StrAttr:$name, Variadic<Expr>:$args);
  let assemblyFormat = [{
    $name `(` $args `)` attr-dict
  }];
}

def SuperOp: ZhlOp<"super", [Terminator]> {
  let summary = "Super";
  let arguments = (ins Expr:$value);
  let assemblyFormat = [{
    $value attr-dict
  }];
}
