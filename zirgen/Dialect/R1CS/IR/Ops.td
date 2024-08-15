// Copyright (c) 2024 RISC Zero, Inc.

include "mlir/IR/OpBase.td"
include "zirgen/Dialect/R1CS/IR/Dialect.td"
include "zirgen/Dialect/R1CS/IR/Types.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class R1CSOp<string mnemonic, list<Trait> traits = []>
    : Op<R1CSDialect, mnemonic, traits> {
}

def DefOp : R1CSOp<"def", [Pure]> {
  let summary = "Define a wire and its label ID";
  let arguments = (ins UI64Attr:$label, BoolAttr:$isPublic);
  let results = (outs Wire:$out);
  let assemblyFormat = [{
    $label `,` $isPublic `->` type($out) attr-dict
  }];
}

def MulOp : R1CSOp<"mul", []> {
  let summary = "Multiply a wire by a constant";
  let arguments = (ins Wire:$wire, APIntAttr:$value, APIntAttr:$prime);
  let results = (outs Factor:$out);
  let assemblyFormat = [{
    $wire `*` $value `->` type($out) attr-dict
  }];
}

def SumOp : R1CSOp<"sum", []> {
  let summary = "Combine factors by summing";
  let arguments = (ins Factor:$lhs, Factor:$rhs);
  let results = (outs Factor:$out);
  let assemblyFormat = [{
    $lhs `+` $rhs `->` type($out) attr-dict
  }];
}

def ConstrainOp : R1CSOp<"constrain", []> {
  let summary = "Assert constraint: a*b-c=0";
  let arguments = (ins Factor:$a, Factor:$b, Optional<Factor>:$c);
  let assemblyFormat = [{
    $a `,` $b (`,` $c^)? attr-dict
  }];
}

