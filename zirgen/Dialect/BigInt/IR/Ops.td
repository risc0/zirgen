// Copyright (c) 2024 RISC Zero, Inc.

include "mlir/IR/OpBase.td"
include "zirgen/Dialect/BigInt/IR/Dialect.td"
include "zirgen/Dialect/BigInt/IR/Types.td"
include "zirgen/Dialect/Zll/IR/Interfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class BigIntOp<string mnemonic, list<Trait> traits = []>
    : Op<BigIntDialect, mnemonic, traits> {
}

def DefOp : BigIntOp<"def", [InferTypeOpAdaptor, DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Define a input/witness value and its label ID";
  let arguments = (ins UI32Attr:$bitWidth, UI64Attr:$label, BoolAttr:$isPublic, DefaultValuedAttr<UI32Attr, "0">:$minBits);
  let results = (outs BigInt:$out);
  let assemblyFormat = [{ $bitWidth `,` $label `,` $isPublic `->` type($out) attr-dict }];
}

def ConstOp : BigIntOp<"const", [Pure, ConstantLike, InferTypeOpAdaptor, CodegenNeverInlineOp, DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Introduce a numeric constant";
  let arguments = (ins APIntAttr:$value);
  let results = (outs BigInt:$out);
  let assemblyFormat = [{ $value `->` type($out) attr-dict }];
  let hasFolder = 1;
}

class BinaryOp<string mnemonic, list<Trait> traits = []>
    : BigIntOp<mnemonic, !listconcat([InferTypeOpAdaptor], traits)> {
  let arguments = (ins BigInt:$lhs, BigInt:$rhs);
  let results = (outs BigInt:$out);
  let assemblyFormat = [{ $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `->` type($out) attr-dict }];
}

def AddOp : BinaryOp<"add", [Pure, Commutative]> {}
def SubOp : BinaryOp<"sub", [Pure, ]> {}
def MulOp : BinaryOp<"mul", [Pure, Commutative]> {}
def NondetRemOp : BinaryOp<"nondet_rem", [DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {}
def NondetQuotOp : BinaryOp<"nondet_quot", [DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {}
def ReduceOp : BinaryOp<"reduce", []> {}

def EqualZeroOp : BigIntOp<"eqz", [DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Require a number to be equal to zero, or fail";
  let arguments = (ins BigInt:$in);
  let assemblyFormat = [{ $in `:` type($in) attr-dict }];
}
