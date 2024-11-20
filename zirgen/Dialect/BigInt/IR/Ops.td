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

def LoadOp : BigIntOp<"load", [InferTypeOpAdaptor]> {
  let summary = "Load a value";
  let arguments = (ins UI32Attr:$bitWidth, UI32Attr:$arena, UI32Attr:$offset, DefaultValuedAttr<UI32Attr, "0">:$minBits);
  let results = (outs BigInt:$out);
  let assemblyFormat = [{ $bitWidth `,` $arena `,` $offset `->` type($out) attr-dict }];
}

def StoreOp : BigIntOp<"store", []> {
  let summary = "Load a value";
  let arguments = (ins BigInt:$in, UI32Attr:$arena, UI32Attr:$offset);
  let assemblyFormat = [{ $in `,` $arena `,` $offset `:` type($in) attr-dict }];
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

def AddOp : BinaryOp<"add", [Pure, Commutative, DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {}
def SubOp : BinaryOp<"sub", [Pure, DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {}
def MulOp : BinaryOp<"mul", [Pure, Commutative, DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {}
def NondetRemOp : BinaryOp<"nondet_rem", [DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {}
def NondetQuotOp : BinaryOp<"nondet_quot", [DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {}
def NondetInvOp : BinaryOp<"nondet_inv", [DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {}
def InvOp : BinaryOp<"inv", []> {}
def ReduceOp : BinaryOp<"reduce", []> {}

def EqualZeroOp : BigIntOp<"eqz", [DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Require a number to be equal to zero, or fail";
  let arguments = (ins BigInt:$in);
  let assemblyFormat = [{ $in `:` type($in) attr-dict }];
}
