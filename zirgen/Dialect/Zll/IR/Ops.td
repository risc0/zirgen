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

include "zirgen/Dialect/Zll/IR/Attrs.td"
include "zirgen/Dialect/Zll/IR/Types.td"

def IsPoly : DeclareOpInterfaceMethods<PolyOp>;

def AnyScalar : AnyTypeOf<[Val, Digest]>;
def ExternParameter : AnyTypeOf<[AnyScalar, String, VariadicPack]>;
def Sha256Digest :
    Type<And<[
      CPred<"mlir::isa<DigestType>($_self)">,
      CPred<"mlir::cast<DigestType>($_self).getKind() == DigestKind::Sha256">]>>,
    BuildableType<"DigestType::get($_builder.getContext(), DigestKind::Sha256)">
{}

class ZllOp<string mnemonic, list<Trait> traits = []>
    : Op<ZllDialect, mnemonic, traits> {
}

def ConstOp : ZllOp<"const", [Pure, ConstantLike, EvalOpAdaptor, IsPoly, IsReduce, InferTypeOpAdaptor]> {
  let summary = "Introduce a numeric constant";
  let arguments = (ins PolynomialAttr:$coefficients);
  let results = (outs Val:$out);
  let assemblyFormat = [{ custom<PolynomialAttr>($coefficients) attr-dict }];
  let hasFolder = 1;
  let builders = [
    // add a speecialized builder for base field constants.
    OpBuilder<(ins "uint64_t":$elem), [{
      mlir::PolynomialAttr attr = mlir::PolynomialAttr::get($_builder.getContext(), { elem });
      build($_builder, $_state, attr);
    }]>
  ];
}

def StringOp : ZllOp<"string", [Pure, ConstantLike]> {
  let summary = "Introduce a string constant";
  let arguments = (ins StrAttr:$value);
  let results = (outs String:$out);
  let assemblyFormat = [{ $value attr-dict }];
  let hasFolder = 1;
}

def NondetOp : ZllOp<"nondet", [
    SingleBlockImplicitTerminator<"TerminateOp">, EvalOpAdaptor,
    DeclareOpInterfaceMethods<CodegenStatementOpInterface>]> {
  let regions = (region SizedRegion<1>:$inner);
  let assemblyFormat = [{ $inner attr-dict }];
  let hasCanonicalizer = 1;
}

def IfOp : ZllOp<"if", [
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"TerminateOp">, EvalOpAdaptor,
    DeclareOpInterfaceMethods<CodegenStatementOpInterface>]> {
  let summary = "Execute this region if input is nonzero";
  let arguments = (ins Val:$cond);
  let regions = (region SizedRegion<1>:$inner);
  let assemblyFormat = [{ $cond `:` type($cond) $inner attr-dict }];
  let hasCanonicalizer = 1;
}

def BackOp : ZllOp<"back", [Pure, InferTypeOpAdaptor]> {
  let arguments = (ins Buffer:$in, I32Attr:$back);
  let results = (outs Buffer:$out);
  let assemblyFormat = [{ $in `,`  $back `:` type($in) attr-dict }];
}

def SliceOp : ZllOp<"slice", [Pure, InferTypeOpAdaptor]> {
  let arguments = (ins Buffer:$in, UI32Attr:$offset, UI32Attr:$size);
  let results = (outs Buffer:$out);
  let assemblyFormat = [{ $in `,` $offset `,` $size `:` type($in) attr-dict }];
}

def GetOp : ZllOp<"get", [Pure, EvalOpAdaptor, IsPoly, InferTypeOpAdaptor,
                             DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let arguments = (ins Buffer:$buf, UI32Attr:$offset, UI32Attr:$back, OptionalAttr<UI32Attr>:$tap);
  let results = (outs Val:$out);
  let assemblyFormat = [{ $buf `[` $offset `]` `back` $back (`tap` $tap^)? `:` type($buf) attr-dict }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    BufferKind getBufferKind() { return mlir::cast<BufferType>(getBuf().getType()).getKind(); }
  }];
}

def SetOp : ZllOp<"set", [EvalOpAdaptor, DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let arguments = (ins Buffer:$buf, UI32Attr:$offset, Val:$in);
  let assemblyFormat = [{ $buf `:` type($buf) `[` $offset `]` `=` $in `:` type($in) attr-dict }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def GetGlobalOp : ZllOp<"get_global", [Pure, EvalOpAdaptor, IsPoly, InferTypeOpAdaptor,
                                          CodegenOpWithProperties<["Offset"]>]> {
  let summary = "Read from a buffer that doesn't support cycle offsets (e.g. a Global or Temporary buffer)";
  let arguments = (ins Buffer:$buf, UI32Attr:$offset);
  let results = (outs Val:$out);
  let assemblyFormat = [{ $buf `[` $offset `]` `:` type($buf) attr-dict }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def SetGlobalOp : ZllOp<"set_global", [EvalOpAdaptor, IsReduce,
                                       DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Store to a buffer that doesn't support cycle offsets (e.g. a Global or Temporary buffer)";
  let arguments = (ins Buffer:$buf, UI32Attr:$offset, Val:$in);
  let assemblyFormat = [{ $buf `:` type($buf) `[` $offset `]` `=` $in `:` type($in) attr-dict }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def EqualZeroOp : ZllOp<"eqz", [EvalOpAdaptor, IsReduce, DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Require a number to be equal to zero, or fail";
  let arguments = (ins Val:$in);
  let assemblyFormat = [{ $in `:` type($in) attr-dict }];
  let hasCanonicalizer = 1;
}

def BarrierOp : ZllOp<"barrier", [EvalOpAdaptor]> {
  let summary = "Barrier between stages, return a value";
  let arguments = (ins Val:$in);
  let assemblyFormat = [{ $in `:` type($in) attr-dict }];
}

def PowOp: ZllOp<"pow", [Pure, EvalOpFieldAdaptor, InferTypeOpAdaptor, IsPoly,
                            CodegenOpWithProperties<["Exponent"]>]> {
  let summary = "Raise a value to a constant power";
  let arguments = (ins
      Val:$in,
      UI32Attr:$exponent);
  let results = (outs Val:$out);
  let assemblyFormat = [{ $in `:` type($in) `,` $exponent attr-dict }];
  let hasCanonicalizer = 1;
}

class UnaryOp<string mnemonic, list<Trait> traits = []>
    : ZllOp<mnemonic, !listconcat([Pure, EvalOpFieldAdaptor, IsReduce, InferTypeOpAdaptor], traits)> {
  let arguments = (ins Val:$in);
  let results = (outs Val:$out);
  let assemblyFormat = [{ $in `:` type($in) attr-dict }];
  let hasFolder = 1;
}

def IsZeroOp : UnaryOp<"isz", [Pure]> {}
def NegOp : UnaryOp<"neg", [Pure, IsPoly]> {}
def InvOp : UnaryOp<"inv", [Pure]> {}

class BinaryOp<string mnemonic, list<Trait> traits = []>
    : ZllOp<mnemonic, !listconcat([Pure, EvalOpFieldAdaptor, IsReduce, InferTypeOpAdaptor], traits)> {
  let arguments = (ins Val:$lhs, Val:$rhs);
  let results = (outs Val:$out);
  let assemblyFormat = [{ $lhs `:` type($lhs) `,` $rhs `:` type($rhs) attr-dict }];
  let hasFolder = 1;
}

def AddOp : BinaryOp<"add", [Pure, Commutative, IsPoly, CodegenInfixOp<"+">]> {}
def SubOp : BinaryOp<"sub", [Pure, IsPoly, CodegenInfixOp<"-">]> {}
def MulOp : BinaryOp<"mul", [Pure, Commutative, IsPoly, CodegenInfixOp<"*">]> {}
def BitAndOp : BinaryOp<"bit_and", [Pure]> {}
def ModOp : BinaryOp<"mod", [Pure]> {}

def InRangeOp : ZllOp<"in_range", [Pure, SameOperandsAndResultType]> {
  let summary = [{
    Perform a range check, treating field elements as integers. Returns 1 if
    $low <= $mid < $high and 0 otherwise. It is a runtime error if $low > $high.
  }];
  let arguments = (ins Val:$low, Val:$mid, Val:$high);
  let results = (outs Val:$out);
  let assemblyFormat = [{ $low ` ` `<` `=` $mid ` ` `<` ` ` $high `:` type($out) attr-dict }];
  let hasFolder = 1;
}

def TerminateOp : ZllOp<"terminate", [Pure, Terminator,
                                         ParentOneOf<["NondetOp", "IfOp"]>,
                                         CodegenSkip]> {
  let summary = "Terminate without a value";
  let assemblyFormat = [{ attr-dict }];
}

def TrueOp : ZllOp<"true", [Pure, IsPoly, DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Constraint representing truth";
  let assemblyFormat = [{ attr-dict }];
  let results = (outs Constraint:$out);
}

def AndEqzOp : ZllOp<"and_eqz", [Pure, IsPoly, DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "And a constraint with an eqz constraint";
  let assemblyFormat = [{ $in `,` $val `:` type($val) attr-dict }];
  let arguments = (ins Constraint:$in, Val:$val);
  let results = (outs Constraint:$out);
}

def AndCondOp : ZllOp<"and_cond", [Pure, IsPoly, DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "And a constraint that conditionalizes another constraints";
  let assemblyFormat = [{ $in `,` $cond `:` type($cond) `,` $inner attr-dict }];
  let arguments= (ins Constraint:$in, Val:$cond, Constraint:$inner);
  let results = (outs Constraint:$out);
}

def VariadicPackOp : ZllOp<"variadic_pack", [Pure,
                                             EvalOpAdaptor,
                                             SameTypeOperands,
                                             DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Combines the operands into a variadic parameter pack";
  let arguments = (ins Variadic<Val>:$in);
  let results = (outs VariadicPack:$out);
  let assemblyFormat = [{ $in `:` functional-type($in, $out) attr-dict }];
}

def ExternOp : ZllOp<"extern", [EvalOpAdaptor,
                                DeclareOpInterfaceMethods<CodegenExprOpInterface>]> {
  let summary = "Escape hatch for calling random code in nondet sections";
  let assemblyFormat = [{ $name $extra `,` `(` $in `)` `:` functional-type($in, $out) attr-dict }];
  let arguments = (ins Variadic<ExternParameter>:$in, StrAttr:$name, StrAttr:$extra);
  let results = (outs Variadic<AnyScalar>:$out);
}

def SetGlobalDigestOp : ZllOp<"set_global_digest", [EvalOpAdaptor, IsReduce]> {
  let arguments = (ins Buffer:$buf, UI32Attr:$offset, Digest:$in);
  let assemblyFormat = [{ $buf `:` type($buf) `[` $offset `]` `=` $in `:` type($in) attr-dict }];
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def HashOp : ZllOp<"hash", [Pure, EvalOpAdaptor, IsReduce]> {
  let summary = "Hash a list of field elements";
  let assemblyFormat = [{ $flip `(` $in `)` `:` custom<ArrType>(type($in)) `->` type($out) attr-dict }];
  // TODO: Add verifier that variadic vals are all the same type
  let arguments = (ins I1Attr:$flip, Variadic<Val>:$in);
  let results = (outs Digest:$out);
}

def IntoDigestOp : ZllOp<"into_digest", [Pure, EvalOpAdaptor]> {
  let summary = "Reinterpret field elements of bytes into a Digest";
  let assemblyFormat = [{ `(` $in `)` `:` custom<ArrType>(type($in)) `->` type($out) attr-dict }];
  let arguments = (ins Variadic<Val>:$in);
  let results = (outs Digest:$out);
  let hasVerifier = 1;
}

def FromDigestOp : ZllOp<"from_digest", [Pure, EvalOpAdaptor]> {
  let summary = "Reinterpret a digest into field elements";
  let assemblyFormat = [{ $in `:` type($in) `->` custom<ArrType>(type($out)) attr-dict }];
  let arguments = (ins Digest:$in);
  let results = (outs Variadic<Val>:$out);
  let hasVerifier = 1;
}

def TaggedStructOp : ZllOp<"tagged_struct", [Pure, EvalOpAdaptor, AttrSizedOperandSegments]> {
  let summary = "Hash a tag, some digests and some field elements, using proper padding";
  let assemblyFormat = [{ `(` $digests `)` `,` `(` $vals `)` `:` custom<ArrType>(type($digests)) `,` custom<ArrType>(type($vals)) attr-dict }];
  let arguments = (ins StrAttr:$tag, Variadic<Digest>:$digests, Variadic<Val>:$vals);
  let results = (outs Sha256Digest:$out);
  let hasVerifier = 1;
}

def HashFoldOp : ZllOp<"hash_fold", [Pure, EvalOpAdaptor, IsReduce, SameOperandsAndResultType]> {
  let summary = "Hash two digests into one";
  let assemblyFormat = [{ $lhs `,` $rhs `:` type($lhs) attr-dict }];
  let arguments = (ins Digest:$lhs, Digest:$rhs);
  let results = (outs Digest:$out);
}

def HashAssertEqOp : ZllOp<"hash_assert_eq", [EvalOpAdaptor, IsReduce, SameTypeOperands]> {
  let summary = "Verify two hashes match";
  let assemblyFormat = [{ $lhs `,` $rhs `:` type($lhs) attr-dict }];
  let arguments = (ins Digest:$lhs, Digest:$rhs);
}

def SelectOp : ZllOp<"select", [Pure, EvalOpAdaptor, IsReduce]> {
  let summary = "Select a field element based on a field element";
  let assemblyFormat = [{ $idx `,` `(` $elems `)` `:` custom<SelectType>(type($out), type($idx), type($elems)) attr-dict }];
  // TODO: Add verifier that variadic vals are all the same type
  let arguments = (ins Val:$idx, Variadic<AnyScalar>:$elems);
  let results = (outs AnyScalar:$out);
}

def HashCheckedBytesOp : ZllOp<"hash_checked_bytes", [EvalOpAdaptor, InferTypeOpAdaptor]> {
  let summary = "Load, range check, and hash bytes and evaluate the corresponding polynomials";
  let assemblyFormat = [{ $eval_pt `,` $evals_count `:` type($eval_pt) `->` type($digest) `,` custom<ArrType>(type($evaluations)) attr-dict}];
  let arguments = (ins Val:$eval_pt, I64Attr:$evals_count);
  let results = (outs Digest:$digest, Variadic<Val>:$evaluations);
}

def HashCheckedBytesPublicOp : ZllOp<"hash_checked_bytes_public", [EvalOpAdaptor, InferTypeOpAdaptor]> {
  let summary = "Same as above but also output a SHA hash";
  let assemblyFormat = [{ $eval_pt `,` $evals_count `:` type($eval_pt) `->` type($digest) `,` type($pubDigest) `,` custom<ArrType>(type($evaluations)) attr-dict}];
  let arguments = (ins Val:$eval_pt, I64Attr:$evals_count);
  let results = (outs Digest:$digest, Digest:$pubDigest, Variadic<Val>:$evaluations);
}

def NormalizeOp : ZllOp<"normalize", [Pure, EvalOpAdaptor, IsReduce, InferTypeOpAdaptor]> {
  let summary = "Represents a reduction to normal form when using a higher field";
  let assemblyFormat = [{ $in `,` $bits `,` $low `:` type($in) attr-dict }];
  let arguments = (ins
      Val:$in,
      DefaultValuedAttr<UI32Attr, "0">:$bits,
      DefaultValuedAttr<StrAttr, "\"\"">:$low);
  let results = (outs Val:$out);
}

def MakeTemporaryBufferOp : ZllOp<"temp_buffer"> {
  let summary = "Create a temporary buffer that can be used by `load` and `store`.";
  let builders = [
    OpBuilder<(ins "Zll::ValType":$element, "unsigned":$size), [{
      build($_builder, $_state, $_builder.getType<zirgen::Zll::BufferType>(element, size, Zll::BufferKind::Temporary));
    }]>
  ];
  let results = (outs Buffer:$out);
}
