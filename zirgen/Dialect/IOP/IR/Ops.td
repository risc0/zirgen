// Copyright (c) 2023 RISC Zero, Inc.

include "mlir/IR/OpBase.td"
include "zirgen/Dialect/IOP/IR/Types.td"

class IopOp<string mnemonic, list<Trait> traits = []>
    : Op<IopDialect, mnemonic, traits> {
}

def ReadOp : IopOp<"read", [EvalOpAdaptor, IsReduce]> {
  let summary = "Read a value from the IOP";
  let assemblyFormat = [{ $iop `:` custom<ArrType>(type($outs)) attr-dict }];
  let arguments = (ins IOP:$iop, I1Attr:$flip);
  let results = (outs Variadic<AnyScalar>:$outs);
}

def CommitOp : IopOp<"commit", [EvalOpAdaptor, IsReduce]> {
  let summary = "Commit to a hash";
  let assemblyFormat = [{ $iop `,` $digest `:` type($digest) attr-dict }];
  let arguments = (ins IOP:$iop, Digest:$digest);
}

def RngBitsOp : IopOp<"rng_bits", [EvalOpAdaptor, IsReduce]> {
  let summary = "Get a random field element of some number of bits";
  let assemblyFormat = [{ $iop `,` $bits `:` type($out) attr-dict }];
  let arguments = (ins IOP:$iop, UI32Attr:$bits);
  let results = (outs Val:$out);
}

def RngValOp: IopOp<"rng_val", [EvalOpAdaptor, IsReduce]> {
  let summary = "Get a random field extention element";
  let assemblyFormat = [{ $iop `:` type($out) attr-dict }];
  let arguments = (ins IOP:$iop);
  let results = (outs Val:$out);
}
