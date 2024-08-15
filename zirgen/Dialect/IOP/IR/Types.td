// Copyright (c) 2023 RISC Zero, Inc.

include "mlir/IR/AttrTypeBase.td"
include "zirgen/Dialect/IOP/IR/Dialect.td"
include "zirgen/Dialect/Zll/IR/Types.td"

class IopType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<IopDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def IOP : IopType<"IOP", "iop", [CodegenTypeInterface]> {
  let summary = "An IOP read stream for the verifier";
}

def AnyScalar : AnyTypeOf<[Val, Digest]>;
