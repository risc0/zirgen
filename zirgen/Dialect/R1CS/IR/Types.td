// Copyright (c) 2024 RISC Zero, Inc.

include "mlir/IR/AttrTypeBase.td"
include "zirgen/Dialect/R1CS/IR/Dialect.td"

class R1CSType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<R1CSDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Wire : R1CSType<"Wire", "wire"> {
  let summary = "Wire identifier";
}

def Factor : R1CSType<"Factor", "factor"> {
  let summary = "Multiplication of a wire by a constant";
}
