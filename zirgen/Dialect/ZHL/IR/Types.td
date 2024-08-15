// Copyright (c) 2023 RISC Zero, Inc.

include "mlir/IR/AttrTypeBase.td"
include "zirgen/Dialect/ZHL/IR/Dialect.td"

class ZhlType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ZhlDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Expr : ZhlType<"Expr", "expr"> {
  let summary = "Expression node";
}

