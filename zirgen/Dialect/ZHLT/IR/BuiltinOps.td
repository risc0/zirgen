// Copyright (c) 2023 RISC Zero, Inc.

include "mlir/IR/OpBase.td"
include "zirgen/Dialect/ZHLT/IR/Dialect.td"
include "zirgen/Dialect/ZHLT/IR/Interfaces.td"

class BuiltinComponentOp<string mnemonic, list<string> methods = []>
  : Op<ZhltDialect, "builtin_" # mnemonic, [Symbol,    DeclareOpInterfaceMethods<ComponentOpInterface, methods>]> {
  let summary = "A builtin component with some custom behavior";
  let arguments = (ins
      SymbolNameAttr:$sym_name
  );  
}

def BuiltinLogOp : BuiltinComponentOp<"log", ["buildConstruct"]>;

def BuiltinArrayOp : BuiltinComponentOp<"array",
    ["requireComponent", "buildConstruct", "getValueType", "getLayoutType", "inferType"]>;
