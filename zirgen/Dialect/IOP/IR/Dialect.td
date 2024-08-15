// Copyright (c) 2023 RISC Zero, Inc.

#ifndef IOP_DIALECT
#define IOP_DIALECT

include "mlir/IR/DialectBase.td"

def IopDialect : Dialect {
  let name = "iop";
  let cppNamespace = "::zirgen::Iop";
  let summary = "Zirgen Interactive Oracle Proof Dialect";
  let dependentDialects = ["zirgen::Zll::ZllDialect"];
  let useDefaultTypePrinterParser = 1;
}

#endif // IOP_DIALECT
