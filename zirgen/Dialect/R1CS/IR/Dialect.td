// Copyright (c) 2024 RISC Zero, Inc.

#ifndef R1CS_DIALECT
#define R1CS_DIALECT

include "mlir/IR/DialectBase.td"

def R1CSDialect : Dialect {
  let name = "r1cs";
  let cppNamespace = "::zirgen::R1CS";
  let summary = "R1CS dialect";
  let useDefaultTypePrinterParser = 1;
}

#endif // R1CS_DIALECT
