// Copyright (c) 2024 RISC Zero, Inc.

#ifndef BIGINT_DIALECT
#define BIGINT_DIALECT

include "mlir/IR/DialectBase.td"

def BigIntDialect : Dialect {
  let name = "bigint";
  let cppNamespace = "::zirgen::BigInt";
  let summary = "BigInt dialect";
  let useDefaultTypePrinterParser = 1;
}

#endif // BIGINT_DIALECT
