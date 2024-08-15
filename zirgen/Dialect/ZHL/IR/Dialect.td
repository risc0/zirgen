// Copyright (c) 2023 RISC Zero, Inc.

#ifndef ZHL_DIALECT
#define ZHL_DIALECT

include "mlir/IR/DialectBase.td"

def ZhlDialect : Dialect {
  let name = "zhl";
  let cppNamespace = "::zirgen::Zhl";
  let summary = "Zirgen high-level dialect";
  let useDefaultTypePrinterParser = 1;
}

#endif // ZHL_DIALECT
