// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#ifndef ZSTRUCT_DIALECT
#define ZSTRUCT_DIALECT

include "mlir/IR/DialectBase.td"

def ZStructDialect : Dialect {
  let name = "zstruct";
  let cppNamespace = "::zirgen::ZStruct";
  let summary = "The ZStruct Dialect";
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::func::FuncDialect",
    "zirgen::Zll::ZllDialect"
  ];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // ZSTRUCT_DIALECT
