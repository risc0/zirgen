// Copyright (c) 2023 RISC Zero, Inc.

#ifndef ZHLT_DIALECT
#define ZHLT_DIALECT

include "mlir/IR/DialectBase.td"

def ZhltDialect : Dialect {
  let name = "zhlt";
  let cppNamespace = "::zirgen::Zhlt";
  let summary = "Zirgen high-level dialect (typed)";
  let dependentDialects = ["mlir::func::FuncDialect", "zirgen::Zll::ZllDialect"];
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
}

#endif // ZHLT_DIALECT
