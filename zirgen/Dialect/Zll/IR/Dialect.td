// Copyright (c) 2023 RISC Zero, Inc.

#ifndef ZLL_DIALECT
#define ZLL_DIALECT

include "mlir/IR/DialectBase.td"

def ZllDialect : Dialect {
  let name = "zll";
  let cppNamespace = "::zirgen::Zll";
  let summary = "Zirgen Low-Level Dialect";
  let dependentDialects = ["mlir::func::FuncDialect"];
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    FieldAttr getDefaultField() { return getField("BabyBear"); }

    FieldAttr getField(llvm::StringRef fieldName);

private:
    llvm::StringMap<FieldAttr> fields;
  }];
}

#endif // ZLL_DIALECT
