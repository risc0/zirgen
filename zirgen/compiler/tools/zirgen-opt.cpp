// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/Transforms/Passes.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/Passes.h"
#include "zirgen/Dialect/Zll/Conversion/ZStructToZll/Passes.h"
#include "zirgen/Dialect/Zll/IR/Codegen.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/dsl/passes/Passes.h"

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

using namespace zirgen;

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSymbolPrivatizePass();
  mlir::registerInlinerPass();
  zirgen::BigInt::registerPasses();
  zirgen::Zll::registerPasses();
  zirgen::ZStructToZll::registerPasses();
  zirgen::ZStruct::registerPasses();
  zirgen::dsl::registerPasses();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<BigInt::BigIntDialect>();
  registry.insert<Zll::ZllDialect>();
  registry.insert<ZStruct::ZStructDialect>();
  registry.insert<Zhlt::ZhltDialect>();
  return failed(mlir::MlirOptMain(argc, argv, "Zirgen optimizer driver\n", registry));
}
