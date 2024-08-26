// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
