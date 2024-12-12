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

#include "zirgen/Main/Main.h"

#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Transforms/Passes.h"
#include "risc0/core/elf.h"
#include "risc0/core/util.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/Passes.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/Passes.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/dsl/passes/Passes.h"

namespace zirgen {

void registerZirgenCommon() {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::tracing::DebugConfig::registerCLOptions();

  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSymbolPrivatizePass();
  mlir::registerInlinerPass();
}

void registerZirgenDialects(mlir::DialectRegistry& registry) {
  // Dialects
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<zirgen::Zll::ZllDialect>();
  registry.insert<zirgen::ZStruct::ZStructDialect>();
  registry.insert<zirgen::Zhlt::ZhltDialect>();
  registry.insert<zirgen::Zhl::ZhlDialect>();

  mlir::func::registerInlinerExtension(registry);
}

void addAccumAndGlobalPasses(mlir::PassManager& pm) {
  pm.addPass(zirgen::dsl::createGenerateAccumPass());
  pm.addPass(zirgen::dsl::createGenerateGlobalsPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void addTypingPasses(mlir::PassManager& pm) {
  pm.addPass(zirgen::dsl::createGenerateBackPass());
  pm.addPass(zirgen::dsl::createGenerateCheckLayoutPass());
  pm.addPass(zirgen::dsl::createGenerateLayoutPass());
  pm.addPass(zirgen::Zhlt::createStripAliasLayoutOpsPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(zirgen::dsl::createGenerateExecPass());
  pm.addPass(mlir::createSymbolPrivatizePass({}));
  pm.addPass(zirgen::Zhlt::createAnalyzeBuffersPass());
  pm.addPass(zirgen::Zhlt::createGenerateStepsPass());
}

mlir::LogicalResult checkDegreeExceeded(mlir::ModuleOp module, size_t maxDegree) {
  bool degreeExceeded = false;
  module.walk([&](zirgen::Zhlt::CheckFuncOp op) {
    if (failed(op.verifyMaxDegree(maxDegree))) {
      degreeExceeded = true;
      // Ugh, apparently we don't get type aliases unless we print an
      // operation without a parent.  Copy this check function into a
      // blank module so we only get the problem check function without all the rest of the stuff.
      mlir::OpBuilder builder(module.getContext());
      auto tmpMod = mlir::ModuleOp::create(builder.getUnknownLoc());
      builder.setInsertionPointToStart(&tmpMod.getBodyRegion().front());
      builder.clone(*op.getOperation());

      llvm::errs() << "Check function:\n";
      tmpMod->print(llvm::outs());
    }
  });

  if (degreeExceeded)
    return mlir::failure();

  return mlir::success();
}

} // namespace zirgen
