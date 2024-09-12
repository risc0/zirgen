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

#include <fstream>
#include <iostream>

#include "risc0/core/elf.h"
#include "risc0/core/util.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "zirgen/Conversions/Typing/BuiltinComponents.h"
#include "zirgen/Conversions/Typing/ComponentManager.h"
#include "zirgen/Dialect/ZHLT/IR/Codegen.h"
#include "zirgen/Dialect/ZHLT/Transforms/Passes.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/Passes.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "zirgen/Main/Main.h"
#include "zirgen/Main/RunTests.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/layout/viz.h"
#include "zirgen/dsl/lower.h"
#include "zirgen/dsl/parser.h"
#include "zirgen/dsl/passes/Passes.h"
#include "zirgen/dsl/stats.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

namespace cl = llvm::cl;
namespace codegen = zirgen::codegen;
namespace ZStruct = zirgen::ZStruct;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input zirgen file>"),
                                          cl::value_desc("filename"),
                                          cl::Required);
static cl::opt<std::string> circuitInfo(cl::desc("circuit identification string"),
                                        cl::value_desc("circuit_info"),
                                        cl::init("ZIRGEN:unknown"));
static cl::list<std::string> includeDirs("I", cl::desc("Add include path"), cl::value_desc("path"));
static cl::opt<size_t>
    maxDegree("max-degree", cl::desc("Maximum degree of validity polynomial"), cl::init(5));

void openMainFile(llvm::SourceMgr& sourceManager, std::string filename) {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code error = fileOrErr.getError())
    sourceManager.PrintMessage(llvm::SMLoc(),
                               llvm::SourceMgr::DiagKind::DK_Error,
                               "could not open input file " + filename);
  sourceManager.AddNewSourceBuffer(std::move(*fileOrErr), mlir::SMLoc());
}

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);

  zirgen::registerZirgenCommon();
  zirgen::registerRunTestsCLOptions();

  cl::ParseCommandLineOptions(argc, argv, "zirgen compiler\n");

  mlir::DialectRegistry registry;
  zirgen::registerZirgenDialects(registry);

  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::SourceMgr sourceManager;
  sourceManager.setIncludeDirs(includeDirs);
  mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceManager, &context);
  openMainFile(sourceManager, inputFilename);

  zirgen::dsl::Parser parser(sourceManager);
  parser.addPreamble(zirgen::Typing::getBuiltinPreamble());

  auto ast = parser.parseModule();
  if (!ast) {
    const auto& errors = parser.getErrors();
    for (const auto& error : errors) {
      sourceManager.PrintMessage(llvm::errs(), error);
    }
    llvm::errs() << "parsing failed with " << errors.size() << " errors\n";
    return 1;
  }

  std::optional<mlir::ModuleOp> zhlModule = zirgen::dsl::lower(context, sourceManager, ast.get());
  if (!zhlModule) {
    return 1;
  }

  std::optional<mlir::ModuleOp> typedModule = zirgen::Typing::typeCheck(context, zhlModule.value());
  if (!typedModule) {
    return 1;
  }

  mlir::PassManager pm(&context);
  applyDefaultTimingPassManagerCLOptions(pm);
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n";
    return 1;
  }
  pm.enableVerifier(true);
  zirgen::addAccumAndGlobalPasses(pm);
  pm.addPass(zirgen::ZStruct::createOptimizeLayoutPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(zirgen::Zhlt::createStripTestsPass());
  zirgen::addTypingPasses(pm);

  pm.addPass(zirgen::dsl::createElideTrivialStructsPass());
  pm.addPass(zirgen::Zhlt::createCircuitDefPass({.circuitInfo = circuitInfo}));
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());

  if (failed(pm.run(typedModule.value()))) {
    llvm::errs() << "an internal compiler error occurred while lowering this module:\n";
    typedModule->print(llvm::errs());
    return 1;
  }

  typedModule->print(llvm::outs());

  /*
    if (failed(zirgen::checkDegreeExceeded(*typedModule, maxDegree))) {
      llvm::errs() << "Degree exceeded; aborting\n";
      return 1;
    }

  */

  /*
    pm.addPass(zirgen::ZStruct::createExpandLayoutPass());

    if (inlineLayout) {
    }


    if (failed(pm.run(typedModule.value()))) {
      llvm::errs() << "an internal compiler error occurred while lowering this module:\n";
      typedModule->print(llvm::errs());
      return 1;
    }

    if (emitAction == Action::PrintLayoutType) {
      if (auto topFunc = typedModule->lookupSymbol<zirgen::Zhlt::ExecFuncOp>("exec$Top")) {
        std::stringstream ss;
        mlir::Type lt = topFunc.getLayoutType();
        zirgen::layout::viz::layoutSizes(lt, ss);
        llvm::outs() << ss.str();
        return 0;
      } else {
        llvm::errs() << "error: circuit contains no component named `Top`\n";
        return 1;
      }
    } else if (emitAction == Action::PrintLayoutAttr) {
      std::stringstream ss;
      zirgen::layout::viz::layoutAttrs(*typedModule, ss);
      llvm::outs() << ss.str();
      return 0;
    } else if (emitAction == Action::PrintZStruct) {
      typedModule->print(llvm::outs());
      return 0;
    }

    if (emitAction == Action::PrintStats) {
      zirgen::dsl::printStats(*typedModule);
      return 0;
    }

    if (doTest) {
      return zirgen::runTests(*typedModule);
    }

    if (emitAction == Action::PrintRust || emitAction == Action::PrintCpp) {
      codegen::CodegenOptions codegenOpts;
      static codegen::RustLanguageSyntax kRust;
      static codegen::CppLanguageSyntax kCpp;

      codegenOpts.lang = (emitAction == Action::PrintRust)
                             ? static_cast<codegen::LanguageSyntax*>(&kRust)
                             : static_cast<codegen::LanguageSyntax*>(&kCpp);

      zirgen::codegen::CodegenEmitter emitter(codegenOpts, &llvm::outs(), &context);
      if (zirgen::Zhlt::emitModule(*typedModule, emitter).failed()) {
        llvm::errs() << "Failed to emit circuit\n";
        return 1;
      }
    }

    return 0; */
}
