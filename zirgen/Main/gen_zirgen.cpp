// Copyright 2025 RISC Zero, Inc.
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
#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/Main/Main.h"
#include "zirgen/Main/RunTests.h"
#include "zirgen/Main/Target.h"
#include "zirgen/Main/Utils.h"
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
using namespace zirgen;
using namespace zirgen::codegen;
using namespace mlir;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("input.zir"), cl::value_desc("filename"), cl::Required);
static cl::list<std::string> includeDirs("I", cl::desc("Add include path"), cl::value_desc("path"));
static cl::opt<size_t>
    maxDegree("max-degree", cl::desc("Maximum degree of validity polynomial"), cl::init(5));
static cl::opt<std::string> protocolInfo("protocol-info",
                                         cl::desc("Protocol information string"),
                                         cl::init("ZIRGEN_TEST_____"));
static cl::opt<bool> multiplyIf("multiply-if",
                                cl::desc("Mulitply out and refactor `if` statements when "
                                         "generating constraints, which can improve CSE."),
                                cl::init(false));
static cl::opt<bool>
    parallelWitgen("parallel-witgen",
                   cl::desc("Assume the witness can be generated in parallel, and that all externs "
                            "used in witness generation are idempotent."),
                   cl::init(false));
static cl::opt<std::string> circuitName("circuit-name", cl::desc("Name of circuit"));

llvm::cl::opt<size_t> stepSplitCount{
    "step-split-count",
    llvm::cl::desc(
        "Split up step functions into this many files to allow for parallel compilation"),
    llvm::cl::value_desc("numParts"),
    llvm::cl::init(1)};

namespace {

void openMainFile(llvm::SourceMgr& sourceManager, std::string filename) {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code error = fileOrErr.getError())
    sourceManager.PrintMessage(llvm::SMLoc(),
                               llvm::SourceMgr::DiagKind::DK_Error,
                               "could not open input file " + filename);
  sourceManager.AddNewSourceBuffer(std::move(*fileOrErr), mlir::SMLoc());
}

std::string getCircuitName(StringRef inputFilename) {
  if (!circuitName.empty())
    return circuitName;

  StringRef fn = StringRef(inputFilename).rsplit('/').second;
  if (fn.empty())
    fn = inputFilename;
  fn.consume_back(".zir");
  return fn.str();
}

ModuleOp makeStepFuncs(ModuleOp mod) {
  mlir::ModuleOp stepFuncs = mod.clone();
  // Privatize everything that we don't need, and generate step functions.
  mlir::PassManager pm(mod.getContext());
  applyDefaultTimingPassManagerCLOptions(pm);
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n";
    exit(1);
  }
  pm.enableVerifier(true);

  pm.addPass(zirgen::Zhlt::createLowerStepFuncsPass());
  pm.addPass(zirgen::ZStruct::createBuffersToArgsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolPrivatizePass(/*excludeSymbols=*/{"step$Top", "step$Top$accum"}));
  pm.addPass(mlir::createSymbolDCEPass());

  if (parallelWitgen) {
    pm.addPass(mlir::createInlinerPass());
    pm.addPass(zirgen::ZStruct::createInlineLayoutPass());
    pm.addPass(zirgen::ZStruct::createUnrollPass());
    pm.addPass(zirgen::Zhlt::createOptimizeParWitgenPass());
    pm.addPass(createCSEPass());
    pm.addPass(zirgen::Zhlt::createOutlineIfsPass());
    pm.addPass(zirgen::Zhlt::createOptimizeParWitgenPass());
  }

  if (failed(pm.run(stepFuncs))) {
    llvm::errs()
        << "an internal compiler error occurred while making step functions for this module:\n";
    stepFuncs.print(llvm::errs());
    exit(1);
  }

  return stepFuncs;
}

} // namespace

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);

  zirgen::registerCodegenCLOptions();
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
  //  pm.addPass(zirgen::ZStruct::createOptimizeLayoutPass());
  pm.addPass(zirgen::Zhlt::createElideRedundantMembersPass());
  pm.addPass(zirgen::dsl::createFieldDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  zirgen::addTypingPasses(pm);

  pm.addPass(zirgen::dsl::createGenerateCheckPass());
  pm.addPass(zirgen::dsl::createInlinePurePass());
  pm.addPass(zirgen::dsl::createHoistInvariantsPass());

  auto& checkPasses = pm.nest<zirgen::Zhlt::CheckFuncOp>();
  checkPasses.addPass(zirgen::ZStruct::createInlineLayoutPass());
  if (multiplyIf)
    checkPasses.addPass(zirgen::Zll::createIfToMultiplyPass());
  checkPasses.addPass(mlir::createCanonicalizerPass());
  checkPasses.addPass(mlir::createCSEPass());
  if (multiplyIf) {
    checkPasses.addPass(zirgen::Zll::createMultiplyToIfPass());
    checkPasses.addPass(mlir::createCanonicalizerPass());
    checkPasses.addPass(zirgen::dsl::createTopologicalShufflePass());
  }

  if (failed(pm.run(typedModule.value()))) {
    llvm::errs() << "an internal compiler error occurred while lowering this module:\n";
    typedModule->print(llvm::errs());
    return 1;
  }

  typedModule->walk([&](mlir::FunctionOpInterface op) {
    if (op.getName().contains("test$"))
      op.erase();
  });

  auto circuitName = getCircuitName(inputFilename);
  auto circuitNameAttr = zirgen::Zll::CircuitNameAttr::get(&context, circuitName);

  setModuleAttr(*typedModule, circuitNameAttr);

  emitPoly(*typedModule, circuitName, protocolInfo);

  pm.clear();
  pm.addPass(zirgen::dsl::createElideTrivialStructsPass());
  pm.addPass(zirgen::ZStruct::createExpandLayoutPass());
  pm.addPass(zirgen::dsl::createFieldDCEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(zirgen::dsl::createFieldDCEPass());
  pm.addPass(mlir::createCSEPass());

  if (failed(pm.run(typedModule.value()))) {
    llvm::errs() << "an internal compiler error occurred while optimizing this module:\n";
    typedModule->print(llvm::errs());
    return 1;
  }

  if (failed(zirgen::checkDegreeExceeded(*typedModule, maxDegree))) {
    llvm::errs() << "Degree exceeded; aborting\n";
    return 1;
  }

  // Create step functions
  mlir::ModuleOp stepFuncs = makeStepFuncs(*typedModule);

  emitTarget(RustCodegenTarget(circuitNameAttr),
             *typedModule,
             stepFuncs,
             codegen::getRustCodegenOpts(),
             stepSplitCount);
  emitTarget(CppCodegenTarget(circuitNameAttr),
             *typedModule,
             stepFuncs,
             codegen::getCppCodegenOpts(),
             stepSplitCount);
  emitTarget(CudaCodegenTarget(circuitNameAttr),
             *typedModule,
             stepFuncs,
             codegen::getCudaCodegenOpts(),
             stepSplitCount);

  return 0;
}
