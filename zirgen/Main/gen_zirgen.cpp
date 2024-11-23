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

#include "mlir/IR/IRMapping.h"
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

namespace {

void openMainFile(llvm::SourceMgr& sourceManager, std::string filename) {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code error = fileOrErr.getError())
    sourceManager.PrintMessage(llvm::SMLoc(),
                               llvm::SourceMgr::DiagKind::DK_Error,
                               "could not open input file " + filename);
  sourceManager.AddNewSourceBuffer(std::move(*fileOrErr), mlir::SMLoc());
}

std::unique_ptr<llvm::raw_ostream> openOutput(StringRef filename) {
  std::string path = (codegenCLOptions->outputDir + "/" + filename).str();
  std::error_code ec;
  auto ofs = std::make_unique<llvm::raw_fd_ostream>(path, ec);
  if (ec) {
    throw std::runtime_error("Unable to open file: " + path);
  }
  return ofs;
}

void emitDefs(CodegenEmitter& cg, ModuleOp mod, StringRef filename) {
  auto os = openOutput(filename);
  CodegenEmitter::StreamOutputGuard guard(cg, os.get());
  auto emitZhlt = Zhlt::getEmitter(mod, cg);
  if (emitZhlt->emitDefs().failed()) {
    llvm::errs() << "Failed to emit circuit definitions to " << filename << "\n";
    exit(1);
  }
}

void emitTypes(CodegenEmitter& cg, ModuleOp mod, StringRef filename) {
  auto os = openOutput(filename);
  CodegenEmitter::StreamOutputGuard guard(cg, os.get());
  cg.emitTypeDefs(mod);
}

template <typename... OpT> void emitOps(CodegenEmitter& cg, ModuleOp mod, StringRef filename) {
  auto os = openOutput(filename);
  CodegenEmitter::StreamOutputGuard guard(cg, os.get());
  for (auto& op : *mod.getBody()) {
    if (llvm::isa<OpT...>(&op)) {
      cg.emitTopLevel(&op);
    }
  }
}

void emitPoly(ModuleOp mod, StringRef circuitName) {
  ModuleOp funcMod = mod.cloneWithoutRegions();
  OpBuilder builder(funcMod.getContext());
  builder.createBlock(&funcMod->getRegion(0));

  // Convert functions to func::FuncOp, since that's what the edsl
  // codegen knows how to deal with
  mod.walk([&](zirgen::Zhlt::CheckFuncOp funcOp) {
    auto newFuncOp = builder.create<func::FuncOp>(funcOp.getLoc(),
                                                  builder.getStringAttr(circuitName),
                                                  TypeAttr::get(funcOp.getFunctionType()),
                                                  funcOp.getSymVisibilityAttr(),
                                                  funcOp.getArgAttrsAttr(),
                                                  funcOp.getResAttrsAttr());
    IRMapping mapping;
    newFuncOp.getBody().getBlocks().clear();
    funcOp.getBody().cloneInto(&newFuncOp.getBody(), mapping);
  });

  zirgen::Zll::setModuleAttr(funcMod, builder.getAttr<zirgen::Zll::ProtocolInfoAttr>(protocolInfo));

  mlir::PassManager pm(mod.getContext());
  applyDefaultTimingPassManagerCLOptions(pm);
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n";
    exit(1);
  }
  {
    auto& opm = pm.nest<mlir::func::FuncOp>();
    opm.addPass(zirgen::ZStruct::createInlineLayoutPass());
    opm.addPass(zirgen::ZStruct::createBuffersToArgsPass());
    opm.addPass(Zll::createMakePolynomialPass());
    opm.addPass(createCanonicalizerPass());
    opm.addPass(createCSEPass());
    opm.addPass(Zll::createComputeTapsPass());
  }

  //  pm.addPass(createPrintIRPass());

  if (failed(pm.run(funcMod))) {
    llvm::errs() << "an internal compiler error occurred while lowering this module:\n";
    funcMod.print(llvm::errs());
    exit(1);
  }

  emitCodeZirgenPoly(funcMod, codegenCLOptions->outputDir);

  // TODO: modularize generating the validity stuff
  auto rustOpts = codegen::getRustCodegenOpts();
  rustOpts.addFuncContextArgument<func::FuncOp>("ctx: &ValidityCtx");
  rustOpts.addCallContextArgument<Zll::GetOp, Zll::SetOp>("ctx");
  CodegenEmitter rustCg(rustOpts, mod.getContext());
  auto os = openOutput("validity.rs.inc");
  CodegenEmitter::StreamOutputGuard guard(rustCg, os.get());
  rustCg.emitModule(funcMod);
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

  StringRef circuitName = StringRef(inputFilename).rsplit('/').second;
  if (circuitName.empty())
    circuitName = inputFilename;
  circuitName.consume_back(".zir");

  setModuleAttr(*typedModule, zirgen::Zll::CircuitNameAttr::get(&context, circuitName));

  emitPoly(*typedModule, circuitName);

  pm.clear();
  pm.addPass(zirgen::dsl::createElideTrivialStructsPass());
  pm.addPass(zirgen::ZStruct::createExpandLayoutPass());
  pm.addPass(zirgen::dsl::createFieldDCEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(zirgen::dsl::createFieldDCEPass());
  pm.addPass(mlir::createCSEPass());

  if (failed(pm.run(typedModule.value()))) {
    llvm::errs() << "an internal compiler error occurred while lowering this module:\n";
    typedModule->print(llvm::errs());
    return 1;
  }

  if (failed(zirgen::checkDegreeExceeded(*typedModule, maxDegree))) {
    llvm::errs() << "Degree exceeded; aborting\n";
    return 1;
  }

  // Create step functions
  mlir::ModuleOp stepFuncs = typedModule->clone();
  // Privatize everything that we don't need, and generate step functions.
  pm.clear();
  pm.addPass(zirgen::Zhlt::createLowerStepFuncsPass());
  pm.addPass(zirgen::ZStruct::createBuffersToArgsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolPrivatizePass(/*excludeSymbols=*/{"step$Top", "step$Top$accum"}));
  pm.addPass(mlir::createSymbolDCEPass());

  if (failed(pm.run(stepFuncs))) {
    llvm::errs() << "an internal compiler error occurred while lowering this module:\n";
    stepFuncs.print(llvm::errs());
    return 1;
  }

  auto rustOpts = codegen::getRustCodegenOpts();
  CodegenEmitter rustCg(rustOpts, &context);
  emitDefs(rustCg, *typedModule, "defs.rs.inc");
  emitTypes(rustCg, *typedModule, "types.rs.inc");
  emitOps<ZStruct::GlobalConstOp>(rustCg, *typedModule, "layout.rs.inc");
  emitOps<Zhlt::StepFuncOp>(rustCg, stepFuncs, "steps.rs.inc");

  auto cppOpts = codegen::getCppCodegenOpts();
  CodegenEmitter cppCg(cppOpts, &context);
  emitDefs(cppCg, *typedModule, "defs.cpp.inc");
  emitTypes(cppCg, *typedModule, "types.h.inc");
  emitOps<ZStruct::GlobalConstOp>(cppCg, *typedModule, "layout.cpp.inc");
  emitOps<Zhlt::StepFuncOp, Zhlt::ExecFuncOp>(cppCg, stepFuncs, "steps.cpp.inc");

  return 0;
}
