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
using namespace zirgen;
using namespace mlir;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("input.zir"), cl::value_desc("filename"), cl::Required);
static cl::opt<std::string>
    outputDir("output-dir", cl::desc("Output directory"), cl::value_desc("dir"), cl::Required);
static cl::list<std::string> includeDirs("I", cl::desc("Add include path"), cl::value_desc("path"));
static cl::opt<size_t>
    maxDegree("max-degree", cl::desc("Maximum degree of validity polynomial"), cl::init(5));

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
  std::string path = (outputDir + "/" + filename).str();
  std::error_code ec;
  auto ofs = std::make_unique<llvm::raw_fd_ostream>(path, ec);
  if (ec) {
    throw std::runtime_error("Unable to open file: " + path);
  }
  return ofs;
}

void emitDefs(ModuleOp mod, codegen::LanguageSyntax* lang, StringRef filename) {
  codegen::CodegenOptions opts;
  opts.lang = lang;

  auto os = openOutput(filename);
  zirgen::codegen::CodegenEmitter emitter(opts, os.get(), mod.getContext());
  auto emitZhlt = Zhlt::getEmitter(mod, emitter);
  if (emitZhlt->emitDefs().failed()) {
    llvm::errs() << "Failed to emit circuit definitions to " << filename << "\n";
    exit(1);
  }
}

void emitTypes(ModuleOp mod, codegen::LanguageSyntax* lang, StringRef filename) {
  codegen::CodegenOptions opts;
  opts.lang = lang;

  auto os = openOutput(filename);
  zirgen::codegen::CodegenEmitter emitter(opts, os.get(), mod.getContext());
  emitter.emitTypeDefs(mod);
}

template <typename... OpT>
void emitOps(ModuleOp mod, codegen::LanguageSyntax* lang, StringRef filename) {
  codegen::CodegenOptions opts;
  opts.lang = lang;

  auto os = openOutput(filename);
  zirgen::codegen::CodegenEmitter emitter(opts, os.get(), mod.getContext());

  OpBuilder builder(mod.getContext());
  OwningOpRef<ModuleOp> modCopy = builder.create<ModuleOp>(builder.getUnknownLoc());
  for (auto& op : *mod.getBody()) {
    if (llvm::isa<OpT...>(&op)) {
      emitter.emitTopLevel(&op);
    }
  }
}

} // namespace

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
  zirgen::addTypingPasses(pm);

  pm.addPass(zirgen::dsl::createGenerateCheckPass());
  pm.addPass(zirgen::dsl::createGenerateTapsPass());
  pm.addPass(zirgen::dsl::createGenerateValidityRegsPass());
  pm.addPass(zirgen::dsl::createGenerateValidityTapsPass());

  pm.addPass(zirgen::dsl::createElideTrivialStructsPass());
  pm.addPass(zirgen::ZStruct::createExpandLayoutPass());

  //  pm.addPass(mlir::createSymbolPrivatizePass(/*excludeSymbols=*/{"exec", "compute_accum"}));
  //  pm.addPass(mlir::createSymbolDCEPass());

  if (failed(pm.run(typedModule.value()))) {
    llvm::errs() << "an internal compiler error occurred while lowering this module:\n";
    typedModule->print(llvm::errs());
    return 1;
  }

  if (failed(zirgen::checkDegreeExceeded(*typedModule, maxDegree))) {
    llvm::errs() << "Degree exceeded; aborting\n";
    return 1;
  }

  static codegen::RustLanguageSyntax kRust;
  static codegen::CppLanguageSyntax kCpp;

  emitDefs(*typedModule, &kRust, "defs.rs.inc");
  emitTypes(*typedModule, &kRust, "types.rs.inc");
  emitOps<Zhlt::ValidityRegsFuncOp>(*typedModule, &kRust, "validity_regs.rs.inc");
  emitOps<Zhlt::ValidityTapsFuncOp>(*typedModule, &kRust, "validity_taps.rs.inc");
  emitOps<ZStruct::GlobalConstOp>(*typedModule, &kRust, "layout.rs.inc");
  emitOps<Zhlt::StepFuncOp, Zhlt::ExecFuncOp>(*typedModule, &kRust, "steps.rs.inc");

  emitDefs(*typedModule, &kCpp, "defs.cpp.inc");
  emitTypes(*typedModule, &kCpp, "types.h.inc");
  emitOps<Zhlt::ValidityTapsFuncOp>(*typedModule, &kCpp, "validity_regs.cpp.inc");
  emitOps<Zhlt::ValidityRegsFuncOp>(*typedModule, &kCpp, "validity_taps.cpp.inc");
  emitOps<ZStruct::GlobalConstOp>(*typedModule, &kCpp, "layout.cpp.inc");
  emitOps<Zhlt::StepFuncOp, Zhlt::ExecFuncOp>(*typedModule, &kCpp, "steps.cpp.inc");

  typedModule->print(*openOutput("circuit.ir"));

  return 0;
}
