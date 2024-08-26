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

#include <iostream>

#include "mlir/IR/AsmState.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/Transforms/Passes.h"
#include "zirgen/Dialect/IOP/IR/IR.h"
#include "zirgen/Dialect/R1CS/Conversion/R1CSToBigInt/Passes.h"
#include "zirgen/Dialect/R1CS/IR/R1CS.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/r1cs/lower.h"
#include "zirgen/compiler/r1cs/r1csfile.h"
#include "zirgen/compiler/r1cs/validate.h"
#include "zirgen/compiler/r1cs/wtnsfile.h"

// Define command line interface
namespace cl = llvm::cl;
static cl::opt<std::string> inputR1csFile(cl::Positional,
                                          cl::desc("<input r1cs file>"),
                                          cl::value_desc("filename"),
                                          cl::Required);

namespace {
enum Action {
  None,
  MLIR,
  BigInt,
  Zll,
  Rust,
};
} // namespace

namespace {

std::unique_ptr<llvm::raw_fd_ostream> openOutputFile(llvm::StringRef path, llvm::StringRef name) {
  std::string filename = (path + "/" + name).str();
  std::error_code ec;
  auto ofs = std::make_unique<llvm::raw_fd_ostream>(filename, ec);
  if (ec) {
    throw std::runtime_error("Unable to open file: " + filename);
  }
  return ofs;
}

void emitLang(llvm::StringRef langName,
              zirgen::codegen::LanguageSyntax* lang,
              llvm::StringRef path,
              mlir::ModuleOp module) {
  zirgen::codegen::CodegenOptions codegenOpts;
  codegenOpts.lang = lang;
  if (path.empty()) {
    llvm::raw_ostream &output = llvm::outs();
    zirgen::codegen::CodegenEmitter emitter(codegenOpts, &output, module.getContext());
    emitter.emitModule(module);
  } else {
    auto ofs = openOutputFile(path, ("bigint." + langName + ".inc").str());
    zirgen::codegen::CodegenEmitter emitter(codegenOpts, ofs.get(), module.getContext());
    emitter.emitModule(module);
  }
}

} // namespace

static cl::opt<enum Action>
    emitAction("emit",
               cl::desc("Desired output"),
               cl::values(clEnumValN(MLIR, "mlir", "Plain MLIR representation of R1CS"),
                          clEnumValN(BigInt, "bigint", "Compute using integers"),
                          clEnumValN(Zll, "zll", "Lower to ZLL dialect"),
                          clEnumValN(Rust, "rust", "Generate Rust validation function")));

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);

  mlir::registerAsmPrinterCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "zirgen R1CS importer\n");

  mlir::DialectRegistry registry;
  registry.insert<zirgen::R1CS::R1CSDialect>();
  registry.insert<zirgen::BigInt::BigIntDialect>();
  registry.insert<zirgen::Zll::ZllDialect>();
  registry.insert<zirgen::Iop::IopDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Open the input R1CS file
  FILE* stream = fopen(inputR1csFile.c_str(), "rb");
  if (!stream) {
    std::cerr << "could not open R1CS input file " + inputR1csFile << "\n";
    return 1;
  }

  // Read file contents
  std::unique_ptr<zirgen::r1csfile::System> sys;
  try {
    sys = zirgen::r1csfile::read(stream);
  } catch (const zirgen::r1csfile::IOException& e) {
    std::cerr << "check failure while reading; r1cs file contents invalid\n";
    return 1;
  }
  fclose(stream);

  if (emitAction == Action::None) {
    std::cerr << inputR1csFile.c_str() << "\n";
    return 0;
  }

  // Convert to MLIR representation
  auto op = zirgen::R1CS::lower(context, *sys.get());
  if (!op) {
    return 1;
  }

  if (emitAction == Action::MLIR) {
    op->dump();
    return 0;
  }

  // Lower initial R1CS representation to ZLL
  mlir::PassManager pm(&context);
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n";
    return 1;
  }
  pm.enableVerifier(true);
  pm.addPass(zirgen::R1CSToBigInt::createR1CSToBigIntPass());
  pm.addPass(zirgen::BigInt::createLowerReducePass());
  if (failed(pm.run(*op))) {
    llvm::errs() << "an internal validation error occurred:\n";
    op->print(llvm::errs());
    return 1;
  }

  if (emitAction == Action::BigInt) {
    op->dump();
    return 0;
  }

  if (emitAction == Action::Zll) {
    mlir::PassManager pm2(&context);
    pm2.addPass(zirgen::BigInt::createLowerZllPass());
    if (mlir::failed(pm2.run(*op))) {
      throw std::runtime_error("Failed to apply bigint lowering passes");
    }

    op->dump();
    return 0;
  } else if (emitAction == Action::Rust) {
    static zirgen::codegen::RustLanguageSyntax rustLang;
    rustLang.addContextArgument("ctx: &mut BigIntContext");
    std::string dir = "";
    emitLang("rs", &rustLang, dir, *op);
  }

  return 0;
}
