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

#include "mlir/Transforms/Passes.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/Transforms/Passes.h"
#include "zirgen/Dialect/IOP/IR/IR.h"
#include "zirgen/Dialect/R1CS/Conversion/R1CSToBigInt/Passes.h"
#include "zirgen/Dialect/R1CS/IR/R1CS.h"
#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/recursion/encode.h"
#include "zirgen/circuit/verify/merkle.h"
#include "zirgen/circuit/verify/wrap_recursion.h"
#include "zirgen/circuit/verify/wrap_rv32im.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/r1cs/lower.h"
#include "zirgen/compiler/r1cs/r1csfile.h"
#include "zirgen/compiler/r1cs/validate.h"
#include "zirgen/compiler/r1cs/wtnsfile.h"
#include "llvm/Support/Format.h"

using namespace zirgen;
using namespace mlir;
namespace cl = llvm::cl;

namespace {

cl::list<std::string> outputFiles{
    cl::Positional, cl::OneOrMore, cl::desc("files in output directory")};

cl::opt<std::string> inputDir("input-dir",
                              cl::init("zirgen/circuit/circom"));

enum class Action {
  MLIR,
  BigInt,
  Zll,
  Zkr,
};

static cl::opt<Action>
    emitAction("emit",
               cl::desc("Desired output"),
               cl::values(clEnumValN(Action::MLIR, "mlir", "Plain MLIR representation of R1CS"),
                          clEnumValN(Action::BigInt, "bigint", "Compute using integers"),
                          clEnumValN(Action::Zll, "zll", "Lower to ZLL dialect")),
               cl::init(Action::Zkr));


std::unique_ptr<llvm::raw_fd_ostream> openOutputFile(StringRef path, StringRef name) {
  std::string filename = (path + "/" + name).str();
  std::error_code ec;
  auto ofs = std::make_unique<llvm::raw_fd_ostream>(filename, ec);
  if (ec) {
    throw std::runtime_error("Unable to open file: " + filename);
  }
  return ofs;
}

void emitLang(StringRef langName,
              zirgen::codegen::LanguageSyntax* lang,
              StringRef path,
              ModuleOp module) {
  auto ofs = openOutputFile(path, ("circom." + langName + ".inc").str());

  codegen::CodegenOptions codegenOpts;
  codegenOpts.lang = lang;
  zirgen::codegen::CodegenEmitter cg(codegenOpts, ofs.get(), module.getContext());
  cg.emitModule(module);

  // Emit witness info
  auto progMacroName = cg.getStringAttr("bigint_program_info");
  auto infoMacroName = cg.getStringAttr("bigint_witness_info");
  SmallVector<codegen::EmitPart> progNames;
  module.walk([&](mlir::func::FuncOp funcOp) {
    SmallVector<codegen::EmitPart> progMacroArgs;
    auto progName = codegen::CodegenIdent<codegen::IdentKind::Func>(funcOp.getNameAttr());
    progMacroArgs.emplace_back(progName);
    progNames.push_back(progName);

    progMacroArgs.emplace_back(
        [&](auto& cg) { cg << "/*iters=*/" << BigInt::getIterationCount(funcOp); });

    funcOp.walk([&](BigInt::DefOp defOp) {
      progMacroArgs.emplace_back([&, defOp](auto& cg) mutable {
        cg << "\n";
        cg.emitInvokeMacroV(
            infoMacroName,
            [&](auto& cg) { cg << "/*bits=*/" << defOp.getBitWidth(); },
            [&](auto& cg) { cg << "/*label=*/" << defOp.getLabel(); },
            [&](auto& cg) {
              if (defOp.getIsPublic())
                cg << "/*public=*/ true";
              else
                cg << "/*public=*/ false";
            },
            [&](auto& cg) { cg << "/*min_bits=*/" << defOp.getMinBits(); });
      });
    });
    cg.emitInvokeMacro(progMacroName, progMacroArgs);
    cg << "\n";
  });

  cg.emitInvokeMacro(cg.getStringAttr("bigint_program_list"), progNames);
}

mlir::func::FuncOp makeCircom(Module& module, llvm::StringRef name, std::string inputR1csFile) {
  // Open the input R1CS file
  FILE* stream = fopen(inputR1csFile.c_str(), "rb");
  if (!stream) {
    llvm::errs() << "could not open R1CS input file " << inputR1csFile << "\n";
    exit(1);
  }

  // Read file contents
  std::unique_ptr<zirgen::r1csfile::System> sys;
  try {
    sys = zirgen::r1csfile::read(stream);
  } catch (const zirgen::r1csfile::IOException& e) {
    llvm::errs() << "check failure while reading; r1cs file contents invalid\n";
    exit(1);
  }
  fclose(stream);

  // Convert to MLIR representation
  auto funcOp = module.addFunc<0>(name.str(), {}, [&]() {
    auto& builder = Module::getCurModule()->getBuilder();
    zirgen::R1CS::lower(builder, *sys.get());
  });
  return funcOp;
}

} // namespace

std::pair<llvm::StringLiteral, /*iters=*/size_t> kCircuits[] = {{"poseidon2", 1}};

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);
  registerEdslCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "gen_circom");

  std::string dir = llvm::StringRef(outputFiles[0]).rsplit('/').first.str();

  Module module;
  auto* ctx = module.getCtx();
  ctx->loadAllAvailableDialects();
  ctx->getOrLoadDialect<BigInt::BigIntDialect>();
  ctx->getOrLoadDialect<Iop::IopDialect>();
  ctx->getOrLoadDialect<R1CS::R1CSDialect>();
  ctx->getOrLoadDialect<Zll::ZllDialect>();
  ctx->getOrLoadDialect<mlir::func::FuncDialect>();


  for (auto [name, iters] : kCircuits) {
    auto funcOp = makeCircom(module, name, (inputDir + "/" + name + ".r1cs").str());
    BigInt::setIterationCount(funcOp, iters);
  }

  if (emitAction == Action::MLIR) {
    module.dump();
    return 0;
  }

  // Lower initial R1CS representation to ZLL
  mlir::PassManager pm(ctx);
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n";
    return 1;
  }
  pm.enableVerifier(true);
  pm.addPass(zirgen::R1CSToBigInt::createR1CSToBigIntPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(BigInt::createLowerReducePass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(module.getModule()))) {
    throw std::runtime_error("Failed to apply basic optimization passes");
  }

  if (emitAction == Action::BigInt) {
    module.dump();
    return 0;
  }

  static codegen::RustLanguageSyntax rustLang;
  rustLang.addContextArgument("ctx: &mut BigIntContext");
  rustLang.addItemsMacro("bigint_program_info");
  rustLang.addItemsMacro("bigint_program_list");
  emitLang("rs", &rustLang, dir, module.getModule());

  static codegen::CppLanguageSyntax cppLang;
  cppLang.addContextArgument("BigIntContext& ctx");
  emitLang("cpp", &cppLang, dir, module.getModule());

  PassManager pm2(module.getCtx());
  if (failed(applyPassManagerCLOptions(pm2))) {
    throw std::runtime_error("Failed to apply command line options");
  }
  pm2.addPass(createCanonicalizerPass());
  pm2.addPass(createCSEPass());
  pm2.addPass(BigInt::createLowerZllPass());
  pm2.addPass(createCanonicalizerPass());
  pm2.addPass(createCSEPass());
  if (failed(pm2.run(module.getModule()))) {
    throw std::runtime_error("Failed to apply basic optimization passes (2)");
  }

  if (emitAction == Action::Zll) {
    module.dump();
    return 0;
  }

  assert(emitAction == Action::Zkr);

  bool exceeded = false;
  module.getModule().walk([&](mlir::func::FuncOp func) {
    recursion::EncodeStats stats;
    zirgen::emitRecursion(dir, func, &stats);
    size_t iters = BigInt::getIterationCount(func);
    if (stats.totCycles > (1 << recursion::kRecursionPo2)) {
      exceeded = true;
    }
    llvm::errs() << "Encoded " << func.getName()
                 << llvm::format(
                        " with %d iterations using %d/%d (%.2f%%) cycles (%d cycles/iter)\n",
                        iters,
                        stats.totCycles,
                        1 << recursion::kRecursionPo2,
                        stats.totCycles * 100. / (1 << recursion::kRecursionPo2),
                        stats.totCycles / iters);
  });

  if (exceeded) {
    llvm::errs() << "One or more bigint probgrams exceeded the total number of allowed cycles.  "
                    "Perhaps decrease iterations?\n";
    return 1;
  }

  return 0;
}
