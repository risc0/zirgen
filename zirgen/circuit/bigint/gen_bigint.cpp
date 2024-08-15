// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/Transforms/Passes.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/Transforms/Passes.h"
#include "zirgen/circuit/bigint/op_tests.h"
#include "zirgen/circuit/bigint/rsa.h"
#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/recursion/encode.h"
#include "zirgen/circuit/verify/merkle.h"
#include "zirgen/circuit/verify/wrap_recursion.h"
#include "zirgen/circuit/verify/wrap_rv32im.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "llvm/Support/Format.h"

using namespace zirgen;
using namespace mlir;
namespace cl = llvm::cl;

namespace {

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
  auto ofs = openOutputFile(path, ("bigint." + langName + ".inc").str());

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

struct RsaSpec {
  llvm::StringLiteral name;
  size_t numBits;
  size_t iters;
};

const RsaSpec kRsaSpecs[] = {
    // 256-bit RSA; primarily used for testing.
    {"rsa_256_x1", 256, 1},
    {"rsa_256_x2", 256, 2},

    // 3072-bit RSA.  As of this writing, verifying more than 15
    // claims makes the ZKR too big to run in BIGINT_PO2.
    {"rsa_3072_x15", 3072, 15},
};

} // namespace

cl::list<std::string> outputFiles{
    cl::Positional, cl::OneOrMore, cl::desc("files in output directory")};

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);
  registerEdslCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "gen_bigint");

  std::string dir = llvm::StringRef(outputFiles[0]).rsplit('/').first.str();

  Module module;
  auto* ctx = module.getCtx();
  ctx->getOrLoadDialect<BigInt::BigIntDialect>();
  ctx->getOrLoadDialect<Iop::IopDialect>();

  for (auto rsa : kRsaSpecs) {
    auto funcOp = module.addFunc<0>(std::string(rsa.name), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::makeRSA(builder, builder.getUnknownLoc(), rsa.numBits);
    });
    BigInt::setIterationCount(funcOp, rsa.iters);
  }
  // TODO: More bitwidth coverage?
  for (size_t numBits : {8}) {
    module.addFunc<0>("const_add_test_" + std::to_string(numBits), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::makeConstAddTest(builder, builder.getUnknownLoc(), numBits);
    });
  }
  for (size_t numBits : {16}) {
    module.addFunc<0>("const_add_alt_test_" + std::to_string(numBits), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::makeConstAddAltTest(builder, builder.getUnknownLoc(), numBits);
    });
  }
  for (size_t numBits : {8}) {
    module.addFunc<0>("const_mul_test_" + std::to_string(numBits), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::makeConstMulTest(builder, builder.getUnknownLoc(), numBits);
    });
  }
  for (size_t numBits : {8, 16, 128}) {
    module.addFunc<0>("add_test_" + std::to_string(numBits), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::makeAddTest(builder, builder.getUnknownLoc(), numBits);
    });
  }
  for (size_t numBits : {8}) {
    module.addFunc<0>("const_one_test_" + std::to_string(numBits), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::makeConstOneTest(builder, builder.getUnknownLoc(), numBits);
    });
  }
  for (size_t numBits : {16}) {
    module.addFunc<0>("const_twobyte_test_" + std::to_string(numBits), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::makeConstTwoByteTest(builder, builder.getUnknownLoc(), numBits);
    });
  }
  for (size_t numBits : {8, 128}) {
    module.addFunc<0>("sub_test_" + std::to_string(numBits), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::makeSubTest(builder, builder.getUnknownLoc(), numBits);
    });
  }
  for (size_t numBits : {8, 128}) {
    module.addFunc<0>("mul_test_" + std::to_string(numBits), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::makeMulTest(builder, builder.getUnknownLoc(), numBits);
    });
  }
  for (size_t numBits : {8, 128}) {
    module.addFunc<0>("reduce_test_" + std::to_string(numBits), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::makeReduceTest(builder, builder.getUnknownLoc(), numBits);
    });
  }

  PassManager pm(ctx);
  if (failed(applyPassManagerCLOptions(pm))) {
    throw std::runtime_error("Failed to apply command line options");
  }

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(BigInt::createLowerReducePass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(module.getModule()))) {
    throw std::runtime_error("Failed to apply basic optimization passes");
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
