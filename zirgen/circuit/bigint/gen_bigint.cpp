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
#include "zirgen/circuit/bigint/elliptic_curve.h"
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

void emit(StringRef langName,
          const zirgen::codegen::CodegenOptions& codegenOpts,
          StringRef path,
          ModuleOp module) {
  auto ofs = openOutputFile(path, ("bigint." + langName + ".inc").str());

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

// Specification of elliptic curve (EC) parameters used to generate EC ZKRs
struct ECSpec {
  llvm::StringLiteral name;
  size_t numBits;
  zirgen::BigInt::EC::WeierstrassCurve curve;
};

const RsaSpec kRsaSpecs[] = {
    // 256-bit RSA; primarily used for testing.
    {"rsa_256_x1", 256, 1},
    {"rsa_256_x2", 256, 2},

    // 3072-bit RSA.  As of this writing, verifying more than 15
    // claims makes the ZKR too big to run in BIGINT_PO2.
    {"rsa_3072_x15", 3072, 15},
};

// rz8test parameters
// rz8test is an 8-bit testing curve that is far too small to be secure but good for short tests
const APInt rz8test_prime(8, 179);
const APInt rz8test_a(8, 1);
const APInt rz8test_b(8, 12);
// Base point
const APInt rz8test_G_x(8, 157);
const APInt rz8test_G_y(8, 34);
const APInt rz8test_order(8, 199);

// secp256k1 parameters
const APInt secp256k1_prime = APInt::getAllOnes(256) - APInt::getOneBitSet(256, 32) -
                              APInt::getOneBitSet(256, 9) - APInt::getOneBitSet(256, 8) -
                              APInt::getOneBitSet(256, 7) - APInt::getOneBitSet(256, 6) -
                              APInt::getOneBitSet(256, 4);
const APInt secp256k1_a(8, 0);
const APInt secp256k1_b(8, 7);
// Base point
const APInt
    secp256k1_G_x(256, "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", 16);
const APInt
    secp256k1_G_y(256, "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", 16);
const APInt
    secp256k1_order(256, "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", 16);

const ECSpec kECSpecs[] = {
    // rz8test -- an in-house 8-bit testing curve; nowhere near big enough to be secure
    {"rz8test", 8, {rz8test_prime, rz8test_a, rz8test_b}},

    // secp256k1
    {"secp256k1", 256, {secp256k1_prime, secp256k1_a, secp256k1_b}},
};

} // namespace

cl::opt<std::string> outputDir{
    "output-dir", cl::desc("Output directory"), cl::value_desc("dir"), cl::Required};

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);
  registerEdslCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "gen_bigint");

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
  for (size_t numBits : {8}) {
    module.addFunc<0>("nondet_inv_test_" + std::to_string(numBits), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::makeNondetInvTest(builder, builder.getUnknownLoc(), numBits);
    });
  }
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
  // Elliptic Curve tests
  for (auto ec : kECSpecs) {
    module.addFunc<0>(std::string("ec_add_").append(ec.name), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::EC::makeECAddTest(builder,
                                        builder.getUnknownLoc(),
                                        ec.numBits,
                                        ec.curve.prime(),
                                        ec.curve.a(),
                                        ec.curve.b());
    });
  }
  for (auto ec : kECSpecs) {
    module.addFunc<0>(std::string("ec_doub_").append(ec.name), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::EC::makeECDoubleTest(builder,
                                           builder.getUnknownLoc(),
                                           ec.numBits,
                                           ec.curve.prime(),
                                           ec.curve.a(),
                                           ec.curve.b());
    });
  }
  for (auto ec : kECSpecs) {
    module.addFunc<0>(std::string("ec_mul_").append(ec.name), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::EC::makeECMultiplyTest(builder,
                                             builder.getUnknownLoc(),
                                             ec.numBits,
                                             ec.curve.prime(),
                                             ec.curve.a(),
                                             ec.curve.b());
    });
  }
  for (auto ec : kECSpecs) {
    module.addFunc<0>(std::string("ec_neg_").append(ec.name), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::EC::makeECNegateTest(builder,
                                           builder.getUnknownLoc(),
                                           ec.numBits,
                                           ec.curve.prime(),
                                           ec.curve.a(),
                                           ec.curve.b());
    });
  }
  for (auto ec : kECSpecs) {
    module.addFunc<0>(std::string("ec_sub_").append(ec.name), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::EC::makeECSubtractTest(builder,
                                             builder.getUnknownLoc(),
                                             ec.numBits,
                                             ec.curve.prime(),
                                             ec.curve.a(),
                                             ec.curve.b());
    });
  }
  for (auto ec : kECSpecs) {
    module.addFunc<0>(std::string("ec_pts_eq_").append(ec.name), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::EC::makeECValidatePointsEqualTest(builder,
                                                        builder.getUnknownLoc(),
                                                        ec.numBits,
                                                        ec.curve.prime(),
                                                        ec.curve.a(),
                                                        ec.curve.b());
    });
  }
  for (auto ec : kECSpecs) {
    module.addFunc<0>(std::string("ec_on_curve_").append(ec.name), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::EC::makeECValidatePointOnCurveTest(builder,
                                                         builder.getUnknownLoc(),
                                                         ec.numBits,
                                                         ec.curve.prime(),
                                                         ec.curve.a(),
                                                         ec.curve.b());
    });
  }
  for (auto ec : kECSpecs) {
    if (ec.name != "rz8test") {
      continue;
      // only need the test curve for the `freely` ZKRs
    }
    module.addFunc<0>(std::string("ec_add_freely_").append(ec.name), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::EC::makeECAddFreelyTest(builder,
                                              builder.getUnknownLoc(),
                                              ec.numBits,
                                              ec.curve.prime(),
                                              ec.curve.a(),
                                              ec.curve.b());
    });
  }
  for (auto ec : kECSpecs) {
    if (ec.name != "rz8test") {
      continue;
      // only need the test curve for the `freely` ZKRs
    }
    module.addFunc<0>(std::string("ec_doub_freely_").append(ec.name), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::EC::makeECDoubleFreelyTest(builder,
                                                 builder.getUnknownLoc(),
                                                 ec.numBits,
                                                 ec.curve.prime(),
                                                 ec.curve.a(),
                                                 ec.curve.b());
    });
  }
  for (auto ec : kECSpecs) {
    if (ec.name != "rz8test") {
      continue;
      // only need the test curve for the `freely` ZKRs
    }
    module.addFunc<0>(std::string("ec_mul_freely_").append(ec.name), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::EC::makeECMultiplyFreelyTest(builder,
                                                   builder.getUnknownLoc(),
                                                   ec.numBits,
                                                   ec.curve.prime(),
                                                   ec.curve.a(),
                                                   ec.curve.b());
    });
  }
  for (auto ec : kECSpecs) {
    if (ec.name != "rz8test") {
      continue;
      // only need the test curve for the `freely` ZKRs
    }
    module.addFunc<0>(std::string("ec_neg_freely_").append(ec.name), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::EC::makeECNegateFreelyTest(builder,
                                                 builder.getUnknownLoc(),
                                                 ec.numBits,
                                                 ec.curve.prime(),
                                                 ec.curve.a(),
                                                 ec.curve.b());
    });
  }
  for (auto ec : kECSpecs) {
    if (ec.name != "rz8test") {
      continue;
      // only need the test curve for the `freely` ZKRs
    }
    module.addFunc<0>(std::string("ec_sub_freely_").append(ec.name), {}, [&]() {
      auto& builder = Module::getCurModule()->getBuilder();
      zirgen::BigInt::EC::makeECSubtractFreelyTest(builder,
                                                   builder.getUnknownLoc(),
                                                   ec.numBits,
                                                   ec.curve.prime(),
                                                   ec.curve.a(),
                                                   ec.curve.b());
    });
  }
  // Perf tests
  // If enabled, these repeatedly perform the same operation, giving a better sense of the core
  // costs of the operation without setup/teardown overhead
  // for (size_t numReps : {5, 10, 256}) {
  //   const size_t numBits = 256;
  //   module.addFunc<0>("rep_ec_add_secp256k1_r" + std::to_string(numReps), {}, [&]() {
  //     auto& builder = Module::getCurModule()->getBuilder();
  //     zirgen::BigInt::EC::makeRepeatedECAddTest(builder, builder.getUnknownLoc(), numBits,
  //     numReps,
  //         secp256k1_prime, secp256k1_a, secp256k1_b);
  //   });
  // }
  // for (size_t numReps : {5, 10, 256}) {
  //   const size_t numBits = 256;
  //   module.addFunc<0>("rep_ec_doub_secp256k1_r" + std::to_string(numReps), {}, [&]() {
  //     auto& builder = Module::getCurModule()->getBuilder();
  //     zirgen::BigInt::EC::makeRepeatedECDoubleTest(builder, builder.getUnknownLoc(), numBits,
  //     numReps,
  //         secp256k1_prime, secp256k1_a, secp256k1_b);
  //   });
  // }

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

  auto rustOpts = codegen::getRustCodegenOpts();
  rustOpts.addFuncContextArgument<mlir::func::FuncOp>("ctx: &mut BigIntContext");
  rustOpts.addCallContextArgument<mlir::func::CallOp>("ctx");
  auto rustLang = dynamic_cast<codegen::RustLanguageSyntax*>(rustOpts.lang);
  assert(rustLang && "expecting getRustCodegenOpts to use RustLanguage");
  rustLang->addItemsMacro("bigint_program_info");
  rustLang->addItemsMacro("bigint_program_list");
  emit("rs", rustOpts, outputDir, module.getModule());

  auto cppOpts = codegen::getCppCodegenOpts();
  cppOpts.addFuncContextArgument<mlir::func::FuncOp>("BigIntContext& ctx");
  cppOpts.addCallContextArgument<mlir::func::CallOp>("ctx");
  emit("cpp", cppOpts, outputDir, module.getModule());

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
    zirgen::emitRecursion(outputDir, func, &stats);
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
    llvm::errs() << "One or more bigint programs exceeded the total number of allowed cycles.  "
                    "Perhaps decrease iterations?\n";
    return 1;
  }

  return 0;
}
