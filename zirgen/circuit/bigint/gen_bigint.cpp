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
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/Passes.h"
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

  module.addFunc<0>("sum_test", {}, [&]() {
    auto& builder = Module::getCurModule()->getBuilder();
    auto loc = builder.getUnknownLoc();

    mlir::Type inType = builder.getIntegerType(8);

    mlir::Type reduceType = builder.getType<BigInt::BigIntType>(/*coeffs=*/1,
                                                                /*maxPos=*/255,
                                                                /*maxNeg=*/0,
                                                                /*minBits=*/0);

    auto three =
        builder.create<BigInt::ConstOp>(loc, reduceType, builder.getIntegerAttr(inType, 3));

    auto reduceTo =
        builder.create<BigInt::ConstOp>(loc, reduceType, builder.getIntegerAttr(inType, 255));

    SmallVector<Value> inputVals(5, three);
    auto arrayOp = builder.create<ZStruct::ArrayOp>(
        loc, builder.getType<ZStruct::ArrayType>(reduceType, 5), inputVals);
    auto reduceOp =
        builder.create<ZStruct::ReduceOp>(loc, reduceType, arrayOp, three, /*layout=*/Value{});
    {
      OpBuilder::InsertionGuard guard(builder);
      Block* bl = builder.createBlock(&reduceOp.getBody());
      Value lhs = bl->addArgument(reduceType, loc);
      Value rhs = bl->addArgument(reduceType, loc);
      auto mulOp = builder.create<BigInt::MulOp>(loc, lhs, rhs);
      auto reduceBigIntOp = builder.create<BigInt::ReduceOp>(loc, mulOp, reduceTo);
      builder.create<ZStruct::YieldOp>(loc, reduceBigIntOp);
    }

    auto expected = builder.create<BigInt::ConstOp>(
        loc, builder.getIntegerAttr(inType, (3 * 3 * 3 * 3 * 3) % 255));
    auto diff = builder.create<BigInt::SubOp>(loc, expected, reduceOp);
    builder.create<BigInt::EqualZeroOp>(loc, diff);
  });

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
  module.dump();

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
  pm2.addPass(ZStruct::createUnrollPass());
  pm2.addPass(BigInt::createLowerZllPass());
  pm2.addPass(createCanonicalizerPass());
  pm2.addPass(createCSEPass());
  if (failed(pm2.run(module.getModule()))) {
    throw std::runtime_error("Failed to apply basic optimization passes (2)");
  }

  module.dump();

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
