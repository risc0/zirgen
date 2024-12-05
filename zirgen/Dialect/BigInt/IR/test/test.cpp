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

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/InitLLVM.h"

#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/IR/Eval.h"
#include "zirgen/Dialect/BigInt/Transforms/Passes.h"

#include "zirgen/Dialect/IOP/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "zirgen/circuit/bigint/rsa.h"
#include "zirgen/circuit/recursion/encode.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/zkp/hash.h"
#include "zirgen/compiler/zkp/poseidon2.h"
#include "zirgen/compiler/zkp/sha256.h"

#include "llvm/Support/Format.h"

using namespace llvm;
using namespace mlir;
using namespace zirgen;
using namespace zirgen::BigInt;

// Very low quality random integers for testing
APInt randomTestInteger(size_t bits) {
  APInt ret(bits, 1, false);
  for (size_t i = 0; i < bits - 1; i++) {
    ret = ret * 2;
    ret = ret + (rand() % 27644437) % 2;
  }
  return ret;
}

std::string toStr(APInt val) {
  SmallVector<char, 128> chars;
  val.toStringUnsigned(chars, 16);
  return std::string(chars.data(), chars.size());
}

Digest hashPublic(llvm::ArrayRef<APInt> inputs) {
  size_t roundBits = BigInt::kBitsPerCoeff * BigInt::kCoeffsPerPoly;
  std::vector<uint32_t> words;
  for (size_t i = 0; i < inputs.size(); i++) {
    size_t roundedWidth = ceilDiv(inputs[i].getBitWidth(), roundBits) * roundBits;
    APInt rounded = inputs[i].zext(roundedWidth);
    for (size_t j = 0; j < roundedWidth; j += 32) {
      words.push_back(rounded.extractBitsAsZExtValue(32, j));
    }
  }
  return shaHash(words.data(), words.size());
}

struct CheckedBytesExternHandler : public Zll::ExternHandler {
  std::deque<uint8_t> coeffs;
  std::optional<std::vector<uint64_t>> doExtern(llvm::StringRef name,
                                                llvm::StringRef extra,
                                                llvm::ArrayRef<const Zll::InterpVal*> arg,
                                                size_t outCount) override {
    if (name == "readCoefficients") {
      assert(outCount == 16);
      if (coeffs.size() < 16) {
        llvm::errs() << "RAN OUT OF COEFFICIENTS\n";
        throw std::runtime_error("OUT_OF_COEFFICIENTS");
      }
      std::vector<uint64_t> ret;
      for (size_t i = 0; i < 16; i++) {
        ret.push_back(coeffs.front());
        coeffs.pop_front();
      }
      return ret;
    }
    return ExternHandler::doExtern(name, extra, arg, outCount);
  }
};

namespace {
enum class Action {
  None,
  PrintRust,
  PrintCpp,
  PrintBigInt,
  PrintZll,
  PrintZkr,
  DumpWom,
};
} // namespace

static cl::opt<enum Action> emitAction(
    "emit",
    cl::desc("The kind of output desired"),
    cl::init(Action::None),
    cl::values(
        clEnumValN(Action::None, "none", "Don't emit anything"),
        clEnumValN(Action::PrintBigInt, "bigint", "Output generated BigInt IR for execution"),
        clEnumValN(Action::PrintZll, "zll", "Output generated verifification ZLL IR"),
        clEnumValN(Action::PrintZkr, "zkr", "Output verification zkr"),
        clEnumValN(Action::PrintRust, "rust", "Output generated execution rust code"),
        clEnumValN(Action::PrintCpp, "cpp", "Output generated execution cpp code"),
        clEnumValN(Action::DumpWom,
                   "wom",
                   "Output WOM values generated when interpreting verification circuit ")));

static cl::opt<bool> doTest("test", cl::desc("Run test in interpreter"));

// TODO: Figure out what to do with this option that zirgen_genfiles gives us.
static cl::list<std::string> includeDirs("I", cl::desc("Add include path"), cl::value_desc("path"));

int main(int argc, const char** argv) {
  llvm::InitLLVM y(argc, argv);
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "bigint test");

  if (doTest && emitAction != Action::None) {
    llvm::errs() << "Cannot both emit and run tests\n";
    cl::PrintHelpMessage();
    exit(1);
  }

  if (!doTest && emitAction == Action::None) {
    llvm::errs() << "Nothing to do!\n";
    cl::PrintHelpMessage();
    exit(1);
  }

  MLIRContext context;
  context.getOrLoadDialect<BigInt::BigIntDialect>();
  context.getOrLoadDialect<Iop::IopDialect>();

  size_t numBits = 256;
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto inModule = ModuleOp::create(loc);
  builder.setInsertionPointToEnd(&inModule.getBodyRegion().front());
  auto inFunc = builder.create<func::FuncOp>(loc, "main", FunctionType::get(&context, {}, {}));
  builder.setInsertionPointToEnd(inFunc.addEntryBlock());
  makeRSAChecker(builder, loc, numBits);
  builder.create<func::ReturnOp>(loc);

  PassManager pm(&context);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(BigInt::createLowerReducePass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(inModule))) {
    throw std::runtime_error("Failed to apply basic optimization passes");
  }

  std::vector<APInt> values;
  values.push_back(randomTestInteger(numBits));
  values.push_back(randomTestInteger(numBits));
  values.push_back(RSA(values[0], values[1]));
  for (size_t i = 0; i < 3; i++) {
    errs() << "values[" << i << "] = " << toStr(values[i]) << "\n";
  }
  Digest expected = hashPublic(values);
  if (emitAction == Action::PrintBigInt) {
    llvm::outs() << inModule;
    exit(0);
  }

  if (emitAction == Action::PrintRust || emitAction == Action::PrintCpp) {
    codegen::CodegenOptions codegenOpts;
    static codegen::RustLanguageSyntax kRust;
    static codegen::CppLanguageSyntax kCpp;

    codegenOpts.lang = (emitAction == Action::PrintRust)
                           ? static_cast<codegen::LanguageSyntax*>(&kRust)
                           : static_cast<codegen::LanguageSyntax*>(&kCpp);

    zirgen::codegen::CodegenEmitter emitter(codegenOpts, &llvm::outs(), &context);
    emitter.emitModule(inModule);
    exit(0);
  }

  // Do the lowering
  auto outModule = inModule.clone();
  PassManager pm2(&context);
  pm2.addPass(createLowerZllPass());
  pm2.addPass(createCanonicalizerPass());
  pm2.addPass(createCSEPass());
  if (failed(pm2.run(outModule))) {
    throw std::runtime_error("Failed to apply basic optimization passes");
  }
  if (emitAction == Action::PrintZll) {
    llvm::outs() << outModule;
    exit(0);
  }

  auto outFunc = outModule.lookupSymbol<mlir::func::FuncOp>("main");
  if (emitAction == Action::PrintZkr) {
    std::vector<uint32_t> encoded =
        recursion::encode(recursion::HashType::POSEIDON2, &outFunc.front());
    llvm::outs().write(reinterpret_cast<const char*>(encoded.data()),
                       encoded.size() * sizeof(uint32_t));
    exit(0);
  }

  //  Do the evaluation that the lowering will verify
  EvalOutput retEval;
  size_t evalCount = 0;
  inModule.walk([&](func::FuncOp evalFunc) {
    retEval = BigInt::eval(evalFunc, values);
    retEval.print(llvm::errs());
    ++evalCount;
  });
  assert(evalCount == 1);

  // Set up the IOP for interpretation
  std::vector<uint32_t> iopVals(/*control root=*/8 + /*z=*/4);
  for (size_t i = 8; i < 8 + 4; i++) {
    iopVals[i] = toMontgomery(retEval.z[i]);
  }
  auto readIop = std::make_unique<zirgen::ReadIop>(
      std::make_unique<Poseidon2Rng>(), iopVals.data(), iopVals.size());

  // Add the checked bytes
  CheckedBytesExternHandler externHandler;
  auto addBytes = [&](const std::vector<BigInt::BytePoly>& in) {
    for (size_t i = 0; i < in.size(); i++) {
      for (size_t j = 0; j < in[i].size(); j++) {
        externHandler.coeffs.push_back(in[i][j]);
      }
      if (in[i].size() % BigInt::kCoeffsPerPoly != 0) {
        for (size_t j = in[i].size() % BigInt::kCoeffsPerPoly; j < BigInt::kCoeffsPerPoly; j++) {
          externHandler.coeffs.push_back(0);
        }
      }
    }
  };
  addBytes(retEval.constantWitness);
  addBytes(retEval.publicWitness);
  addBytes(retEval.privateWitness);

  assert(doTest && "Unhandled command line case");

  // Run the lowered stuff
  Zll::Interpreter interp(&context, poseidon2HashSuite());
  interp.setExternHandler(&externHandler);
  auto outBuf = interp.makeBuf(outFunc.getArgument(0), 32, Zll::BufferKind::Global);
  interp.setIop(outFunc.getArgument(1), readIop.get());
  if (failed(interp.runBlock(outFunc.front()))) {
    errs() << "Failed to interpret\n";
    throw std::runtime_error("FAIL");
  }

  if (emitAction == Action::DumpWom) {
    // Now encode it as microcode for the recursion circuit and get the WOM associations
    llvm::DenseMap<mlir::Value, uint64_t> toId;
    std::vector<uint32_t> code = encode(recursion::HashType::POSEIDON2, &outFunc.front(), &toId);
    // 'Reverse' toId so that it is in execution order
    std::map<uint64_t, mlir::Value> toValue;
    for (auto kvp : toId) {
      toValue[kvp.second] = kvp.first;
    }

    AsmState asmState(outModule);
    for (auto [id, val] : toValue) {
      llvm::outs() << "WOM[" << id << "]: ";
      if (interp.hasVal(val)) {
        auto ival = interp.getInterpVal(val);
        ival->print(llvm::outs(), asmState);
      } else {
        llvm::outs() << "(missing)";
      }
      llvm::outs() << " src=";
      if (val.getDefiningOp() && val.getDefiningOp()->getNumResults() > 1) {
        val.printAsOperand(llvm::outs(), asmState);
        llvm::outs() << " from ";
      }
      llvm::outs() << val << "\n";
    }
  }

  // TODO: Compute digest of public inputs to verify against values below

  Digest actual;
  for (size_t i = 0; i < 8; i++) {
    actual.words[i] = 0;
    for (size_t j = 0; j < 2; j++) {
      actual.words[i] |= outBuf[i * 2 + j][0] << (j * 16);
    }
  }
  if (actual != expected) {
    errs() << "Hash mismatch\n";
    errs() << hexDigest(actual) << "\n";
    errs() << hexDigest(expected) << "\n";
    throw std::runtime_error("Mismatch");
  }

  // errs() << outModule;
}
