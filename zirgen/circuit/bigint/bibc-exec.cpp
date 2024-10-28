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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "zirgen/Dialect/BigInt/Bytecode/bibc.h"
#include "zirgen/Dialect/BigInt/Bytecode/decode.h"
#include "zirgen/Dialect/BigInt/Bytecode/file.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/IR/Eval.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include <iostream>

namespace bibc = zirgen::BigInt::Bytecode;
namespace cl = llvm::cl;
static cl::opt<std::string> inputBibcFile(cl::Positional,
                                          cl::desc("<input bibc file>"),
                                          cl::value_desc("filename"),
                                          cl::Required);

static cl::list<std::string>
    inputs(cl::Positional, cl::desc("values for inputs"), cl::value_desc("inputs"), cl::ZeroOrMore);

static cl::opt<bool> verbose("v", cl::desc("Verbose output"));

using BytePoly = zirgen::BigInt::BytePoly;

void printBytePoly(const BytePoly& bp) {
  bool first = true;
  for (auto coeff : bp) {
    if (first) {
      first = false;
    } else {
      std::cerr << " ";
    }
    std::cerr << coeff;
  }
}

void printWitness(std::string name, const std::vector<BytePoly>& witness) {
  std::cerr << name << " witness";
  if (witness.size()) {
    std::cerr << ":\n";
    for (auto& bp : witness) {
      std::cerr << "  ";
      printBytePoly(bp);
      std::cerr << "\n";
    }
  } else {
    std::cerr << " is empty\n";
  }
}

int main(int argc, char** argv) {
  llvm::InitLLVM y(argc, argv);
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "bibc evaluator\n");

  FILE* stream = fopen(inputBibcFile.c_str(), "rb");
  if (!stream) {
    std::cerr << "could not open bibc input file " + inputBibcFile << "\n";
    return 1;
  }

  bibc::Program prog;
  try {
    bibc::read(prog, stream);
  } catch (const bibc::IOException& e) {
    std::cerr << "check failure while reading; bibc file contents invalid\n";
    std::cerr << e.what() << "\n";
    return 1;
  }
  fclose(stream);

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<zirgen::BigInt::BigIntDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Unpack the bibc structure into MLIR ops
  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  auto func = bibc::decode(module, prog);

  // run the evaluator and generate digests
  std::vector<mlir::APInt> inputVals;
  for (auto& input : inputs) {
    inputVals.push_back(mlir::APInt(64, strtoull(input.c_str(), NULL, 10)));
  }
  if (inputVals.size() < prog.inputs.size()) {
    std::cerr << "not enough inputs: expected " << prog.inputs.size();
    std::cerr << " but only got " << inputVals.size() << "\n";
    return 1;
  }
  if (inputVals.size() > prog.inputs.size()) {
    std::cerr << "too many inputs: expected " << prog.inputs.size();
    std::cerr << " but got " << inputVals.size() << "\n";
    return 1;
  }

  auto output = zirgen::BigInt::eval(func, inputVals);

  if (verbose) {
    printWitness("constant", output.constantWitness);
    printWitness("public", output.publicWitness);
    printWitness("private", output.privateWitness);
  }
  // The evaluator returns an extension element which is not in Montgomery
  // form, which we will convert before display in order to match the result
  // expected from random_ext_elem() in the risc0_zkp crate.
  std::cout << zirgen::toMontgomery(output.z[0]) << " ";
  std::cout << zirgen::toMontgomery(output.z[1]) << " ";
  std::cout << zirgen::toMontgomery(output.z[2]) << " ";
  std::cout << zirgen::toMontgomery(output.z[3]) << "\n";
}
