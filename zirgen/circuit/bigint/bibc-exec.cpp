// read a bibc file
// accept inputs as arguments
// run the evaluator
// print the Z values

// then:
// convert the bibc structure to MLIR
// run the MLIR evaluator
// print the Z values

// make sure they match

#include <iostream>
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/IR/Eval.h"
#include "zirgen/Dialect/BigInt/Bytecode/bibc.h"
#include "zirgen/Dialect/BigInt/Bytecode/file.h"
#include "zirgen/Dialect/BigInt/Bytecode/decode.h"

namespace bibc = zirgen::BigInt::Bytecode;
namespace cl = llvm::cl;
static cl::opt<std::string> inputBibcFile(cl::Positional,
                                          cl::desc("<input bibc file>"),
                                          cl::value_desc("filename"),
                                          cl::Required);

static cl::list<std::string> inputs(cl::Positional,
  cl::desc("values for inputs"),
  cl::value_desc("inputs"),
  cl::ZeroOrMore);

static cl::opt<bool> verbose("v", cl::desc("Verbose output"));

using BytePoly = zirgen::BigInt::BytePoly;

void printBytePoly(const BytePoly &bp) {
  bool first = true;
  for (auto coeff: bp) {
    if (first) {
      first = false;
    } else {
      std::cerr << " ";
    }
    std::cerr << coeff;
  }
}

void printWitness(std::string name, const std::vector<BytePoly> &witness) {
  std::cerr << name << " witness";
  if (witness.size()) {
    std::cerr << ":\n";
     for (auto &bp: witness) {
        std::cerr << "  ";
        printBytePoly(bp);
        std::cerr << "\n";
      }
   } else {
    std::cerr << " is empty\n";
  }
}

int main(int argc, char **argv) {
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
  for (auto &input: inputs) {
    inputVals.push_back(mlir::APInt(64, strtoull(input.c_str(), NULL, 10)));
  }
  if (inputVals.size() < prog.inputs.size()) {
    std::cerr << "not enough inputs: expected " << prog.inputs.size();
    std::cerr << " but only got " << inputVals.size() << "\n";
    return 1;
  }
  if (inputVals.size() > prog.inputs.size()) {
    std::cerr << "too many inputs: expected " << prog.inputs.size();
    std::cerr << " but got " <<  inputVals.size() << "\n";
    return 1;
  }

  auto output = zirgen::BigInt::eval(func, inputVals);

  if (verbose) {
    printWitness("constant", output.constantWitness);
    printWitness("public", output.publicWitness);
    printWitness("private", output.privateWitness);
  }
  std::cout << output.z[0] << " " << output.z[1] << " ";
  std::cout << output.z[2] << " " << output.z[3] << "\n";
}
