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

#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

#include "risc0/core/elf.h"
#include "risc0/core/util.h"

#include "zirgen/Conversions/Typing/BuiltinComponents.h"
#include "zirgen/Conversions/Typing/ComponentManager.h"
#include "zirgen/Dialect/ZHL/IR/ZHL.h"
#include "zirgen/Dialect/ZHLT/IR/Codegen.h"
#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/Dialect/ZHLT/Transforms/Passes.h"
#include "zirgen/Dialect/ZStruct/Analysis/BufferAnalysis.h"
#include "zirgen/Dialect/ZStruct/IR/ZStruct.h"
#include "zirgen/Dialect/ZStruct/Transforms/Passes.h"
#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/IR/Interpreter.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/layout/viz.h"
#include "zirgen/dsl/lower.h"
#include "zirgen/dsl/parser.h"
#include "zirgen/dsl/passes/Passes.h"
#include "zirgen/dsl/stats.h"

namespace cl = llvm::cl;
namespace codegen = zirgen::codegen;
namespace ZStruct = zirgen::ZStruct;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input zirgen file>"),
                                          cl::value_desc("filename"),
                                          cl::Required);

static cl::opt<std::string> inputDataFilename("input-data-file",
                                              cl::desc("file containing input data for readInput"),
                                              cl::value_desc("filename"));
static cl::opt<std::string> inputDataHex("input-data-hex",
                                         cl::desc("inline hexadecimal input for readInput"),
                                         cl::value_desc("hex data"));

namespace {
enum Action {
  None,
  PrintAST,
  PrintZHL,
  PrintZHLT,
  OptimizeZHLT,
  PrintZStruct,
  PrintLayoutType,
  PrintLayoutAttr,
  PrintRust,
  PrintCpp,
  PrintStats,
};
} // namespace

static cl::opt<enum Action> emitAction(
    "emit",
    cl::desc("The kind of output desired"),
    cl::values(
        clEnumValN(PrintAST, "ast", "output the AST"),
        clEnumValN(PrintZHL, "zhl", "output untyped high level Zirgen IR"),
        clEnumValN(PrintZHLT, "zhlt", "output typed high level Zirgen IR"),
        clEnumValN(OptimizeZHLT, "zhltopt", "output optimized high level IR"),
        clEnumValN(PrintZStruct, "zstruct", "output Zirgen IR after semantic lowering"),
        clEnumValN(PrintLayoutType, "layouttype", "structure of circuit layout types as HTML"),
        clEnumValN(PrintLayoutAttr, "layoutattr", "content of generated layout attributes as text"),
        clEnumValN(PrintRust, "rust", "Output generated rust code"),
        clEnumValN(PrintCpp, "cpp", "Output generated cpp code"),
        clEnumValN(PrintStats, "stats", "Display statistics on generated circuit")));

static cl::list<std::string> includeDirs("I", cl::desc("Add include path"), cl::value_desc("path"));
static cl::opt<bool> doTest("test", cl::desc("Run tests for the main module"));
static cl::opt<bool> genValidity("validity",
                                 cl::desc("Generate validity polynomial evaluation functions"),
                                 cl::init(true));
static cl::opt<bool> inlineLayout("inline-layout",
                                  cl::desc("Inline layout into validity and check functions"),
                                  cl::init(false));
static cl::opt<size_t>
    maxDegree("max-degree", cl::desc("Maximum degree of validity polynomial"), cl::init(5));
static cl::opt<size_t>
    testCycles("test-cycles", cl::init(1), cl::desc("When running tests, run this many cycles"));
static cl::opt<std::string>
    testElf("test-elf", cl::desc("Elf to load to test ram"), cl::value_desc("path"));
static cl::opt<std::string>
    testGlobals("test-globals", cl::desc("Global values"), cl::value_desc("values"));

void openMainFile(llvm::SourceMgr& sourceManager, std::string filename) {
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code error = fileOrErr.getError())
    sourceManager.PrintMessage(llvm::SMLoc(),
                               llvm::SourceMgr::DiagKind::DK_Error,
                               "could not open input file " + filename);
  sourceManager.AddNewSourceBuffer(std::move(*fileOrErr), mlir::SMLoc());
}

struct TestExternHandler : public zirgen::Zll::ExternHandler {
  std::map<uint32_t, uint32_t> memory;
  std::map<uint64_t, std::map<uint64_t, uint64_t>> lookups;
  TestExternHandler() {
    if (testElf != "") {
      llvm::outs() << "LOADING FILE: '" << testElf << "'\n";
      auto file = risc0::loadFile(testElf);
      /* uint32_t entryPoint = */
      risc0::loadElf(file, memory);
    }
  }
  void check(bool condition, llvm::StringRef error) {
    if (!condition) {
      throw std::runtime_error(error.str());
    }
  }

  void divide(std::vector<uint64_t>& results, llvm::ArrayRef<uint64_t> args) {
    uint32_t numer = args[0] | (args[1] << 16);
    uint32_t denom = args[2] | (args[3] << 16);
    uint32_t signType = args[4];
    auto [quot, rem] = risc0::divide_rv32im(numer, denom, signType);
    results.push_back(quot & 0xffff);
    results.push_back(quot >> 16);
    results.push_back(rem & 0xffff);
    results.push_back(rem >> 16);
  }

  std::vector<uint64_t> doExtern(llvm::StringRef name,
                                 llvm::StringRef extra,
                                 llvm::ArrayRef<const zirgen::Zll::InterpVal*> args,
                                 size_t outCount) override {
    auto& os = llvm::outs();
    os << "[" << cycle << "] ";
    llvm::printEscapedString(name, os);

    // Including arguments for Log duplicates information in the output
    if (name != "Log") {
      os << "(";
      if (!extra.empty()) {
        printEscapedString(extra, os);
        os << ", ";
      }
      interleaveComma(args, os, [&](auto arg) { arg->print(os); });
      os << ") -> (";
    }

    // If a result is requested, return some information based on what's being requested.
    std::vector<uint64_t> results;
    if (name == "IsFirstCycle") {
      check(args.size() == 0, "IsFirstCycle expects no arguments");
      check(outCount == 1, "IsFirstCycle returns one result");
      results.push_back((cycle == 0) ? 1 : 0);
    } else if (name == "GetCycle") {
      check(args.size() == 0, "GetCycle expects no arguments");
      check(outCount == 1, "GetCycle returns one result");
      results.push_back(cycle);
    } else if (name == "SimpleMemoryPoke") {
      check(args.size() == 2, "SimpleMemoryPoke expects 2 arguments");
      auto fpArgs = asFpArray(args);
      memory[fpArgs[0]] = fpArgs[1];
    } else if (name == "SimpleMemoryPeek") {
      check(args.size() == 1, "SimpleMemoryPeek expects 1 arguments");
      auto fpArgs = asFpArray(args);
      results.push_back(memory[fpArgs[0]]);
    } else if (name == "MemoryPoke") {
      auto fpArgs = asFpArray(args);
      check(args.size() == 3, "MemoryPoke expects 3 arguments");
      check(outCount == 0, "MemoryPoke returns no results");
      check(fpArgs[0] < 0x40000000, "MemoryPoke address out of range");
      check(fpArgs[1] < 0x10000, "MemoryPoke low short out of range");
      check(fpArgs[2] < 0x10000, "MemoryPoke high short out of range");
      memory[fpArgs[0]] = fpArgs[1] | (fpArgs[2] << 16);
    } else if (name == "MemoryPeek") {
      auto fpArgs = asFpArray(args);
      check(args.size() == 1, "MemoryPoke expects 1 arguments");
      check(outCount == 2, "MemoryPoke returns two results");
      check(fpArgs[0] < 0x40000000, "MemoryPoke address out of range");
      // TODO: Maybe enforce this?  Right now, we don't 'set' registers
      // check(memory.count(fpArgs[0]), "MemoryPeek before MemoryPoke");
      results.push_back(memory[fpArgs[0]] & 0xffff);
      results.push_back(memory[fpArgs[0]] >> 16);
    } else if (name == "LookupDelta") {
      auto fpArgs = asFpArray(args);
      check(args.size() == 3, "LookupDelta expects 3 arguments");
      check(outCount == 0, "LookupDelta returns no results");
      lookups[fpArgs[0]][fpArgs[1]] += fpArgs[2];
      lookups[fpArgs[0]][fpArgs[1]] %= 15 * (1 << 27) + 1;
      if (lookups[fpArgs[0]][fpArgs[1]] == 0) {
        lookups[fpArgs[0]].erase(fpArgs[1]);
        if (lookups[fpArgs[0]].size() == 0) {
          lookups.erase(fpArgs[0]);
        }
      }
    } else if (name == "LookupPeek") {
      auto fpArgs = asFpArray(args);
      check(args.size() == 2, "LookupDelta expects 2 arguments");
      check(outCount == 1, "LookupDelta returns one element");
      uint64_t ret = 0;
      auto it1 = lookups.find(fpArgs[0]);
      if (it1 != lookups.end()) {
        auto it2 = it1->second.find(fpArgs[1]);
        if (it2 != it1->second.end()) {
          ret = it2->second;
        }
      }
      results.push_back(ret);
    } else if (name == "Divide") {
      check(args.size() == 5, "Divide expects 5 arguments");
      check(outCount == 4, "Divide returns 5 results");
      auto fpArgs = asFpArray(args);
      divide(results, fpArgs);
    } else if (name == "Log") {
      os << ": ";
      // Propagate the extern to the legacy "log" handler, repackaging the
      // variadic parameters as "regular" parameters
      using zirgen::Zll::InterpVal;
      llvm::StringRef message = args[0]->getAttr<mlir::StringAttr>().getValue();
      auto varArgs = args[1]->getAttr<mlir::ArrayAttr>().getValue();
      std::vector<InterpVal> vals(varArgs.size());
      std::vector<InterpVal*> valPtrs(varArgs.size());
      for (size_t i = 0; i < varArgs.size(); i++) {
        vals[i].setVal(varArgs[i].cast<mlir::PolynomialAttr>().asArrayRef());
        valPtrs[i] = &vals[i];
      }
      results = zirgen::Zll::ExternHandler::doExtern("log", message, valPtrs, outCount);
    } else if (name == "configureInput" || name == "readInput") {
      // Pass through to common implementation
      results = zirgen::Zll::ExternHandler::doExtern(name, extra, args, outCount);
    } else {
      // By default, let random externs pass
      // Fill with 0, 1, 2, ...
      for (uint64_t i = 0; i != outCount; ++i) {
        results.push_back(i);
      }
    }
    if (name != "Log") {
      interleaveComma(results, os);
      os << ")\n";
    }
    os.flush();
    return results;
  }

  size_t cycle;
};

std::vector<uint64_t> parseIntList(const std::string& str) {
  std::vector<uint64_t> ret;
  size_t pos = 0;
  while (pos != str.size() && pos != std::string::npos) {
    size_t next = str.find(',', pos);
    std::string numStr;
    if (next == std::string::npos) {
      numStr = str.substr(pos, next);
      pos = next;
    } else {
      numStr = str.substr(pos, next - pos);
      pos = next + 1;
    }
    ret.push_back(atoi(numStr.c_str()));
  }
  return ret;
}

int runTests(mlir::ModuleOp& module) {
  mlir::MLIRContext& context = *module.getContext();
  // Set all the symbols to private
  mlir::PassManager pm(&context);
  applyDefaultTimingPassManagerCLOptions(pm);
  if (failed(applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Pass manager does not agree with command line options.\n";
    return 1;
  }
  pm.enableVerifier(true);
  pm.addPass(zirgen::dsl::createEraseUnusedAspectsPass(/*forTests=*/true));
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createInlinerPass());
  mlir::OpPassManager& opm = pm.nest<zirgen::Zhlt::StepFuncOp>();
  opm.addPass(zirgen::ZStruct::createUnrollPass());
  // Canonicalization at this point seems to be unprofitable when running in the
  // interpreter
  // opm.addPass(mlir::createCanonicalizerPass());
  opm.addPass(mlir::createCSEPass());
  if (failed(pm.run(module))) {
    llvm::errs() << "an internal compiler error occurred while inlining the tests:\n";
    module.print(llvm::errs());
    return 1;
  }

  zirgen::ZStruct::BufferAnalysis bufferAnalysis(module);
  // Finally, run the tests
  for (zirgen::Zhlt::StepFuncOp stepFuncOp : module.getBody()->getOps<zirgen::Zhlt::StepFuncOp>()) {
    llvm::StringRef baseName = stepFuncOp.getName();
    baseName.consume_front("step$");
    if (!baseName.starts_with("test$"))
      continue;
    if (baseName.ends_with("$accum"))
      continue;
    bool expectFailure = baseName.starts_with("test$fail$");
    if (!expectFailure)
      assert(baseName.starts_with("test$succ$"));
    llvm::StringRef testName =
        baseName.drop_front(strlen("test$fail$") /* == strlen("test$succ$") */);
    llvm::errs() << "Running " << testName << "\n";

    zirgen::Zll::Interpreter interp(&context);

    // Allocate buffers
    using Polynomial = llvm::SmallVector<uint64_t, 4>;
    llvm::SmallVector<std::unique_ptr<std::vector<Polynomial>>> bufs;
    auto allBufs = bufferAnalysis.getAllBuffers();
    bufs.reserve(allBufs.size());
    for (auto bufDesc : allBufs) {
      auto& newBuf = bufs.emplace_back(std::make_unique<std::vector<Polynomial>>());
      if (bufDesc.kind == zirgen::Zll::BufferKind::Global) {
        newBuf->resize(bufDesc.regCount, Polynomial(1, zirgen::Zll::kFieldInvalid));
        interp.setNamedBuf(bufDesc.name, *newBuf, 0 /* no per-cycle offset */);
        if (bufDesc.name.str() == "global") {
          auto globals = parseIntList(testGlobals);
          for (size_t i = 0; i < globals.size(); i++) {
            (*newBuf)[i][0] = globals[i];
          }
        } else if (bufDesc.name.str() == "mix") {
          std::default_random_engine generator;
          std::uniform_int_distribution<int> distribution(1, zirgen::Zll::kFieldPrimeDefault - 1);
          newBuf->clear();
          newBuf->resize(bufDesc.regCount, Polynomial(4, zirgen::Zll::kFieldInvalid));
          for (size_t i = 0; i < bufDesc.regCount; i++) {
            for (size_t j = 0; j < 4; j++) {
              (*newBuf)[i][j] = static_cast<uint64_t>(distribution(generator));
            }
          }
        }
      } else {
        newBuf->resize(bufDesc.regCount * testCycles, Polynomial(1, zirgen::Zll::kFieldInvalid));
        interp.setNamedBuf(bufDesc.name, *newBuf, bufDesc.regCount);
      }
    }
    TestExternHandler testExterns;
    if (!inputDataHex.empty() && !inputDataFilename.empty()) {
      llvm::errs() << "Cannot specify both --input-data-file and --input-data-hex\n";
      exit(1);
    } else if (!inputDataFilename.empty()) {
      auto fileOrErr = llvm::MemoryBuffer::getFile(inputDataFilename);
      if (fileOrErr.getError()) {
        llvm::errs() << "Unable to read input data from " << inputDataFilename << "\n";
        exit(1);
      }
      testExterns.addInput((*fileOrErr)->getBuffer());
    } else if (!inputDataHex.empty()) {
      testExterns.addInput(llvm::fromHex(inputDataHex));
    }
    interp.setExternHandler(&testExterns);
    interp.setSilenceErrors(expectFailure);
    // interp.setDebug(true);
    bool failed = false;
    size_t cycle = 0;
    std::string exceptionName;
    for (; cycle != testCycles; ++cycle) {
      interp.setCycle(cycle);
      testExterns.cycle = cycle;
      if (mlir::failed(interp.runBlock(stepFuncOp.getBody().front()))) {
        failed = true;
        break;
      }
    }
    if (testExterns.lookups.size()) {
      llvm::errs() << "Lookups pending:\n";
      for (const auto& kvp : testExterns.lookups) {
        for (const auto& kvp2 : kvp.second) {
          llvm::errs() << "  lookup table " << kvp.first << ", entry " << kvp2.first
                       << ", value = " << kvp2.second << "\n";
        }
      }
    } else {
      llvm::errs() << "Lookups resolved\n";
    }

    // Run accum step for the test if there is one
    std::string name = (stepFuncOp.getName() + "$accum").str();
    auto accum = module.lookupSymbol<zirgen::Zhlt::StepFuncOp>(name);
    if (!failed && accum) {
      llvm::outs() << "run accum: " << name << "\n";
      cycle = 0;
      for (; cycle != testCycles; ++cycle) {
        interp.setCycle(cycle);
        if (mlir::failed(interp.runBlock(accum.getBody().front()))) {
          failed = true;
          break;
        }
      }

      // TODO: the accum code doesn't actually generate a constraint that the
      // accumulator grand sum starts and ends at the same value... We can't do
      // that until "circular constraints" work or the accum code generates extra
      // major mux arms for initialization and finalization. In the meantime,
      // print out the final accumulator sum so that we can assert on it in tests
      for (auto [buf, bufDesc] : llvm::zip(bufs, allBufs)) {
        if (bufDesc.name == "accum") {
          llvm::outs() << "final accum: [";
          llvm::interleaveComma(buf->back(), llvm::outs());
          llvm::outs() << "]\n";
        }
      }
    }

    if (failed != expectFailure) {
      if (!failed) {
        llvm::errs() << "Expected failure, but no failure found\n";
      } else {
        llvm::errs() << "Unexpected failure on cycle " << cycle << "\n";
      }
      return 1;
    }
  }
  return 0;
}

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);

  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  cl::ParseCommandLineOptions(argc, argv, "zirgen compiler\n");

  mlir::DialectRegistry registry;
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSymbolPrivatizePass();
  mlir::registerInlinerPass();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<zirgen::Zll::ZllDialect>();
  registry.insert<zirgen::ZStruct::ZStructDialect>();
  registry.insert<zirgen::Zhlt::ZhltDialect>();
  mlir::func::registerInlinerExtension(registry);

  mlir::MLIRContext context(registry);

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

  if (emitAction == Action::PrintAST) {
    ast->print(llvm::outs());
    return 0;
  }

  context.getOrLoadDialect<zirgen::Zhl::ZhlDialect>();
  std::optional<mlir::ModuleOp> zhlModule = zirgen::dsl::lower(context, sourceManager, ast.get());
  if (!zhlModule) {
    return 1;
  }

  if (emitAction == Action::PrintZHL) {
    zhlModule->print(llvm::outs());
    return 0;
  }

  context.getOrLoadDialect<zirgen::Zhlt::ZhltDialect>();
  context.getOrLoadDialect<zirgen::ZStruct::ZStructDialect>();
  context.getOrLoadDialect<zirgen::Zll::ZllDialect>();

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
  pm.addPass(zirgen::dsl::createGenerateAccumPass());
  pm.addPass(zirgen::dsl::createGenerateGlobalsPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  if (failed(pm.run(typedModule.value()))) {
    llvm::errs() << "an internal compiler error ocurred while type checking this module:\n";
    typedModule->print(llvm::errs());
    return 1;
  }

  if (emitAction == Action::PrintZHLT) {
    typedModule->print(llvm::outs());
    return 0;
  }

  pm.clear();
  // TODO: HoistAllocs is failing
  // pm.addPass(zirgen::Zhlt::createHoistAllocsPass());
  pm.addPass(zirgen::ZStruct::createOptimizeLayoutPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  if (failed(pm.run(typedModule.value()))) {
    llvm::errs() << "an internal error occurred while optimizing this module:\n";
    typedModule->print(llvm::errs());
    return 1;
  }

  if (emitAction == Action::OptimizeZHLT) {
    typedModule->print(llvm::outs());
    return 0;
  }

  pm.clear();
  if (!doTest)
    pm.addPass(zirgen::Zhlt::createStripTestsPass());
  pm.addPass(zirgen::dsl::createGenerateCheckLayoutPass());
  pm.addPass(zirgen::dsl::createGenerateLayoutPass());
  pm.addPass(zirgen::Zhlt::createStripAliasLayoutOpsPass());
  pm.addPass(zirgen::dsl::createGenerateBackPass());
  pm.addPass(zirgen::dsl::createGenerateExecPass());
  pm.addPass(mlir::createSymbolPrivatizePass({}));
  pm.addPass(zirgen::Zhlt::createGenerateStepsPass());
  pm.addPass(zirgen::dsl::createGenerateCheckPass());
  if (genValidity) {
    pm.addPass(zirgen::dsl::createGenerateTapsPass());
    pm.addPass(zirgen::dsl::createGenerateValidityRegsPass());
    pm.addPass(zirgen::dsl::createGenerateValidityTapsPass());
  }
  pm.addPass(zirgen::dsl::createElideTrivialStructsPass());
  pm.addPass(zirgen::ZStruct::createExpandLayoutPass());

  if (inlineLayout) {
    pm.nest<zirgen::Zhlt::CheckFuncOp>().addPass(zirgen::ZStruct::createInlineLayoutPass());
    if (genValidity) {
      pm.nest<zirgen::Zhlt::ValidityRegsFuncOp>().addPass(
          zirgen::ZStruct::createInlineLayoutPass());
      pm.nest<zirgen::Zhlt::ValidityTapsFuncOp>().addPass(
          zirgen::ZStruct::createInlineLayoutPass());
    }
  }

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());

  if (failed(pm.run(typedModule.value()))) {
    llvm::errs() << "an internal compiler error occurred while lowering this module:\n";
    typedModule->print(llvm::errs());
    return 1;
  }

  bool degreeExceeded = false;
  typedModule->walk([&](zirgen::Zhlt::CheckFuncOp op) {
    if (failed(op.verifyMaxDegree(maxDegree))) {
      degreeExceeded = true;
      // Ugh, apparently we don't get type aliases unless we print an
      // operation without a parent.  Copy this check function into a
      // blank module so we only get the problem check function without all the rest of the stuff.
      mlir::OpBuilder builder(&context);
      auto tmpMod = mlir::ModuleOp::create(builder.getUnknownLoc());
      builder.setInsertionPointToStart(&tmpMod.getBodyRegion().front());
      builder.clone(*op.getOperation());

      llvm::errs() << "Check function:\n";
      tmpMod->print(llvm::outs());
    }
  });
  if (degreeExceeded)
    return 1;

  if (emitAction == Action::PrintLayoutType) {
    if (auto topFunc = typedModule->lookupSymbol<zirgen::Zhlt::ExecFuncOp>("exec$Top")) {
      std::stringstream ss;
      mlir::Type lt = topFunc.getLayoutType();
      zirgen::layout::viz::layoutSizes(lt, ss);
      llvm::outs() << ss.str();
      return 0;
    } else {
      llvm::errs() << "error: circuit contains no component named `Top`\n";
      return 1;
    }
  } else if (emitAction == Action::PrintLayoutAttr) {
    std::stringstream ss;
    zirgen::layout::viz::layoutAttrs(*typedModule, ss);
    llvm::outs() << ss.str();
    return 0;
  } else if (emitAction == Action::PrintZStruct) {
    typedModule->print(llvm::outs());
    return 0;
  }

  if (emitAction == Action::PrintStats) {
    zirgen::dsl::printStats(*typedModule);
    return 0;
  }

  if (doTest) {
    return runTests(*typedModule);
  }

  if (emitAction == Action::PrintRust || emitAction == Action::PrintCpp) {
    codegen::CodegenOptions codegenOpts;
    static codegen::RustLanguageSyntax kRust;
    static codegen::CppLanguageSyntax kCpp;

    codegenOpts.lang = (emitAction == Action::PrintRust)
                           ? static_cast<codegen::LanguageSyntax*>(&kRust)
                           : static_cast<codegen::LanguageSyntax*>(&kCpp);

    zirgen::codegen::CodegenEmitter emitter(codegenOpts, &llvm::outs(), &context);
    if (zirgen::Zhlt::emitModule(*typedModule, emitter).failed()) {
      llvm::errs() << "Failed to emit circuit\n";
      return 1;
    }
  }

  return 0;
}
