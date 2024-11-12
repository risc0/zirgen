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

#include <gtest/gtest.h>

#include "zirgen/Dialect/ZHLT/IR/ZHLT.h"
#include "zirgen/circuit/verify/verify.h"
#include "zirgen/circuit/verify/wrap_zirgen.h"

using namespace zirgen;
using namespace zirgen::verify;
using namespace zirgen::Zll;

TEST(verify_zirgen, calculator) {
  // Read the seal if it exists
  FILE* file = fopen("/tmp/calc.seal", "rb");
  if (!file) {
    // TODO: Make this test always run, right now we skip if file doesn't exist
    std::cerr << "Didn't find file: /tmp/calc.seal, to generate run:\n";
    std::cerr << "cargo run --example calculator -- --seal /tmp/calc.seal\n";
    return;
  }
  fseek(file, 0, SEEK_END);
  size_t size = ftell(file) / 4;
  fseek(file, 0, SEEK_SET);
  std::vector<uint32_t> proof(size);
  size_t nread = fread(proof.data(), 4, size, file);
  ASSERT_EQ(nread, size);
  fclose(file);

  Module module;
  mlir::MLIRContext* ctx = module.getModule().getContext();
  ctx->getOrLoadDialect<Zhlt::ZhltDialect>();

  // Extract the PO2 from its known position
  auto circuit = getInterfaceZirgen(module.getModule().getContext(),
                                    "zirgen/dsl/examples/calculator/validity.ir");
  size_t po2 = proof[circuit->out_size()];

  std::cerr << "po2 = " << po2 << "\n";
  // Compile the verifiers
  module.addFunc<1>(
      "verify", {ioparg()}, [&](ReadIopVal iop) { zirgen::verify::verify(iop, po2, *circuit); });
  module.optimize();
  // module.dump();

  // Run the verifier
  auto func = module.getModule().lookupSymbol<mlir::func::FuncOp>("verify");
  ExternHandler baseExternHandler;
  Interpreter interp(module.getCtx(), poseidon2HashSuite());
  interp.setExternHandler(&baseExternHandler);
  auto rng = interp.getHashSuite().makeRng();
  ReadIop riop(std::move(rng), proof.data(), proof.size());
  interp.setIop(func.getArgument(0), &riop);
  // interp.setDebug(true);
  if (failed(interp.runBlock(func.front())))
    FAIL() << "failed to evaluate block in interpreter";

  // Compute some stats
  std::map<std::string, size_t> opCounts;
  size_t totCycles = 0;
  size_t hashCycles = 0;
  // size_t hashInitCost = 4;
  // size_t hashPerBlockCost = 68;
  size_t hashInitCost = 0;
  size_t hashPerBlockCost = 12;
  for (mlir::Operation& op : func.front().without_terminator()) {
    if (auto hashOp = llvm::dyn_cast<HashOp>(op)) {
      size_t count = hashOp.getIn().size();
      size_t k = llvm::cast<ValType>(hashOp.getIn()[0].getType()).getFieldK();
      size_t blocks = ((count * k) + 15) / 16;
      totCycles += hashInitCost + blocks * hashPerBlockCost;
      hashCycles += hashInitCost + blocks * hashPerBlockCost;
    }
    if (auto hashOp = llvm::dyn_cast<HashFoldOp>(op)) {
      totCycles += hashInitCost + hashPerBlockCost;
      hashCycles += hashInitCost + hashPerBlockCost;
    }
    if (auto iopReadOp = llvm::dyn_cast<HashFoldOp>(op)) {
      totCycles += hashInitCost + hashPerBlockCost;
      hashCycles += hashInitCost + hashPerBlockCost;
    }
    opCounts[op.getName().getStringRef().str()]++;
    totCycles++;
  }
  std::cout << "Tot cycles = " << totCycles << "\n";
  std::cout << "Hash cycles = " << hashCycles << "\n";
  for (const auto& kvp : opCounts) {
    std::cout << kvp.first << ": " << kvp.second << "\n";
  }
}
