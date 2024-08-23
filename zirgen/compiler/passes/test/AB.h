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

#pragma once

#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/compiler/edsl/edsl.h"
#include "zirgen/compiler/zkp/poseidon2.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <gtest/gtest.h>

namespace zirgen {

// A function for A/B testing of InlineFpExt pass.
// Basically, we run the code twice, once before and once after the pass and
// make sure the results are the same.
template <typename Func> void doInlineFpExtAB(const std::vector<uint32_t>& iopVals, Func userFunc) {
  // Compile normally
  std::array<ArgumentInfo, 1> argTypes = {ioparg()};
  Module module;
  module.addFunc<1>("test", argTypes, userFunc);
  module.optimize();
  // module.dump();

  // Make the two readIOPs
  auto readIop1 =
      std::make_unique<ReadIop>(std::make_unique<Poseidon2Rng>(), iopVals.data(), iopVals.size());
  auto readIop2 =
      std::make_unique<ReadIop>(std::make_unique<Poseidon2Rng>(), iopVals.data(), iopVals.size());

  // Run normally
  Zll::ExternHandler baseExternHandler;
  auto func = module.getModule().lookupSymbol<mlir::func::FuncOp>("test");
  Zll::Interpreter interp1(module.getCtx(), poseidon2HashSuite());
  // interp1.setDebug(true);
  interp1.setExternHandler(&baseExternHandler);
  interp1.setIop(func.getArgument(0), readIop1.get());
  if (failed(interp1.runBlock(func.front()))) {
    FAIL() << "failed to evaluate block in interpreter";
  }
  uint32_t out1 = readIop1->generateFp();
  readIop1->verifyComplete();
  llvm::errs() << "Out 1 = " << out1 << "\n";

  // Run the InlineFpExt pass
  mlir::PassManager pm(module.getCtx());
  mlir::OpPassManager& opm = pm.nest<mlir::func::FuncOp>();
  opm.addPass(Zll::createInlineFpExtPass());
  opm.addPass(Zll::createAddReductionsPass());
  if (failed(pm.run(module.getModule()))) {
    throw std::runtime_error("Failed to apply basic optimization passes");
  }
  module.optimize();
  // module.dump();

  // Run modified
  func = module.getModule().lookupSymbol<mlir::func::FuncOp>("test");
  Zll::Interpreter interp2(module.getCtx(), poseidon2HashSuite());
  // interp2.setDebug(true);
  interp2.setExternHandler(&baseExternHandler);
  interp2.setIop(func.getArgument(0), readIop2.get());
  if (failed(interp2.runBlock(func.front())))
    FAIL() << "failed to evaluate block in interpreter";
  uint32_t out2 = readIop2->generateFp();
  readIop2->verifyComplete();
  llvm::errs() << "Out 2 = " << out2 << "\n";

  // Print final op counts
  std::map<std::string, size_t> opCounts;
  for (mlir::Operation& op : func.front().without_terminator()) {
    opCounts[op.getName().getStringRef().str()]++;
  }
  for (const auto& kvp : opCounts) {
    llvm::errs() << kvp.first << ": " << kvp.second << "\n";
  }

  // Make sure results match
  ASSERT_EQ(out1, out2);
}

} // namespace zirgen
