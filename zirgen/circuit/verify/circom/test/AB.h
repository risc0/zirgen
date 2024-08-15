// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/Pass/PassManager.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/circuit/verify/circom/test/run_circom.h"
#include "zirgen/compiler/edsl/edsl.h"

#include <gtest/gtest.h>

namespace zirgen::snark {

inline void push_fp(std::vector<uint32_t>& iop, uint32_t val) {
  iop.push_back(uint64_t(val) * kBabyBearToMontgomery % kBabyBearP);
}

inline void push_digest(std::vector<uint32_t>& iop, Digest digest) {
  for (size_t i = 0; i < 8; i++) {
    iop.push_back(digest.words[i]);
  }
}

template <typename Func>
void doAB(size_t outSize, const std::vector<uint32_t>& input, Func userFunc) {
  if (system("which circom") != 0 || system("which snarkjs") != 0) {
    return;
  }
  // Make MLIR for the function
  std::array<ArgumentInfo, 2> argTypes;
  argTypes[0] = gbuf(outSize);
  argTypes[1] = ioparg();
  Module module;
  module.addFunc<2>("test", std::move(argTypes), userFunc);
  module.optimize();

  // Convert it to the form required for circom
  mlir::PassManager pm(module.getCtx());
  mlir::OpPassManager& opm = pm.nest<mlir::func::FuncOp>();
  opm.addPass(Zll::createInlineFpExtPass());
  opm.addPass(Zll::createAddReductionsPass());
  if (failed(pm.run(module.getModule()))) {
    throw std::runtime_error("Failed to apply basic optimization passes");
  }

  // Get the function
  auto func = module.getModule().lookupSymbol<mlir::func::FuncOp>("test");
  // module.dump();

  // Run the code directly in the interpreter
  Zll::ExternHandler baseExternHandler;
  Zll::Interpreter interp(module.getCtx(), poseidon254HashSuite());
  interp.setExternHandler(&baseExternHandler);
  auto outBuf = interp.makeBuf(func.getArgument(0), outSize, Zll::BufferKind::Global);
  ReadIop riop(interp.getHashSuite().makeRng(), input.data(), input.size());
  interp.setIop(func.getArgument(1), &riop);
  if (failed(interp.runBlock(func.front())))
    FAIL() << "failed to evaluate block in interpreter";

  // Print outputs
  for (size_t i = 0; i < outSize; i++) {
    llvm::errs() << "out[" << i << "] = " << outBuf[i][0] << "\n";
  }
  // Run it as circom code
  std::string tmpdir = std::getenv("TEST_TMPDIR");
  auto out2 = run_circom(func, input, tmpdir);

  // Print outputs
  assert(out2.size() == outSize);
  for (size_t i = 0; i < outSize; i++) {
    llvm::errs() << "out[" << i << "] = " << out2[i] << "\n";
    assert(out2[i] == outBuf[i][0]);
  }
}

} // namespace zirgen::snark
