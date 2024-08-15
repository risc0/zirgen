// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/Zll/IR/IR.h"

#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/circuit/rv32im/v1/edsl/top.h"
#include "zirgen/circuit/rv32im/v1/platform/constants.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/codegen/protocol_info_const.h"
#include "zirgen/compiler/edsl/edsl.h"

using namespace zirgen;
using namespace zirgen::rv32im_v1;
using namespace risc0;
using namespace mlir;

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);
  registerEdslCLOptions();
  registerCodegenCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "rv32im edsl");

  Module module;
  module.addFunc<5>(
      "rv32im",
      {cbuf(kCodeSize), gbuf(kInOutSize), mbuf(kDataSize), gbuf(kMixSize), mbuf(kAccumSize)},
      [&](Buffer code, Buffer out, Buffer data, Buffer mix, Buffer accum) {
        CompContext::init({"_ram_finalize",
                           "ram_verify",
                           "_bytes_finalize",
                           "bytes_verify",
                           "compute_accum",
                           "verify_accum"});

        CompContext::addBuffer("code", code);
        CompContext::addBuffer("out", out);
        CompContext::addBuffer("data", data);
        CompContext::addBuffer("mix", mix);
        CompContext::addBuffer("accum", accum);

        Top top;
        Val isHalt = top->set();

        CompContext::fini(isHalt);
        CompContext::emitLayout(top);
      });
  // module.optimize();
  // module.optimize(3);
  // module.dump(/*debug=*/true);
  // exit(1);
  module.optimize();

  EmitCodeOptions opts = {
      .info = RV32IM_CIRCUIT_INFO,
      .stages = {{"exec", "verify_mem", "verify_bytes", "compute_accum", "verify_accum"}}};

  // Accum is processed in parallel, so we don't have access to
  // BACKs to be able to check constraints when executing.
  opts.stages[4 /* compute_accum*/].addExtraPasses = [&](mlir::OpPassManager& opm) {
    opm.addPass(Zll::createDropConstraintsPass());
  };

  emitCode(module.getModule(), opts);
}
