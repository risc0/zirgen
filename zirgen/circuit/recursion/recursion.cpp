// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/Dialect/Zll/IR/IR.h"
#include "zirgen/Dialect/Zll/Transforms/Passes.h"
#include "zirgen/circuit/recursion/top.h"
#include "zirgen/compiler/codegen/codegen.h"
#include "zirgen/compiler/codegen/protocol_info_const.h"

using namespace zirgen;
using namespace zirgen::recursion;
using namespace risc0;
using namespace mlir;

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);
  registerEdslCLOptions();
  registerCodegenCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "recursion edsl");

  Module module;
  module.addFunc<5>(
      "recursion",
      {cbuf(kCodeSize), gbuf(kOutSize), mbuf(kDataSize), gbuf(kMixSize), mbuf(kAccumSize)},
      [&](Buffer code, Buffer out, Buffer data, Buffer mix, Buffer accum) {
        CompContext::init({"_wom_finalize",
                           "wom_verify",
                           "do_nothing",
                           "compute_accum",
                           "verify_accum",
                           "_green_wire"});

        CompContext::addBuffer("code", code);
        CompContext::addBuffer("out", out);
        CompContext::addBuffer("data", data);
        CompContext::addBuffer("mix", mix);
        CompContext::addBuffer("accum", accum);

        recursion::Top top(Label("top"));
        top->set();
        Val isHalt = 0;

        CompContext::fini(isHalt);
        CompContext::emitLayout(top);
      });
  module.optimize();
  // module.optimize();
  // module.optimize(3);
  // module.dump(/*debug=*/true);
  // exit(1);

  EmitCodeOptions opts = {
      .info = RECURSION_CIRCUIT_INFO,
      .stages = {{"exec", "verify_mem", "verify_bytes", "compute_accum", "verify_accum"}}};

  opts.stages[0 /* exec */].addExtraPasses = [&](mlir::OpPassManager& opm) {
    // For speed, don't check constraints during exec phase.
    opm.addPass(Zll::createDropConstraintsPass());
  };

  opts.stages[4 /* compute_accum */].addExtraPasses = [&](mlir::OpPassManager& opm) {
    // Accum operates in parallel, so don't check constraints during it.
    opm.addPass(Zll::createDropConstraintsPass());
  };

  emitCode(module.getModule(), opts);
}
