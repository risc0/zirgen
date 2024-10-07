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
                           "verify_bytes",
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
  module.setProtocolInfo(RECURSION_CIRCUIT_INFO);
  module.optimize();
  // module.optimize();
  // module.optimize(3);
  // module.dump(/*debug=*/true);
  // exit(1);

  EmitCodeOptions opts;

  opts.stages["exec"].addExtraPasses = [&](mlir::OpPassManager& opm) {
    // For speed, don't check constraints during exec phase.
    opm.addPass(Zll::createDropConstraintsPass());
  };

  // bazel expects the wom_verify step to be in a file called "verify_mem"
  opts.stages["wom_verify"].outputFile = "verify_mem";

  opts.stages["verify_accum"].addExtraPasses = [&](mlir::OpPassManager& opm) {
    // Accum operates in parallel, so don't check constraints during it.
    opm.addPass(Zll::createDropConstraintsPass());
  };

  emitCode(module.getModule(), opts);
}
