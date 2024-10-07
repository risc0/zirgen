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
  module.setProtocolInfo(RV32IM_CIRCUIT_INFO);
  // module.optimize();
  // module.optimize(3);
  // module.dump(/*debug=*/true);
  // exit(1);
  module.optimize();

  EmitCodeOptions opts;
  // bazel expects output files to be named differently than the stages
  opts.stages["ram_verify"].outputFile = "verify_mem";
  opts.stages["bytes_verify"].outputFile = "verify_bytes";

  // Accum is processed in parallel, so we don't have access to
  // BACKs to be able to check constraints when executing.
  opts.stages["verify_accum"].addExtraPasses = [&](mlir::OpPassManager& opm) {
    opm.addPass(Zll::createDropConstraintsPass());
  };

  emitCode(module.getModule(), opts);
}
