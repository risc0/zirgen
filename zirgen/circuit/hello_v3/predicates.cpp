// Copyright 2025 RISC Zero, Inc.
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

#include "zirgen/circuit/predicates/predicates.h"
#include "zirgen/circuit/hello_v3/CircuitInterface.h"
#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/verify/verify.h"
#include "zirgen/compiler/codegen/codegen.h"
#include <memory>

using namespace zirgen;
using namespace zirgen::verify;
using namespace zirgen::predicates;

namespace cl = llvm::cl;

static cl::opt<std::string>
    outputDir("output-dir", cl::desc("Output directory"), cl::value_desc("dir"), cl::Required);

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);
  registerEdslCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "keccak predicates");

  Module module;
  size_t po2 = 12;
  std::unique_ptr<CircuitInterfaceV3> circuit = hello_v3::getCircuitInterface();
  module.addFunc<3>("hello_lift_12",
                    {gbuf(recursion::kOutSize), ioparg(), ioparg()},
                    [&](Buffer out, ReadIopVal rootIop, ReadIopVal seal) {
                      DigestVal root = rootIop.readDigests(1)[0];
                      zirgen::verify::verifyV3(seal, po2, *circuit);
                      out.setDigest(0, root, "root");
                    });

  module.optimize();
  module.getModule().walk([&](mlir::func::FuncOp func) { zirgen::emitRecursion(outputDir, func); });
}
