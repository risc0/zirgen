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

#include "zirgen/circuit/predicates/predicates.h"
#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/verify/wrap_zirgen.h"
#include "zirgen/compiler/codegen/codegen.h"

using namespace zirgen;
using namespace zirgen::verify;
using namespace zirgen::predicates;

namespace cl = llvm::cl;

template <typename Func>
void addZirgenLift(Module& module, const std::string name, const std::string path, Func func) {
  auto circuit = getInterfaceZirgen(module.getModule().getContext(), path);
  for (size_t po2 = 14; po2 < 19; ++po2) {
    module.addFunc<3>(name + "_" + std::to_string(po2),
                      {gbuf(recursion::kOutSize), ioparg(), ioparg()},
                      [&](Buffer out, ReadIopVal rootIop, ReadIopVal zirgenSeal) {
                        DigestVal root = rootIop.readDigests(1)[0];
                        VerifyInfo info = zirgen::verify::verify(zirgenSeal, po2, *circuit);
                        llvm::ArrayRef inStream(info.out);
                        DigestVal outData = func(inStream);
                        std::vector<Val> outStream;
                        writeSha(outData, outStream);
                        doExtern("write", "", 0, outStream);
                        out.setDigest(1, outData, "outDigest");
                        out.setDigest(0, root, "root");
                      });
  }
}

static cl::opt<std::string>
    outputDir("output-dir", cl::desc("Output directory"), cl::value_desc("dir"), cl::Required);

static cl::opt<std::string> keccakIR("keccak-ir",
                                     cl::desc("Keccak validity polynomial IR"),
                                     cl::value_desc("path"),
                                     cl::init("bazel-bin/zirgen/circuit/keccak/validity.ir"));

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);
  registerEdslCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "keccak predicates");

  Module module;
  addZirgenLift(module, "keccak_lift", keccakIR.getValue(), [](llvm::ArrayRef<Val>& inStream) {
    return readSha(inStream);
  });

  module.optimize();
  module.getModule().walk([&](mlir::func::FuncOp func) { zirgen::emitRecursion(outputDir, func); });
}
