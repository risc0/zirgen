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
#include "zirgen/circuit/verify/merkle.h"
#include "zirgen/circuit/verify/wrap_recursion.h"
#include "zirgen/circuit/verify/wrap_rv32im.h"
#include "zirgen/circuit/verify/wrap_zirgen.h"
#include "zirgen/compiler/codegen/codegen.h"

using namespace zirgen;
using namespace zirgen::verify;
using namespace zirgen::predicates;
// using Zll::DigestKind;

namespace cl = llvm::cl;

// VerifyInfo
// verifyAndValidate(DigestVal root, ReadIopVal seal, size_t po2, const CircuitInterface& circuit) {
//   auto info = zirgen::verify::verify(seal, po2, circuit);
//   Val codeRootIndex = seal.readBaseVals(1)[0];
//   auto merklePath = seal.readDigests(recursion::kAllowedCodeMerkleDepth);
//   verifyMerkleGroupMember(info.codeRoot, codeRootIndex, merklePath, root);
//   return info;
// }

// template <typename T>
// static T getRecursiveObj(DigestVal root, ReadIopVal seal, const CircuitInterface& circuit) {
//   auto info = verifyAndValidate(root, seal, recursion::kRecursionPo2, circuit);
//   DigestVal innerRoot = intoDigest(llvm::ArrayRef<Val>(info.out).slice(0, 16),
//   DigestKind::Default); DigestVal innerOut = intoDigest(llvm::ArrayRef<Val>(info.out).slice(16,
//   16), DigestKind::Sha256); auto dataVec = seal.readBaseVals(T::size); llvm::ArrayRef<Val>
//   stream(dataVec); T data(stream); assert_eq(data.digest(), innerOut); assert_eq(innerRoot,
//   root); return data;
// }

// // Verify the recursion VM seal encoded in the IOP and return the assumption that it verifies.
// // The caller must check the root on the returned assumption and decide if it is acceptable.
// static Assumption
// verifyAssumption(DigestVal condRoot, ReadIopVal seal, const CircuitInterface& circuit) {
//   // Does the same work as verify and validate, without having a root to check against.
//   auto info = zirgen::verify::verify(seal, recursion::kRecursionPo2, circuit);
//   Val codeRootIndex = seal.readBaseVals(1)[0];
//   auto merklePath = seal.readDigests(recursion::kAllowedCodeMerkleDepth);
//   auto calculatedRoot = calculateMerkleProofRoot(info.codeRoot, codeRootIndex, merklePath);

//   // Decode the two globals and ensure the first is equal to the calculated root.
//   DigestVal innerRoot = intoDigest(llvm::ArrayRef<Val>(info.out).slice(0, 16),
//   DigestKind::Default); DigestVal innerOut = intoDigest(llvm::ArrayRef<Val>(info.out).slice(16,
//   16), DigestKind::Sha256); assert_eq(innerRoot, calculatedRoot);

//   // Read an additional boolean Val to indicate whether the assumption control
//   // root is zero. If it is zero, this means "default" and the root that was
//   // calculatedRoot needs to match the control root on the conditional receipt.
//   // Otherwise, we return the control root as is and the resolve check will
//   // ensure consistency with the conditional claim.
//   Val zeroAssumeRoot = seal.readBaseVals(1)[0];
//   eqz(zeroAssumeRoot * (1 - zeroAssumeRoot));
//   std::vector zeroVec(16, Val(0));
//   auto zeroHash = intoDigest(zeroVec, DigestKind::Default);
//   assert_eq(select(zeroAssumeRoot, {condRoot, calculatedRoot}), condRoot);
//   DigestVal assumRoot = select(zeroAssumeRoot, {calculatedRoot, zeroHash});

//   Assumption assum;
//   assum.claim = innerOut;
//   assum.controlRoot = assumRoot;
//   return assum;
// }

// template <typename T> static void writeOutObj(Buffer out, T outData) {
//   std::vector<Val> outStream;
//   outData.write(outStream);
//   doExtern("write", "", 0, outStream);
//   out.setDigest(1, outData.digest(), "outDigest");
// }

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
