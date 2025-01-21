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
#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/verify/merkle.h"
#include "zirgen/circuit/verify/wrap_recursion.h"
#include "zirgen/circuit/verify/wrap_rv32im.h"
#include "zirgen/circuit/verify/wrap_zirgen.h"
#include "zirgen/compiler/codegen/codegen.h"

using namespace zirgen;
using namespace zirgen::verify;
using namespace zirgen::predicates;
using Zll::DigestKind;
namespace cl = llvm::cl;

VerifyInfo
verifyAndValidate(DigestVal root, ReadIopVal seal, size_t po2, const CircuitInterface& circuit) {
  auto info = zirgen::verify::verify(seal, po2, circuit);
  Val codeRootIndex = seal.readBaseVals(1)[0];
  auto merklePath = seal.readDigests(recursion::kAllowedCodeMerkleDepth);
  verifyMerkleGroupMember(info.codeRoot, codeRootIndex, merklePath, root);
  return info;
}

template <typename T>
static T getRecursiveObj(DigestVal root, ReadIopVal seal, const CircuitInterface& circuit) {
  auto info = verifyAndValidate(root, seal, recursion::kRecursionPo2, circuit);
  DigestVal innerRoot = intoDigest(llvm::ArrayRef<Val>(info.out).slice(0, 16), DigestKind::Default);
  DigestVal innerOut = intoDigest(llvm::ArrayRef<Val>(info.out).slice(16, 16), DigestKind::Sha256);
  auto dataVec = seal.readBaseVals(T::size);
  llvm::ArrayRef<Val> stream(dataVec);
  T data(stream);
  assert_eq(data.digest(), innerOut);
  assert_eq(innerRoot, root);
  return data;
}

// Verify the recursion VM seal encoded in the IOP and return the assumption that it verifies.
// The caller must check the root on the returned assumption and decide if it is acceptable.
static Assumption
verifyAssumption(DigestVal condRoot, ReadIopVal seal, const CircuitInterface& circuit) {
  // Does the same work as verify and validate, without having a root to check against.
  auto info = zirgen::verify::verify(seal, recursion::kRecursionPo2, circuit);
  Val codeRootIndex = seal.readBaseVals(1)[0];
  auto merklePath = seal.readDigests(recursion::kAllowedCodeMerkleDepth);
  auto calculatedRoot = calculateMerkleProofRoot(info.codeRoot, codeRootIndex, merklePath);

  // Decode the two globals and ensure the first is equal to the calculated root.
  DigestVal innerRoot = intoDigest(llvm::ArrayRef<Val>(info.out).slice(0, 16), DigestKind::Default);
  DigestVal innerOut = intoDigest(llvm::ArrayRef<Val>(info.out).slice(16, 16), DigestKind::Sha256);
  assert_eq(innerRoot, calculatedRoot);

  // Read an additional boolean Val to indicate whether the assumption control
  // root is zero. If it is zero, this means "default" and the root that was
  // calculatedRoot needs to match the control root on the conditional receipt.
  // Otherwise, we return the control root as is and the resolve check will
  // ensure consistency with the conditional claim.
  Val zeroAssumeRoot = seal.readBaseVals(1)[0];
  eqz(zeroAssumeRoot * (1 - zeroAssumeRoot));
  std::vector zeroVec(16, Val(0));
  auto zeroHash = intoDigest(zeroVec, DigestKind::Default);
  assert_eq(select(zeroAssumeRoot, {condRoot, calculatedRoot}), condRoot);
  DigestVal assumRoot = select(zeroAssumeRoot, {calculatedRoot, zeroHash});

  Assumption assum;
  assum.claim = innerOut;
  assum.controlRoot = assumRoot;
  return assum;
}

template <typename T> static void writeOutObj(Buffer out, T outData) {
  std::vector<Val> outStream;
  outData.write(outStream);
  doExtern("write", "", 0, outStream);
  out.setDigest(1, outData.digest(), "outDigest");
}

template <typename Func> void addRv32imV1Lift(Module& module, const std::string& name, Func func) {
  for (size_t po2 = 14; po2 < 25; ++po2) {
    module.addFunc<3>(name + "_" + std::to_string(po2),
                      {gbuf(recursion::kOutSize), ioparg(), ioparg()},
                      [&](Buffer out, ReadIopVal rootIop, ReadIopVal rv32seal) {
                        auto circuit = getInterfaceRV32IM();
                        DigestVal root = rootIop.readDigests(1)[0];
                        VerifyInfo info = verifyAndValidate(root, rv32seal, po2, *circuit);
                        llvm::ArrayRef inStream(info.out);
                        ReceiptClaim claim(inStream, true);
                        auto outData = func(claim);
                        writeOutObj(out, outData);
                        out.setDigest(0, root, "root");
                      });
  }
}

void addRv32imV2Lift(Module& module, const std::string name, const std::string& irPath) {
  auto circuit = getInterfaceZirgen(module.getModule().getContext(), irPath);
  for (size_t po2 = 14; po2 < 25; ++po2) {
    module.addFunc<3>(name + "_" + std::to_string(po2),
                      {gbuf(recursion::kOutSize), ioparg(), ioparg()},
                      [&](Buffer out, ReadIopVal rootIop, ReadIopVal seal) {
                        DigestVal root = rootIop.readDigests(1)[0];
                        VerifyInfo info = zirgen::verify::verify(seal, po2, *circuit);
                        llvm::ArrayRef inStream(info.out);
                        ReceiptClaim claim = ReceiptClaim::fromRv32imV2(inStream, po2);
                        writeOutObj(out, claim);
                        out.setDigest(0, root, "root");
                      });
  }
}

template <typename Func> void addJoin(Module& module, const std::string& name, Func func) {
  module.addFunc<4>(name,
                    {gbuf(recursion::kOutSize), ioparg(), ioparg(), ioparg()},
                    [&](Buffer out, ReadIopVal rootIop, ReadIopVal in1, ReadIopVal in2) {
                      auto circuit = getInterfaceRecursion();
                      DigestVal root = rootIop.readDigests(1)[0];
                      ReceiptClaim val1 = getRecursiveObj<ReceiptClaim>(root, in1, *circuit);
                      ReceiptClaim val2 = getRecursiveObj<ReceiptClaim>(root, in2, *circuit);
                      auto outData = func(val1, val2);
                      writeOutObj(out, outData);
                      out.setDigest(0, root, "root");
                    });
}

template <typename Func> void addResolve(Module& module, const std::string& name, Func func) {
  module.addFunc<6>(name,
                    {gbuf(recursion::kOutSize), ioparg(), ioparg(), ioparg(), ioparg(), ioparg()},
                    [&](Buffer out,
                        ReadIopVal rootIop,
                        ReadIopVal condIop,
                        ReadIopVal assumIop,
                        ReadIopVal tailIop,
                        ReadIopVal journalIop) {
                      auto circuit = getInterfaceRecursion();
                      DigestVal root = rootIop.readDigests(1)[0];
                      ReceiptClaim cond = getRecursiveObj<ReceiptClaim>(root, condIop, *circuit);
                      Assumption assum = verifyAssumption(root, assumIop, *circuit);

                      // NOTE: readBaseVals is used here instead of readDigest
                      // because we need to read in the SHA-256 digest as
                      // halves and then call intoDigest. Poseidon2 digests can
                      // be read in directly since they are encoded as words.
                      DigestVal tail = intoDigest(tailIop.readBaseVals(16), DigestKind::Sha256);
                      DigestVal journal =
                          intoDigest(journalIop.readBaseVals(16), DigestKind::Sha256);
                      auto outData = func(cond, assum, tail, journal);
                      writeOutObj(out, outData);
                      out.setDigest(0, root, "root");
                    });
}

template <typename Func> void addSingleton(Module& module, const std::string& name, Func func) {
  module.addFunc<3>(name,
                    {gbuf(recursion::kOutSize), ioparg(), ioparg()},
                    [&](Buffer out, ReadIopVal rootIop, ReadIopVal in) {
                      auto circuit = getInterfaceRecursion();
                      DigestVal root = rootIop.readDigests(1)[0];
                      ReceiptClaim val = getRecursiveObj<ReceiptClaim>(root, in, *circuit);
                      auto outData = func(val);
                      writeOutObj(out, outData);
                      out.setDigest(0, root, "root");
                    });
}

template <typename Func> void addUnion(Module& module, const std::string& name, Func func) {
  module.addFunc<4>(name,
                    {gbuf(recursion::kOutSize), ioparg(), ioparg(), ioparg()},
                    [&](Buffer out, ReadIopVal rootIop, ReadIopVal leftIop, ReadIopVal rightIop) {
                      auto circuit = getInterfaceRecursion();
                      DigestVal root = rootIop.readDigests(1)[0];
                      Assumption left = verifyAssumption(root, leftIop, *circuit);
                      Assumption right = verifyAssumption(root, rightIop, *circuit);

                      auto outData = func(left, right);
                      writeOutObj(out, outData);
                      out.setDigest(0, root, "root");
                    });
}

static cl::opt<std::string>
    outputDir("output-dir", cl::desc("Output directory"), cl::value_desc("dir"), cl::Required);

static cl::opt<std::string>
    rv32imV2IR("rv32im-v2-ir",
               cl::desc("rv32im-v2 validity polynomial IR"),
               cl::value_desc("path"),
               cl::init("bazel-bin/zirgen/circuit/rv32im/v2/dsl/validity.ir"));

int main(int argc, char* argv[]) {
  llvm::InitLLVM y(argc, argv);
  registerEdslCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "gen_predicates edsl");

  Module module;
  module.addFunc<2>("test_recursion_circuit",
                    {gbuf(recursion::kOutSize), ioparg()},
                    [&](Buffer out, ReadIopVal iop) {
                      auto hashes = iop.readDigests(2);

                      auto folded = fold(hashes[0], hashes[1]);
                      auto claim = taggedStruct("risc0.TestRecursionCircuit", {folded}, {});

                      // Test folding together the two input digests.
                      out.setDigest(0, hashes[0], "root");
                      out.setDigest(1, claim, "claim");
                    });

  addRv32imV1Lift(module, "lift", [](ReceiptClaim claim) { return claim; });
  addRv32imV2Lift(module, "lift_rv32im_v2", rv32imV2IR.getValue());

  addJoin(module, "join", [&](ReceiptClaim a, ReceiptClaim b) { return join(a, b); });

  addResolve(
      module, "resolve", [&](ReceiptClaim a, Assumption b, DigestVal tail, DigestVal journal) {
        return resolve(a, b, tail, journal);
      });

  addSingleton(module, "identity", [&](ReceiptClaim a) { return identity(a); });

  addUnion(
      module, "union", [&](Assumption left, Assumption right) { return unionFunc(left, right); });

  mlir::PassManager pm(module.getModule()->getContext());
  if (failed(applyPassManagerCLOptions(pm))) {
    exit(1);
  }
  module.addOptimizationPasses(pm);
  pm.nest<mlir::func::FuncOp>().addPass(createEmitRecursionPass(outputDir));

  if (failed(pm.run(module.getModule()))) {
    llvm::errs() << "Unable to run recursion pipeline\n";
    exit(1);
  }
}
