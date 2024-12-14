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
static T
getRecursiveObj(DigestVal root, ReadIopVal seal, size_t po2, const CircuitInterface& circuit) {
  auto info = verifyAndValidate(root, seal, po2, circuit);
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
verifyAssumption(DigestVal condRoot, ReadIopVal seal, size_t po2, const CircuitInterface& circuit) {
  // Does the same work as verify and validate, without having a root to check against.
  auto info = zirgen::verify::verify(seal, po2, circuit);
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

// Add a program that verifies n RV32IM receipts with the given po2, then passes the claim into the
// given func.
template <typename Func>
void addLift(Module& module, const std::string name, size_t po2, Func func) {
  module.addFunc<3>(name,
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

// Add a program that verifies n RV32IM receipts with the given po2, then passes the claims into the
// provided func.
template <typename Func>
void addLiftJoin(Module& module, const std::string name, size_t n, size_t po2, Func func) {
  module.addFunc<3>(name,
                    {gbuf(recursion::kOutSize), ioparg(), ioparg()},
                    [&](Buffer out, ReadIopVal rootIop, ReadIopVal in) {
                      auto circuit = getInterfaceRV32IM();
                      // Read the control root.
                      DigestVal root = rootIop.readDigests(1)[0];
                      // Verify and extract the receipt claims.
                      std::vector<ReceiptClaim> claims;
                      for (size_t i = 0; i < n; ++i) {
                        VerifyInfo info = verifyAndValidate(root, in, po2, *circuit);
                        llvm::ArrayRef inStream(info.out);
                        claims.emplace_back(inStream, true);
                      }
                      // Run the (join) logic to verify the claims and construct the output.
                      auto outData = func(claims);
                      writeOutObj(out, outData);
                      out.setDigest(0, root, "root");
                    });
}

// Add a program that verifies n recursion receipts at the given po2, then passes the claims into
// the provided func.
template <typename Func>
void addJoin(Module& module, const std::string name, size_t n, size_t po2, Func func) {
  module.addFunc<3>(name,
                    {gbuf(recursion::kOutSize), ioparg(), ioparg()},
                    [&](Buffer out, ReadIopVal rootIop, ReadIopVal in) {
                      auto circuit = getInterfaceRecursion();
                      // Read the control root.
                      DigestVal root = rootIop.readDigests(1)[0];
                      // Verify and extract the receipt claims.
                      std::vector<ReceiptClaim> claims;
                      for (size_t i = 0; i < n; ++i) {
                        claims.push_back(getRecursiveObj<ReceiptClaim>(root, in, po2, *circuit));
                      }
                      // Run the (join) logic to verify the claims and construct the output.
                      auto outData = func(claims);
                      writeOutObj(out, outData);
                      out.setDigest(0, root, "root");
                    });
}

// Add a program that verifies one in-tree recursion receipt and one out-of-tree recursion receipt
// (assumption), both at the given po2.
template <typename Func>
void addResolve(Module& module, const std::string name, size_t po2, Func func) {
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
                      ReceiptClaim cond =
                          getRecursiveObj<ReceiptClaim>(root, condIop, po2, *circuit);
                      Assumption assum = verifyAssumption(root, assumIop, po2, *circuit);

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

// Add a program that verifies a recursion receipt at the given po2, then passes the claim into the
// provided func.
template <typename Func>
void addSingleton(Module& module, const std::string name, size_t po2, Func func) {
  module.addFunc<3>(name,
                    {gbuf(recursion::kOutSize), ioparg(), ioparg()},
                    [&](Buffer out, ReadIopVal rootIop, ReadIopVal in) {
                      auto circuit = getInterfaceRecursion();
                      DigestVal root = rootIop.readDigests(1)[0];
                      ReceiptClaim val = getRecursiveObj<ReceiptClaim>(root, in, po2, *circuit);
                      auto outData = func(val);
                      writeOutObj(out, outData);
                      out.setDigest(0, root, "root");
                    });
}

// Add a program that verifies two out-of-tree recursion receipts at the given po2, then passes the
// claims into the provided func.
template <typename Func>
void addUnion(Module& module, const std::string name, size_t po2, Func func) {
  module.addFunc<4>(name,
                    {gbuf(recursion::kOutSize), ioparg(), ioparg(), ioparg()},
                    [&](Buffer out, ReadIopVal rootIop, ReadIopVal leftIop, ReadIopVal rightIop) {
                      auto circuit = getInterfaceRecursion();
                      DigestVal root = rootIop.readDigests(1)[0];
                      Assumption left = verifyAssumption(root, leftIop, po2, *circuit);
                      Assumption right = verifyAssumption(root, rightIop, po2, *circuit);

                      auto outData = func(left, right);
                      writeOutObj(out, outData);
                      out.setDigest(0, root, "root");
                    });
}

static cl::opt<std::string>
    outputDir("output-dir", cl::desc("Output directory"), cl::value_desc("dir"), cl::Required);

const size_t MIN_RV32IM_PO2 = 14;
const size_t MAX_RV32IM_PO2 = 24;
const size_t MIN_RECURSION_PO2 = 18;
const size_t MAX_RECURSION_PO2 = 21;
const size_t MAX_JOIN_WIDTH = 12;

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

  // Add the programs that verify RVM32IM receipts.
  for (size_t po2 = MIN_RV32IM_PO2; po2 <= MAX_RV32IM_PO2; ++po2) {
    addLift(module, "lift_" + std::to_string(po2), po2, [](ReceiptClaim claim) { return claim; });

    for (size_t n = 2; n <= MAX_JOIN_WIDTH; ++n) {
      addLiftJoin(module,
                  "lift_join" + std::to_string(n) + "_" + std::to_string(po2),
                  n,
                  po2,
                  [&](llvm::ArrayRef<ReceiptClaim> claims) { return join(claims); });
    }
  }

  // Add the programs that verify recursion receipts.
  for (size_t po2 = MIN_RECURSION_PO2; po2 <= MAX_RECURSION_PO2; ++po2) {
    addSingleton(module, "identity_" + std::to_string(po2), po2, [&](ReceiptClaim a) {
      return identity(a);
    });

    for (size_t n = 2; n <= MAX_JOIN_WIDTH; ++n) {
      addJoin(module,
              "join" + std::to_string(n) + "_" + std::to_string(po2),
              n,
              po2,
              [&](llvm::ArrayRef<ReceiptClaim> claims) { return join(claims); });
    }

    addResolve(module,
               "resolve_" + std::to_string(po2),
               po2,
               [&](ReceiptClaim a, Assumption b, DigestVal tail, DigestVal journal) {
                 return resolve(a, b, tail, journal);
               });

    addUnion(module, "union_" + std::to_string(po2), po2, [&](Assumption left, Assumption right) {
      return unionFunc(left, right);
    });
  }

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
