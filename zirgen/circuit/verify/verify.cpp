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

#include "zirgen/circuit/verify/verify.h"

#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/verify/fri.h"
#include "zirgen/circuit/verify/merkle.h"
#include "zirgen/circuit/verify/poly.h"
#include "zirgen/compiler/codegen/protocol_info_const.h"

namespace zirgen::verify {

namespace {

// If true, emit debug logging.  NOTE: This will change the code root of predicates, so
// should not be left on when generating release ZKRs.
constexpr bool kDebug = false;

// Version of XLOG that returns its single value, so it can be used inline like rust's dbg! macro.
template <typename T> T dbg(std::string fmt, T arg) {
  if (kDebug)
    XLOG(fmt, arg);
  return arg;
}

template <typename F>
void verifyValidity(ReadIopVal& iop,
                    size_t po2,
                    const Zll::TapSet& tapSet,
                    F compute_poly,
                    std::vector<Val>& mix,
                    std::vector<Val>& globals,
                    std::vector<MerkleTreeVerifier>& merkleVerifiers) {
  size_t size = size_t(1) << po2;
  size_t domain = size * kInvRate;

  // Set the Fiat-Shamir parameter for mixing constraint polynomials
  Val polyMix = dbg("polyMix: %u", iop.rngExtVal());

  // Read check merkle root
  MerkleTreeVerifier checkMerkle("check", iop, domain, kCheckSize, kQueries);

  // Pick a random place to check the polynomial constaints at
  Val Z = iop.rngExtVal();
  if (kDebug)
    XLOG("Z: %u", Z);

  // Read the tap coefficents, hash them, and commit to them
  auto coeffU = iop.readExtVals(tapSet.tapCount + kCheckSize, /*flip=*/true);
  iop.commit(hash(coeffU, /*flip=*/true));

  // Now, convert to evaluated tap values
  Val backOne = kRouRev[po2];
  std::vector<Val> evalU;
  for (const auto& group : tapSet.groups) {
    for (const auto& reg : group.regs) {
      std::vector<Val> coeffs(coeffU.begin() + reg.tapPos,
                              coeffU.begin() + reg.tapPos + reg.backs.size());
      for (const auto& back : reg.backs) {
        auto x = raisepow(backOne, back) * Z;
        auto fx = poly_eval(coeffs, x);
        evalU.push_back(fx);
      }
    }
  }

  // Compute the core polynomial
  Val result = compute_poly(evalU, globals, mix, polyMix, Z);
  if (kDebug)
    XLOG("result: %u", result);

  // Generate the check polynomial
  Val check = 0;
  size_t remap[4] = {0, 2, 1, 3};
  for (size_t i = 0; i < 4; i++) {
    Val zi = raisepow(Z, i);
    size_t rmi = remap[i];
    check = check + coeffU[tapSet.tapCount + rmi + 0] * zi * Val({1, 0, 0, 0});
    check = check + coeffU[tapSet.tapCount + rmi + 4] * zi * Val({0, 1, 0, 0});
    check = check + coeffU[tapSet.tapCount + rmi + 8] * zi * Val({0, 0, 1, 0});
    check = check + coeffU[tapSet.tapCount + rmi + 12] * zi * Val({0, 0, 0, 1});
  }
  check = check * (raisepow(3 * Z, size) - 1);
  if (kDebug)
    XLOG("Check polynomial: %u", check);

  // Make sure they match
  eq(check, result);

  // Set the Fiat-Shamir parameter for mixing DEEP polynomials (U)
  Val deepMix = iop.rngExtVal();

  // Make the mixed U polynomials
  std::vector<std::vector<Val>> comboU;
  for (size_t i = 0; i < tapSet.combos.size(); i++) {
    comboU.emplace_back(tapSet.combos[i].backs.size(), 0);
  }
  Val curMix = 1;
  for (const auto& group : tapSet.groups) {
    for (const auto& reg : group.regs) {
      for (size_t i = 0; i < reg.backs.size(); i++) {
        comboU[reg.combo][i] = comboU[reg.combo][i] + curMix * coeffU[reg.tapPos + i];
      }
      curMix = curMix * deepMix;
    }
  }
  // Handle check group
  comboU.emplace_back(1, 0);
  for (size_t i = 0; i < kCheckSize; i++) {
    comboU.back()[0] = comboU.back()[0] + curMix * coeffU[tapSet.tapCount + i];
    curMix = curMix * deepMix;
  }
  // Finally, do a FRI verification
  friVerify(iop, size, [&](ReadIopVal& iop, Val idx) {
    auto x = dynamic_pow(kRouFwd[log2Ceil(domain)], idx, domain);
    std::map<unsigned, std::vector<Val>> rows;
    for (size_t i = 0; i < merkleVerifiers.size(); i++) {
      rows[i] = merkleVerifiers[i].verify(iop, idx);
    }
    auto checkRow = checkMerkle.verify(iop, idx);
    Val curMix = 1;
    std::vector<Val> tot(comboU.size(), 0);
    for (auto [idx, group] : llvm::enumerate(tapSet.groups)) {
      for (const auto& reg : group.regs) {
        tot[reg.combo] = tot[reg.combo] + curMix * rows.at(idx)[reg.offset];
        curMix = curMix * deepMix;
      }
    }
    for (size_t i = 0; i < kCheckSize; i++) {
      tot.back() = tot.back() + curMix * checkRow[i];
      curMix = curMix * deepMix;
    }
    Val ret = 0;
    for (const auto& combo : tapSet.combos) {
      unsigned id = combo.combo;
      for (size_t i = 0; i < comboU[id].size(); i++) {
      }
      Val num = tot[id] - poly_eval(comboU[id], x);
      Val divisor = 1;
      for (auto back : combo.backs) {
        divisor = divisor * (x - Z * raisepow(backOne, back));
      }
      ret = ret + num * inv(divisor);
    }
    Val checkNum = tot.back() - comboU.back()[0];
    Val checkDivisor = x - raisepow(Z, 4);
    ret = ret + checkNum * inv(checkDivisor);
    return ret;
  });
}

} // namespace

VerifyInfo verify(ReadIopVal& iop, size_t po2, const CircuitInterface& circuit) {
  VerifyInfo verifyInfo;

  // At the start of verification, add the version strings to the Fiat-Shamir transcript.
  std::vector<zirgen::Val> proof_system_info;
  std::vector<zirgen::Val> circuit_info;
  auto circuit_info_bytes = circuit.get_circuit_info();
  for (size_t i = 0; i < PROTOCOL_INFO_LEN; i++) {
    proof_system_info.push_back(PROOF_SYSTEM_INFO.at(i));
    circuit_info.push_back(circuit_info_bytes.at(i));
  }
  iop.commit(dbg("proof system: %h", hash(proof_system_info)));
  iop.commit(dbg("circuit: %h", hash(circuit_info)));

  size_t size = size_t(1) << po2;
  size_t domain = size * kInvRate;

  // Get the taps
  const auto& tapSet = circuit.get_taps();
  // Read the outputs and po2, which constitute the statement.
  // Mix the digest into Fiat-Shamir transcript hash.
  auto statement = iop.readBaseVals(circuit.out_size() + 1);
  auto statementDigest = hash(statement);
  iop.commit(statementDigest);

  // Split the statement buffer into the out (i.e. globals) elems and the po2.
  auto iop_po2 = statement[statement.size() - 1];
  statement.pop_back();
  auto out = statement;
  verifyInfo.out = out;
  verifyInfo.outDigest = hash(out);

  // Verify the po2: Need to convert out of Montgomery form since it serialized unencoded.
  eq(iop_po2 * 268435454, Val(po2));
  // Cache the sizes
  size_t codeSize = tapSet.groups.at(/*REGISTER_GROUP_CODE=*/1).regs.size();
  size_t dataSize = tapSet.groups.at(/*REGISTER_GROUP_DATA=*/2).regs.size();
  size_t accumSize = tapSet.groups.at(/*REGISTER_GROUP_ACCUM=*/0).regs.size();
  size_t tapCount = tapSet.tapCount;

  if (kDebug)
    XLOG("code size: %u data size: %u accum size: %u tap count: %u",
         codeSize,
         dataSize,
         accumSize,
         tapCount);

  // Read the code + data merkle roots
  MerkleTreeVerifier codeMerkle("code", iop, domain, codeSize, kQueries);
  MerkleTreeVerifier dataMerkle("data", iop, domain, dataSize, kQueries);

  verifyInfo.codeRoot = codeMerkle.getRoot();

  // Generate accum mixing data
  std::vector<Val> accumMix;
  for (size_t i = 0; i < circuit.mix_size(); i++) {
    accumMix.push_back(iop.rngBaseVal());
  }

  // Read accum merkle root
  MerkleTreeVerifier accumMerkle("accum", iop, domain, accumSize, kQueries);

  auto computePoly = [&](llvm::ArrayRef<Val> u,
                         llvm::ArrayRef<Val> out,
                         llvm::ArrayRef<Val> accumMix,
                         Val polyMix,
                         Val z) { return circuit.compute_poly(u, out, accumMix, polyMix, z); };
  std::vector<MerkleTreeVerifier> merkleVerifiers = {accumMerkle, codeMerkle, dataMerkle};
  verifyValidity(iop, po2, tapSet, computePoly, accumMix, out, merkleVerifiers);
  return verifyInfo;
}

VerifyInfo verifyRecursion(ReadIopVal& allowedRoot,
                           std::vector<ReadIopVal> seals,
                           std::vector<ReadIopVal> alloweds,
                           const CircuitInterface& circuit) {
  VerifyInfo verifyInfo;
  verifyInfo.codeRoot = allowedRoot.readDigests(1)[0];

  for (size_t i = 0; i != recursion::kNumRollup; ++i) {
    VerifyInfo subInfo = zirgen::verify::verify(seals[i], recursion::kRecursionPo2, circuit);
    Val codeRootIndex = alloweds[i].readBaseVals(1)[0];
    verifyMerkleGroupMember(subInfo.codeRoot,
                            codeRootIndex,
                            alloweds[i].readDigests(recursion::kAllowedCodeMerkleDepth),
                            verifyInfo.codeRoot);

    if (i == 0) {
      verifyInfo.outDigest = subInfo.outDigest;
    } else {
      verifyInfo.outDigest = fold(verifyInfo.outDigest, subInfo.outDigest);
    }
  }
  return verifyInfo;
}

} // namespace zirgen::verify

namespace zirgen::verify {

std::vector<Val> verifyV3(ReadIopVal& iop, size_t po2, const CircuitInterfaceV3& circuit) {
  const Zll::TapSet& tapSet = circuit.getTaps();
  size_t size = size_t(1) << po2;
  size_t domain = size * kInvRate;
  std::vector<Val> mix;
  std::vector<Val> globals;
  std::vector<MerkleTreeVerifier> merkleVerifiers;

  for (size_t i = 0; i < circuit.getGroupInfo().size(); i++) {
    const GroupInfoV3& group = circuit.getGroupInfo()[i];

    // Draw Fiat-Shamir randomness for the group.
    for (size_t j = 0; j < group.mixCount; j++) {
      mix.push_back(iop.rngBaseVal());
    }

    // Read and commit to the globals for the group.
    if (group.globalCount > 0) {
      std::vector<Val> groupGlobals = iop.readBaseVals(group.globalCount);
      globals.insert(globals.end(), groupGlobals.begin(), groupGlobals.end());
      iop.commit(hash(groupGlobals));
    }

    // Read the merkle proof for the group from the IOP.
    size_t columns = tapSet.groups.at(i).regs.size();
    merkleVerifiers.emplace_back("group", iop, domain, columns, kQueries);
  }

  auto computePoly = [&](llvm::ArrayRef<Val> u,
                         llvm::ArrayRef<Val> out,
                         llvm::ArrayRef<Val> accumMix,
                         Val polyMix,
                         Val z) { return circuit.computePolyExt(u, out, accumMix, polyMix, z); };
  verifyValidity(iop, po2, tapSet, computePoly, mix, globals, merkleVerifiers);
  return globals;
}

} // namespace zirgen::verify
