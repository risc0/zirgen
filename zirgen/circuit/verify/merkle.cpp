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

#include "zirgen/circuit/verify/merkle.h"

namespace zirgen::verify {

namespace {

// If true, emit debug logging.  NOTE: This will change the code root of predicates, so
// should not be left on when generating release ZKRs.
constexpr bool kDebug = false;

} // namespace

MerkleTreeParams::MerkleTreeParams(size_t rowSize,
                                   size_t colSize,
                                   size_t queries,
                                   bool useExtension)
    : rowSize(rowSize)
    , colSize(colSize)
    , queries(queries)
    , layers(log2Ceil(rowSize))
    , useExtension(useExtension) {
  assert(1U << layers == rowSize);
  topLayer = 0;
  for (size_t i = 1; i < layers; i++) {
    if ((1U << i) > queries) {
      break;
    }
    topLayer = i;
  }
  topSize = 1 << topLayer;
}

MerkleTreeVerifier::MerkleTreeVerifier(std::string bufName,
                                       ReadIopVal& iop,
                                       size_t rowSize,
                                       size_t colSize,
                                       size_t queries,
                                       bool useExtension)
    : MerkleTreeParams(rowSize, colSize, queries, useExtension), top(topSize), bufName(bufName) {
  auto topRec = iop.readDigests(topSize);
  top.insert(top.end(), topRec.begin(), topRec.end());
  for (size_t i = topSize; i-- > 1;) {
    top[i] = fold(top[i * 2], top[i * 2 + 1]);
  }
  iop.commit(top[1]);
}

DigestVal MerkleTreeVerifier::getRoot() const {
  return top[1];
}

std::vector<Val> MerkleTreeVerifier::verify(ReadIopVal& iop, Val idx) const {
  std::vector<Val> out;
  if (useExtension) {
    out = iop.readExtVals(colSize);
  } else {
    out = iop.readBaseVals(colSize);
  }
  DigestVal cur = hash(out);
  idx = idx + rowSize;
  for (size_t i = 0; i < layers - topLayer; i++) {
    Val lowBit = idx & 1;
    DigestVal other = iop.readDigests(1)[0];
    idx = idx - lowBit;
    idx = idx / 2;
    auto lhs = select(lowBit, {cur, other});
    auto rhs = select(lowBit, {other, cur});
    cur = fold(lhs, rhs);
  }
  auto topDigest = select(idx - topSize, llvm::ArrayRef(top).slice(topSize, topSize));
  if (kDebug)
    XLOG("Merkle " + bufName + " expected: %h, calculated %h", topDigest, cur);
  assert_eq(cur, topDigest);
  return out;
}

DigestVal calculateMerkleProofRoot(DigestVal member, Val idx, std::vector<DigestVal> proof) {
  DigestVal cur = member;
  for (DigestVal other : proof) {
    Val lowBit = idx & 1;
    idx = idx - lowBit;
    idx = idx / 2;
    auto lhs = select(lowBit, {cur, other});
    auto rhs = select(lowBit, {other, cur});
    cur = fold(lhs, rhs);
  }
  eqz(idx);
  return cur;
}

void verifyMerkleGroupMember(DigestVal member,
                             Val idx,
                             std::vector<DigestVal> proof,
                             DigestVal expectedRoot) {
  auto root = calculateMerkleProofRoot(member, idx, proof);
  assert_eq(root, expectedRoot);
}

} // namespace zirgen::verify
