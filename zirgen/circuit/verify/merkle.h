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

#pragma once

#include "zirgen/compiler/edsl/edsl.h"

namespace zirgen::verify {

class MerkleTreeParams {
public:
  MerkleTreeParams(size_t rowSize, size_t colSize, size_t queries, bool useExtension);

protected:
  // The size of a row (i.e. the number of columns)
  size_t rowSize;
  // The size of a column (i.e. the number of rows)
  size_t colSize;
  // The expected number of queries
  size_t queries;
  // Number of layers.  A 'zero' level/layer tree has a single root that is also
  // a leaf, whereas a 4 level tree has 16 roots, and 15 non-leaf nodes,
  // including the root.
  size_t layers;
  // The 'top' layer selected
  size_t topLayer;
  // The size of the top layer
  size_t topSize;
  // Enable the field extension
  bool useExtension;
};

class MerkleTreeVerifier : public MerkleTreeParams {
public:
  // Construct a merkle tree verifier, reading the top from the IOP, commit to root.
  MerkleTreeVerifier(std::string bufName,
                     ReadIopVal& iop,
                     size_t rowSize,
                     size_t colSize,
                     size_t queries,
                     bool useExtension = false);

  // Get the root digest of the tree
  DigestVal getRoot() const;

  // Verify a proof, return the column values, throw on error
  std::vector<Val> verify(ReadIopVal& iop, Val idx) const;

private:
  std::vector<DigestVal> top;
  std::string bufName;
};

// Calculates the root of a given Merkle inclusion proof.
// Note that this does nothing to verify that a given proof is valid.
DigestVal calculateMerkleProofRoot(DigestVal member, Val idx, std::vector<DigestVal> proof);

// Verifies that "member" is a member of a merkle tree rooted at
// "group"
void verifyMerkleGroupMember(DigestVal member,
                             Val idx,
                             std::vector<DigestVal> proof,
                             DigestVal expectedRoot);

} // namespace zirgen::verify
