// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "read_iop.h"

namespace zirgen {

constexpr size_t kEncodedDigestSize = 16;

// An abstract class to wrap hash implementations
class IHashSuite {
public:
  virtual ~IHashSuite() = default;
  // Make an RNG for Fiat-Shamir
  virtual std::unique_ptr<IopRng> makeRng() const = 0;
  // Hash baby-bear field elements (sent in normal, non-montgomery, form)
  virtual Digest hash(const uint32_t* data, size_t size) const = 0;
  // Hash two hashes together
  virtual Digest hashPair(const Digest& x, const Digest& y) const = 0;
  // Encode a hash into baby-bear field elements
  virtual std::vector<uint32_t> encode(const Digest& x, size_t size = 16) const = 0;
  // Decode a hash from baby-bear field elements
  virtual Digest decode(const std::vector<uint32_t>& x) const = 0;
};

std::unique_ptr<IHashSuite> shaHashSuite();
std::unique_ptr<IHashSuite> poseidonHashSuite();
std::unique_ptr<IHashSuite> poseidon2HashSuite();
std::unique_ptr<IHashSuite> poseidon254HashSuite();
std::unique_ptr<IHashSuite> mixedPoseidon2ShaHashSuite();

} // namespace zirgen
