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
