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

#include <array>

#include "zirgen/compiler/zkp/digest.h"
#include "zirgen/compiler/zkp/read_iop.h"

namespace zirgen {

// Hashing functions used by the proof system itself
Digest poseidon2Hash(const uint32_t* data, size_t size);
Digest poseidon2HashPair(Digest x, Digest y);

// Raw access to inner poseidon sponge function + friends
void poseidonMultiplyByMExt(std::array<uint32_t, 24>& cells);
void poseidonDoExtRound(std::array<uint32_t, 24>& cells, size_t idx);
void poseidonDoIntRounds(std::array<uint32_t, 24>& cells);
void poseidonSponge(std::array<uint32_t, 24>& cells);

class Poseidon2Rng : public IopRng {
public:
  Poseidon2Rng();
  // Mix the hash into the entropy pool
  void mix(const Digest& data) override;
  // Generate uniform bitsfrom the entropy pool
  uint32_t generateBits(size_t bits) override;
  // Generate a BabbyBear value from the entropy pool
  uint32_t generateFp() override;

  std::unique_ptr<IopRng> newOfThisType() override { return std::make_unique<Poseidon2Rng>(); }

private:
  std::array<uint32_t, 24> cells;
  size_t pool_used;
};

} // namespace zirgen
