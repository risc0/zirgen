// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include <array>

#include "zirgen/compiler/zkp/digest.h"
#include "zirgen/compiler/zkp/read_iop.h"

namespace zirgen {

Digest poseidon2Hash(const uint32_t* data, size_t size);
Digest poseidon2HashPair(Digest x, Digest y);

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
