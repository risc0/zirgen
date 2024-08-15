// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include <array>

#include "zirgen/compiler/zkp/digest.h"
#include "zirgen/compiler/zkp/read_iop.h"

namespace zirgen {

Digest poseidonHash(const uint32_t* data, size_t size);
Digest poseidonHashPair(Digest x, Digest y);

class PoseidonRng : public IopRng {
public:
  PoseidonRng();
  // Mix the hash into the entropy pool
  void mix(const Digest& data) override;
  // Generate uniform bitsfrom the entropy pool
  uint32_t generateBits(size_t bits) override;
  // Generate a BabbyBear value from the entropy pool
  uint32_t generateFp() override;
  std::unique_ptr<IopRng> newOfThisType() override { return std::make_unique<PoseidonRng>(); }

private:
  std::array<uint32_t, 24> cells;
  size_t pool_used;
};

} // namespace zirgen
