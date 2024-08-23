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

#include "zirgen/compiler/zkp/digest.h"
#include "zirgen/compiler/zkp/read_iop.h"
#include "zirgen/compiler/zkp/sha256.h"

namespace zirgen {

// The ShaRng basically provides a cryptographically strong PRNG and is used in the IOP protocol.
// Thus we need an initially empty entropy pool and a way to mix new data in, as well as a way to
// draw as many random bits as we require.  We basically use a construction similar to a sponge,
// with a capacity of 256 bits, a rate of 256 bits, and Sha256 as the mixing function.

class ShaRng : public IopRng {
public:
  // Makes an RNG with an initial 'default' entropy pool (thus it's not very random until you use
  // mix to mix somthing in!).
  ShaRng();
  // Mix the hash into the entropy pool
  void mix(const Digest& data) override;
  // Generate random bits from the entropy pool
  uint32_t generateBits(size_t bits) override;
  // Generate an Fp value from the entropy pool
  uint32_t generateFp() override;

  std::unique_ptr<IopRng> newOfThisType() override { return std::make_unique<ShaRng>(); }

private:
  // Internal function to do base generation
  uint32_t generate();

  void step();
  Digest pool0;
  Digest pool1;
  size_t poolUsed;
};

} // namespace zirgen
