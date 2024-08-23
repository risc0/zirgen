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

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringRef.h"

namespace zirgen {

class P254 {
public:
  inline P254() : inner(512, 0, true) {}
  inline P254(int64_t val) : inner(512, val, true) {}
  inline P254(llvm::StringRef str) : inner(512, str, 10) {}
  P254(Digest val);
  inline P254 operator+(const P254& rhs) const { return P254((inner + rhs.inner).urem(kP)); }
  inline P254 operator-() const { return P254((kP - inner).urem(kP)); }
  inline P254 operator-(const P254& rhs) const { return P254((kP + inner - rhs.inner).urem(kP)); }
  inline P254 operator*(const P254& rhs) const { return P254((inner * rhs.inner).urem(kP)); }
  inline bool operator==(const P254& rhs) const { return inner == rhs.inner; }
  Digest toDigest();
  std::string toString(size_t base = 10);

private:
  inline P254(llvm::APInt inner) : inner(inner) {}
  llvm::APInt inner;
  static llvm::APInt kP;
};

Digest poseidon254Hash(const uint32_t* data, size_t size);
Digest poseidon254HashPair(Digest x, Digest y);

class Poseidon254Rng : public IopRng {
public:
  Poseidon254Rng();
  // Mix the hash into the entropy pool
  void mix(const Digest& data) override;
  // Generate uniform bitsfrom the entropy pool
  uint32_t generateBits(size_t bits) override;
  // Generate a BabbyBear value from the entropy pool
  uint32_t generateFp() override;
  std::unique_ptr<IopRng> newOfThisType() override { return std::make_unique<Poseidon254Rng>(); }

private:
  std::array<P254, 3> cells;
};

} // namespace zirgen
