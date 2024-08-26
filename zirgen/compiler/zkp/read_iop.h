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
#include "zirgen/compiler/zkp/util.h"

#include <cassert>
#include <memory>
#include <vector>

namespace zirgen {

// An abstract type to wrap IOP Rng generation
class IopRng {
public:
  virtual ~IopRng() = default;
  // Mix a commitment into the Rng state
  virtual void mix(const Digest& message) = 0;
  // Generate a uniform n-bit numbers from the entropy pool
  virtual uint32_t generateBits(size_t bits) = 0;
  // Generate an Fp value from the entropy pool
  virtual uint32_t generateFp() = 0;
  // Creates a new RNG of the same type as this one
  virtual std::unique_ptr<IopRng> newOfThisType() = 0;
};

class ReadIop {
public:
  // Intialize a IOP reader with a proof
  ReadIop(std::unique_ptr<IopRng> rng, const uint32_t* proof, size_t size)
      : rng(std::move(rng)), proof(proof), size(size) {}
  // Reads 'unverified' data from the proof (typically verified later via a commitment)
  void read(uint32_t* out, size_t readSize) {
    assert(size >= readSize);
    std::copy(proof, proof + readSize, out);
    proof += readSize;
    size -= readSize;
  }
  void read(Digest* data, size_t count) {
    read(reinterpret_cast<uint32_t*>(data), count * sizeof(Digest) / sizeof(uint32_t));
  }
  // Apply a commitment to the RNG state
  void commit(const Digest& message) { rng->mix(message); }
  // Get the psudeorandom challenge at this point in the protocol
  uint32_t generateBits(size_t bits) { return rng->generateBits(bits); }
  // Get the psudeorandom Fp
  uint32_t generateFp() { return rng->generateFp(); }
  // Verify the proof was complete consumed
  void verifyComplete() { assert(size == 0); }

  // Returns a new empty IOP with the same rng type.
  std::unique_ptr<ReadIop> newOfThisType() {
    return std::make_unique<ReadIop>(rng->newOfThisType(), nullptr, 0);
  }

private:
  std::unique_ptr<IopRng> rng;
  const uint32_t* proof;
  size_t size;
};

} // namespace zirgen
