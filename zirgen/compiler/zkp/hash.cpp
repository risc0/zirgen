// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/compiler/zkp/hash.h"
#include "zirgen/compiler/zkp/baby_bear.h"
#include "zirgen/compiler/zkp/poseidon.h"
#include "zirgen/compiler/zkp/poseidon2.h"
#include "zirgen/compiler/zkp/poseidon_254.h"
#include "zirgen/compiler/zkp/sha256.h"
#include "zirgen/compiler/zkp/sha_rng.h"

namespace zirgen {

namespace {

class ShaHashSuite : public IHashSuite {
public:
  std::unique_ptr<IopRng> makeRng() const override { return std::make_unique<ShaRng>(); }

  Digest hash(const uint32_t* data, size_t size) const override {
    std::vector<uint32_t> toMont(size);
    for (size_t i = 0; i < size; i++) {
      toMont[i] = (uint64_t(data[i]) * kBabyBearToMontgomery) % kBabyBearP;
    }
    return shaHash(toMont.data(), size);
  }

  Digest hashPair(const Digest& x, const Digest& y) const override { return shaHashPair(x, y); }

  std::vector<uint32_t> encode(const Digest& x, size_t count) const override {
    std::vector<uint32_t> ret;
    size_t bits = 256 / count;
    for (size_t i = 0; i < 8; i++) {
      uint32_t cur = x.words[i];
      for (size_t j = 0; j < 32 / bits; j++) {
        ret.push_back(cur & ((1 << bits) - 1));
        cur >>= bits;
      }
    }
    return ret;
  }

  Digest decode(const std::vector<uint32_t>& x) const override {
    Digest ret;
    size_t bits = 256 / x.size();
    size_t offset = 0;
    for (size_t i = 0; i < 8; i++) {
      uint32_t cur = 0;
      for (size_t j = 0; j < 32 / bits; j++) {
        assert(x[offset] < (1U << bits));
        cur |= x[offset] << (j * bits);
        offset++;
      }
      ret.words[i] = cur;
    }
    return ret;
  }
};

class PoseidonHashSuite : public IHashSuite {
public:
  std::unique_ptr<IopRng> makeRng() const override { return std::make_unique<PoseidonRng>(); }

  Digest hash(const uint32_t* data, size_t size) const override { return poseidonHash(data, size); }

  Digest hashPair(const Digest& x, const Digest& y) const override {
    return poseidonHashPair(x, y);
  }

  std::vector<uint32_t> encode(const Digest& x, size_t count) const override {
    std::vector<uint32_t> ret;
    size_t pad = count / 8 - 1;
    for (size_t i = 0; i < 8; i++) {
      ret.push_back(x.words[i]);
      for (size_t j = 0; j < pad; j++) {
        ret.push_back(0);
      }
    }
    return ret;
  }

  Digest decode(const std::vector<uint32_t>& x) const override {
    Digest ret;
    size_t pad = x.size() / 8 - 1;
    size_t offset = 0;
    for (size_t i = 0; i < 8; i++) {
      ret.words[i] = x[offset++];
      offset += pad;
    }
    return ret;
  }
};

class Poseidon2HashSuite : public IHashSuite {
public:
  std::unique_ptr<IopRng> makeRng() const override { return std::make_unique<Poseidon2Rng>(); }

  Digest hash(const uint32_t* data, size_t size) const override {
    return poseidon2Hash(data, size);
  }

  Digest hashPair(const Digest& x, const Digest& y) const override {
    return poseidon2HashPair(x, y);
  }

  std::vector<uint32_t> encode(const Digest& x, size_t count) const override {
    std::vector<uint32_t> ret;
    size_t pad = count / 8 - 1;
    for (size_t i = 0; i < 8; i++) {
      ret.push_back(x.words[i]);
      for (size_t j = 0; j < pad; j++) {
        ret.push_back(0);
      }
    }
    return ret;
  }

  Digest decode(const std::vector<uint32_t>& x) const override {
    Digest ret;
    size_t pad = x.size() / 8 - 1;
    size_t offset = 0;
    for (size_t i = 0; i < 8; i++) {
      ret.words[i] = x[offset++];
      offset += pad;
    }
    return ret;
  }
};

class Poseidon254HashSuite : public IHashSuite {
public:
  std::unique_ptr<IopRng> makeRng() const override { return std::make_unique<Poseidon254Rng>(); }

  Digest hash(const uint32_t* data, size_t size) const override {
    return poseidon254Hash(data, size);
  }

  Digest hashPair(const Digest& x, const Digest& y) const override {
    return poseidon254HashPair(x, y);
  }

  std::vector<uint32_t> encode(const Digest& x, size_t count) const override {
    throw std::runtime_error("Unimplemeneted");
  }

  Digest decode(const std::vector<uint32_t>& x) const override {
    throw std::runtime_error("Unimplemeneted");
  }
};

class MixedPoseidon2ShaHashSuite : public IHashSuite {
public:
  std::unique_ptr<IopRng> makeRng() const override { return std::make_unique<ShaRng>(); }

  Digest hash(const uint32_t* data, size_t size) const override {
    return poseidon2Hash(data, size);
  }

  Digest hashPair(const Digest& x, const Digest& y) const override {
    return poseidon2HashPair(x, y);
  }

  std::vector<uint32_t> encode(const Digest& x, size_t size) const override {
    Poseidon2HashSuite suite;
    return suite.encode(x, size);
  }

  Digest decode(const std::vector<uint32_t>& x) const override {
    Poseidon2HashSuite suite;
    return suite.decode(x);
  }
};

} // namespace

std::unique_ptr<IHashSuite> shaHashSuite() {
  return std::make_unique<ShaHashSuite>();
}

std::unique_ptr<IHashSuite> poseidonHashSuite() {
  return std::make_unique<PoseidonHashSuite>();
}

std::unique_ptr<IHashSuite> poseidon2HashSuite() {
  return std::make_unique<Poseidon2HashSuite>();
}

std::unique_ptr<IHashSuite> poseidon254HashSuite() {
  return std::make_unique<Poseidon254HashSuite>();
}

std::unique_ptr<IHashSuite> mixedPoseidon2ShaHashSuite() {
  return std::make_unique<MixedPoseidon2ShaHashSuite>();
}

} // namespace zirgen
