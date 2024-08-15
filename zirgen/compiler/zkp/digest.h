// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>

namespace zirgen {

// A digest (still in uint32_t parts for easy rolling up in merkle trees).
struct Digest {
  uint32_t words[8];
  // The 'zero' digest, sort of the nullptr of digests.
  static Digest zero() { return {{0, 0, 0, 0, 0, 0, 0, 0}}; }

  int cmp(Digest rhs) const {
    for (size_t i = 0; i < 8; i++) {
      if (words[i] != rhs.words[i]) {
        return words[i] < rhs.words[i] ? -1 : 1;
      }
    }
    return 0;
  }
  bool operator==(Digest rhs) const { return cmp(rhs) == 0; }
  bool operator!=(Digest rhs) const { return cmp(rhs) != 0; }
};

inline std::string hexDigest(const Digest& digest) {
  const char* hexdigits = "0123456789abcdef";
  std::string r(64, 0);
  for (size_t i = 0; i < 8; i++) {
    uint32_t word = digest.words[i];
    for (size_t j = 0; j < 4; j++) {
      uint8_t byte = word >> 24;
      r[i * 8 + (3 - j) * 2] = hexdigits[byte >> 4];
      r[i * 8 + (3 - j) * 2 + 1] = hexdigits[byte & 0xf];
      word <<= 8;
    }
  }
  return r;
}

// StreamType could be a std stream or a LLVM stream.
template <typename StreamType> inline StreamType& operator<<(StreamType& os, const Digest& x) {
  os << hexDigest(x);
  return os;
}

} // namespace zirgen
