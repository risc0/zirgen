// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include <arpa/inet.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "zirgen/compiler/zkp/digest.h"

// A internal implementation of SHA-256 (which is the only one we use).  It's very basic, but I
// would like to avoid dependencies on third party libraries, and in some places we need direct
// access to the compression function, for example, when doing unpadded merkle tree rollups.
// Additionally it's useful for the GPU acceleration.
//
// The core compression function is a modification (for style, GPU, and c++isms), based on the code
// at: https://github.com/B-Con/crypto-algorithms/blob/master/sha256.c which is a public domain
// implementation by Brad Conte (brad@bradconte.com)

namespace zirgen {

// Namespace to hide away some of the details from the user.
namespace impl {

// Generate the initialization for hash state.
inline Digest initState() {
  return {{0x6a09e667,
           0xbb67ae85,
           0x3c6ef372,
           0xa54ff53a,
           0x510e527f,
           0x9b05688c,
           0x1f83d9ab,
           0x5be0cd19}};
}

inline void endianEncode(uint32_t* out, const uint8_t* in) {
  for (size_t i = 0; i < 16; i++) {
    out[i] = (in[4 * i + 0] << 24) | (in[4 * i + 1] << 16) | (in[4 * i + 2] << 8) | in[4 * i + 3];
  }
}

// Internal compression function, presumes chunk of 16 elements)
inline void compress(Digest& state, const uint32_t* chunk) { // NOLINT
  uint32_t roundK[64] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
                         0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
                         0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
                         0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                         0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
                         0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
                         0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
                         0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                         0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
                         0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
                         0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define ROTLEFT(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32 - (b))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x, 2) ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22))
#define EP1(x) (ROTRIGHT(x, 6) ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25))
#define SIG0(x) (ROTRIGHT(x, 7) ^ ROTRIGHT(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ ((x) >> 10))

#define ROUND_FUNC                                                                                 \
  uint32_t t1 = h + EP1(e) + CH(e, f, g) + roundK[i] + w[i];                                       \
  uint32_t t2 = EP0(a) + MAJ(a, b, c);                                                             \
  h = g;                                                                                           \
  g = f;                                                                                           \
  f = e;                                                                                           \
  e = d + t1;                                                                                      \
  d = c;                                                                                           \
  c = b;                                                                                           \
  b = a;                                                                                           \
  a = t1 + t2;
  uint32_t w[64];

  uint32_t a = state.words[0];
  uint32_t b = state.words[1];
  uint32_t c = state.words[2];
  uint32_t d = state.words[3];
  uint32_t e = state.words[4];
  uint32_t f = state.words[5];
  uint32_t g = state.words[6];
  uint32_t h = state.words[7];

  for (size_t i = 0; i < 16; i++) {
    w[i] = chunk[i];
    ROUND_FUNC;
  }
  for (size_t i = 16; i < 64; i++) {
    w[i] = SIG1(w[i - 2]) + w[i - 7] + SIG0(w[i - 15]) + w[i - 16];
    ROUND_FUNC;
  }

  state.words[0] += a;
  state.words[1] += b;
  state.words[2] += c;
  state.words[3] += d;
  state.words[4] += e;
  state.words[5] += f;
  state.words[6] += g;
  state.words[7] += h;

#undef ROTLEFT
#undef ROTRIGHT
#undef CH
#undef MAJ
#undef EP0
#undef EP1
#undef SIG0
#undef SIG1
#undef ROUND_FUNC
}

// Main entry point for uint32_t sized objects
inline Digest shaHashInner(const uint32_t* data, size_t size) {
  // Prepare the inital state
  uint32_t words[16];
  uint32_t curWord = 0;
  Digest state = initState();

  // Push all of the values
  for (size_t i = 0; i < size; i++) {
    words[curWord++] = htonl(data[i]);
    if (curWord == 16) {
      compress(state, words);
      curWord = 0;
    }
  }
  // Clear rest of the block
  for (size_t i = curWord; i < 16; i++) {
    words[i] = 0;
  }

  // Do final compression
  if (curWord != 0) {
    compress(state, words);
  }

  for (size_t i = 0; i < 8; i++) {
    state.words[i] = htonl(state.words[i]);
  }

  return state;
}

} // namespace impl

// Main entry points
inline Digest shaHash(const uint32_t* data, size_t size) {
  return impl::shaHashInner(data, size);
}

inline Digest shaHashPair(Digest x, Digest y) {
  // Copy both hash states into a single buffer
  uint32_t words[16];
  for (size_t i = 0; i < 8; i++) {
    words[i] = htonl(x.words[i]);
  }
  for (size_t i = 0; i < 8; i++) {
    words[8 + i] = htonl(y.words[i]);
  }

  // Initialize state + compress
  Digest state = impl::initState();
  impl::compress(state, words);

  for (size_t i = 0; i < 8; i++) {
    state.words[i] = htonl(state.words[i]);
  }

  // Return the results
  return state;
}

// Character based entry point
inline Digest shaHash(const uint8_t* data, size_t size) {
  // Capture initial # of bits
  uint64_t bitSize = size * uint64_t(8);

  // Initialize state
  Digest state = impl::initState();

  // Make room for a single chunk
  uint32_t words[16];

  // Process all the full chunks
  while (size >= 64) {
    impl::endianEncode(words, data);
    impl::compress(state, words);
    data += 64;
    size -= 64;
  }

  // Copy over the final bytes + pad (always safe, since size < 64)
  uint8_t final[64] = {0};
  memcpy(final, data, size);
  final[size] = 0x80;
  impl::endianEncode(words, final);

  // If we can't fit the size in the remaining room, compress + clear
  if (size + 1 > 56) {
    impl::compress(state, words);
    memset(words, 0, 64);
  }

  // Now add size + do final compession
  words[14] = bitSize >> 32;
  words[15] = bitSize & 0xffffffff;
  impl::compress(state, words);

  for (size_t i = 0; i < 8; i++) {
    state.words[i] = htonl(state.words[i]);
  }

  return state;
}

// Helper function for strings
inline Digest shaHash(const std::string& str) {
  return shaHash(reinterpret_cast<const uint8_t*>(str.data()), str.size());
}

} // namespace zirgen
