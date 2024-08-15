// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include <array>
#include <iostream>

#include "zirgen/compiler/zkp/baby_bear.h"
#include "zirgen/compiler/zkp/poseidon.h"

namespace {

#include "zirgen/compiler/zkp/poseidon_consts.h"

} // namespace

namespace zirgen {

using cells_t = std::array<uint32_t, CELLS>;

cells_t add_const(const cells_t& in, size_t idx) {
  cells_t out;
  for (size_t i = 0; i < CELLS; i++) {
    out[i] = (in[i] + ROUND_CONSTANTS[idx * CELLS + i]) % kBabyBearP;
  }
  return out;
}

cells_t mul_mds(const cells_t& in) {
  cells_t out;
  for (size_t i = 0; i < CELLS; i++) {
    uint64_t tot = 0;
    for (size_t j = 0; j < CELLS; j++) {
      tot += uint64_t(MDS[i * CELLS + j]) * uint64_t(in[j]);
      tot = tot % kBabyBearP;
    }
    out[i] = tot;
  }
  return out;
}

uint32_t sbox(uint64_t in) {
  uint64_t in2 = in * in % kBabyBearP;
  uint64_t in4 = in2 * in2 % kBabyBearP;
  uint64_t in6 = in4 * in2 % kBabyBearP;
  uint64_t in7 = in6 * in % kBabyBearP;
  return in7;
}

cells_t full_round(const cells_t& in, size_t idx) {
  cells_t out = add_const(in, idx);
  for (size_t i = 0; i < CELLS; i++) {
    out[i] = sbox(out[i]);
  }
  return mul_mds(out);
}

cells_t partial_round(const cells_t& in, size_t idx) {
  cells_t out = add_const(in, idx);
  out[0] = sbox(out[0]);
  return mul_mds(out);
}

cells_t poseidon_mix(const cells_t& in) {
  cells_t cur = in;
  size_t idx = 0;
  for (size_t i = 0; i < ROUNDS_HALF_FULL; i++) {
    cur = full_round(cur, idx++);
  }
  for (size_t i = 0; i < ROUNDS_PARTIAL; i++) {
    cur = partial_round(cur, idx++);
  }
  for (size_t i = 0; i < ROUNDS_HALF_FULL; i++) {
    cur = full_round(cur, idx++);
  }
  return cur;
}

Digest poseidonHash(const uint32_t* data, size_t size) {
  cells_t cur = {0};
  size_t curUsed = 0;
  for (size_t i = 0; i < size; i++) {
    cur[curUsed] = (cur[curUsed] + data[i]) % kBabyBearP;
    curUsed++;
    if (curUsed == 16) {
      cur = poseidon_mix(cur);
      curUsed = 0;
    }
  }
  if (curUsed != 0) {
    cur = poseidon_mix(cur);
  }
  Digest out;
  for (size_t i = 0; i < 8; i++) {
    out.words[i] = toMontgomery(cur[i]);
  }
  return out;
}

Digest poseidonHashPair(Digest x, Digest y) {
  cells_t cur = {0};
  for (size_t i = 0; i < 8; i++) {
    cur[i] = fromMontgomery(x.words[i]);
    cur[8 + i] = fromMontgomery(y.words[i]);
  }
  cur = poseidon_mix(cur);
  Digest out;
  for (size_t i = 0; i < 8; i++) {
    out.words[i] = toMontgomery(cur[i]);
  }
  return out;
}

PoseidonRng::PoseidonRng() : pool_used(0) {
  for (size_t i = 0; i < CELLS; i++) {
    cells[i] = 0;
  }
}

void PoseidonRng::mix(const Digest& data) {
  // if switching from squeezing, do a poseidon_mix
  if (pool_used != 0) {
    cells = zirgen::poseidon_mix(cells);
    pool_used = 0;
  }
  // Add in 8 elements (also # of digest words)
  for (size_t i = 0; i < 8; i++) {
    cells[i] += fromMontgomery(data.words[i]);
    cells[i] %= kBabyBearP;
  }
  // Mix
  cells = zirgen::poseidon_mix(cells);
}

uint32_t PoseidonRng::generateBits(size_t bits) {
  assert(bits <= 27);
  uint32_t and_mask = (1 << bits) - 1;
  // Make 4 attempts to get a non-zero value (which means the low 27 bits will be
  // uniformly distributed).  This fails less than 1 in 2^100 times, and so is
  // cryptographically indisinguishable from uniform.
  uint64_t val = generateFp();
  for (size_t i = 0; i < 3; i++) {
    uint64_t newVal = generateFp();
    if (val == 0) {
      val = newVal;
    }
  }
  return and_mask & val;
}

uint32_t PoseidonRng::generateFp() {
  if (pool_used == 16) {
    cells = poseidon_mix(cells);
    pool_used = 0;
  }
  return cells[pool_used++];
}

} // namespace zirgen
