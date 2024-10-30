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

#include <array>
#include <iostream>

#include "zirgen/compiler/zkp/baby_bear.h"
#include "zirgen/compiler/zkp/poseidon2.h"

namespace {

#include "zirgen/compiler/zkp/poseidon2_consts.h"

} // namespace

namespace zirgen {

using cells_t = std::array<uint32_t, CELLS>;

cells_t add_round_constants_partial(const cells_t& in, size_t round) {
  cells_t out = in;
  out[0] += ROUND_CONSTANTS[round * CELLS];
  out[0] %= kBabyBearP;
  return out;
}

cells_t add_round_constants_full(const cells_t& in, size_t round) {
  cells_t out = in;
  for (size_t i = 0; i < CELLS; i++) {
    out[i] += ROUND_CONSTANTS[round * CELLS + i];
    out[i] %= kBabyBearP;
  }
  return out;
}

std::array<uint32_t, 4> multiply_by_4x4_circulant(const std::array<uint32_t, 4>& in) {
  const uint64_t circ_factor_2 = 2;
  const uint64_t circ_factor_4 = 4;
  uint64_t t0 = (in[0] + in[1]) % kBabyBearP;
  uint64_t t1 = (in[2] + in[3]) % kBabyBearP;
  uint64_t t2 = (circ_factor_2 * in[1] + t1) % kBabyBearP;
  uint64_t t3 = (circ_factor_2 * in[3] + t0) % kBabyBearP;
  uint32_t t4 = (circ_factor_4 * t1 + t3) % kBabyBearP;
  uint32_t t5 = (circ_factor_4 * t0 + t2) % kBabyBearP;
  uint32_t t6 = (t3 + t5) % kBabyBearP;
  uint32_t t7 = (t2 + t4) % kBabyBearP;
  return {t6, t5, t7, t4};
}

cells_t multiply_by_m_int(const cells_t& in) {
  // Exploit the fact that off-diagonal entries of M_INT are all 1.
  uint64_t sum = 0;
  cells_t out{};
  for (size_t i = 0; i < CELLS; i++) {
    sum += in[i];
  }
  sum %= kBabyBearP;
  for (size_t i = 0; i < CELLS; i++) {
    out[i] = (sum + M_INT_DIAG_HZN[i] * uint64_t(in[i])) % kBabyBearP;
  }
  return out;
}

cells_t multiply_by_m_ext(const cells_t& in) {
  // Optimized method for multiplication by M_EXT.
  // See appendix B of Poseidon2 paper for additional details.
  cells_t out{};
  std::array<uint32_t, 4> tmp_sums{};
  for (size_t i = 0; i < CELLS / 4; i++) {
    std::array<uint32_t, 4> chunk{in[4 * i], in[4 * i + 1], in[4 * i + 2], in[4 * i + 3]};
    chunk = multiply_by_4x4_circulant(chunk);
    for (size_t j = 0; j < 4; j++) {
      uint64_t to_add = chunk[j];
      to_add %= kBabyBearP;
      tmp_sums[j] += to_add;
      tmp_sums[j] %= kBabyBearP;
      out[4 * i + j] += to_add;
      out[4 * i + j] %= kBabyBearP;
    }
  }
  for (size_t i = 0; i < CELLS; i++) {
    out[i] += tmp_sums[i % 4];
    out[i] %= kBabyBearP;
  }
  return out;
}

cells_t multiply_by_m_ext_naive(const cells_t& in) {
  cells_t out{};
  for (size_t i = 0; i < CELLS; i++) {
    uint64_t tot = 0;
    for (size_t j = 0; j < CELLS; j++) {
      tot += (_M_EXT[i * CELLS + j] * in[j]) % kBabyBearP;
      tot %= kBabyBearP;
    }
    out[i] = tot;
  }
  return out;
}

uint32_t sbox2(uint64_t in) {
  uint64_t in2 = in * in % kBabyBearP;
  uint64_t in4 = in2 * in2 % kBabyBearP;
  uint64_t in6 = in4 * in2 % kBabyBearP;
  uint64_t in7 = in6 * in % kBabyBearP;
  return in7;
}

cells_t full_poseidon2_round(const cells_t& in, size_t idx) {
  cells_t out = add_round_constants_full(in, idx);
  for (size_t i = 0; i < CELLS; i++) {
    out[i] = sbox2(out[i]);
  }
  return multiply_by_m_ext(out);
}

cells_t partial_poseidon2_round(const cells_t& in, size_t idx) {
  cells_t out = add_round_constants_partial(in, idx);
  out[0] = sbox2(out[0]);
  return multiply_by_m_int(out);
}

cells_t poseidon2_mix(const cells_t& in) {
  cells_t cur = in;
  size_t idx = 0; // aka `round`

  // First linear layer.
  cur = multiply_by_m_ext(cur);

  for (size_t i = 0; i < ROUNDS_HALF_FULL; i++) {
    cur = full_poseidon2_round(cur, idx++);
  }
  for (size_t i = 0; i < ROUNDS_PARTIAL; i++) {
    cur = partial_poseidon2_round(cur, idx++);
  }
  for (size_t i = 0; i < ROUNDS_HALF_FULL; i++) {
    cur = full_poseidon2_round(cur, idx++);
  }

  return cur;
}

Digest poseidon2Hash(const uint32_t* data, size_t size) {
  cells_t cur = {0};
  size_t curUsed = 0;
  for (size_t i = 0; i < size; i++) {
    cur[curUsed] = data[i] % kBabyBearP;
    curUsed++;
    if (curUsed == 16) {
      cur = poseidon2_mix(cur);
      curUsed = 0;
    }
  }
  if (curUsed != 0 || size == 0) {
    // If `size` is not an even multiple of 16, zero-pad
    for (size_t loc = curUsed; loc < 16; loc++) {
      cur[loc] = 0;
    }
    cur = poseidon2_mix(cur);
  }
  Digest out;
  for (size_t i = 0; i < 8; i++) {
    out.words[i] = toMontgomery(cur[i]);
  }
  return out;
}

Digest poseidon2HashPair(Digest x, Digest y) {
  cells_t cur = {0};
  for (size_t i = 0; i < 8; i++) {
    cur[i] = fromMontgomery(x.words[i]);
    cur[8 + i] = fromMontgomery(y.words[i]);
  }
  cur = poseidon2_mix(cur);
  Digest out;
  for (size_t i = 0; i < 8; i++) {
    out.words[i] = toMontgomery(cur[i]);
  }
  return out;
}

void poseidonMultiplyByMExt(std::array<uint32_t, 24>& cells) {
  cells = multiply_by_m_ext(cells);
}

void poseidonDoExtRound(std::array<uint32_t, 24>& cells, size_t idx) {
  if (idx >= ROUNDS_HALF_FULL) {
    idx += ROUNDS_PARTIAL;
  };
  cells = full_poseidon2_round(cells, idx);
}

void poseidonDoIntRounds(std::array<uint32_t, 24>& cells) {
  size_t idx = ROUNDS_HALF_FULL;
  for (size_t i = 0; i < ROUNDS_PARTIAL; i++) {
    cells = partial_poseidon2_round(cells, idx++);
  }
}

void poseidonSponge(std::array<uint32_t, 24>& cells) {
  cells = poseidon2_mix(cells);
}

Poseidon2Rng::Poseidon2Rng() : pool_used(0) {
  for (size_t i = 0; i < CELLS; i++) {
    cells[i] = 0;
  }
}

void Poseidon2Rng::mix(const Digest& data) {
  // if switching from squeezing, do a poseidon2_mix
  if (pool_used != 0) {
    cells = zirgen::poseidon2_mix(cells);
    pool_used = 0;
  }
  // Add in 8 elements (also # of digest words)
  for (size_t i = 0; i < 8; i++) {
    cells[i] += fromMontgomery(data.words[i]);
    cells[i] %= kBabyBearP;
  }
  // Mix
  cells = zirgen::poseidon2_mix(cells);
}

uint32_t Poseidon2Rng::generateBits(size_t bits) {
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

uint32_t Poseidon2Rng::generateFp() {
  if (pool_used == 16) {
    cells = poseidon2_mix(cells);
    pool_used = 0;
  }
  return cells[pool_used++];
}

} // namespace zirgen
