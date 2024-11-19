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

#include "zirgen/circuit/verify/fri.h"

#include "zirgen/circuit/verify/merkle.h"
#include "zirgen/circuit/verify/poly.h"

namespace zirgen::verify {

void rev_butterfly(Val* io, size_t po2) {
  if (po2 == 0)
    return;
  size_t half = 1 << (po2 - 1);
  Val step = kRouRev[po2];
  Val cur = 1;
  for (size_t i = 0; i < half; i++) {
    Val a = io[i];
    Val b = io[i + half];
    io[i] = a + b;
    io[i + half] = (a - b) * cur;
    cur = cur * step;
  }
  rev_butterfly(io, po2 - 1);
  rev_butterfly(io + half, po2 - 1);
}

void interpolate_ntt(std::vector<Val>& io) {
  size_t po2 = log2Ceil(io.size());
  rev_butterfly(io.data(), po2);
  Val norm = inv(io.size());
  for (size_t i = 0; i < io.size(); i++) {
    io[i] = io[i] * norm;
  }
}

// A 32-bit reversal of bits
constexpr uint32_t bitReverse(uint32_t x) {
  x = (((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1));
  x = (((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2));
  x = (((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4));
  x = (((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8));
  return ((x >> 16) | (x << 16));
}

void bit_reverse(std::vector<Val>& io) {
  size_t N = log2Ceil(io.size());
  assert((size_t(1) << N) == io.size());
  for (size_t i = 0; i < io.size(); i++) {
    size_t ri = bitReverse(i) >> (32 - N);
    if (i < ri) {
      std::swap(io[i], io[ri]);
    }
  }
}

Val fold_eval(const std::vector<Val>& values, Val x) {
  std::vector<Val> io = values;
  interpolate_ntt(io);
  bit_reverse(io);
  return poly_eval(io, x);
}

Val dynamic_pow(Val in, Val pow, size_t maxPow) {
  Val out = 1;
  Val mul = in;
  for (size_t i = 0; i < log2Ceil(maxPow); i++) {
    Val lowBit = pow & 1;
    Val mulBy = 1 + (mul - 1) * lowBit;
    out = out * mulBy;
    mul = mul * mul;
    pow = (pow - lowBit) / 2;
  }
  return out;
}

namespace {

struct VerifyRoundInfo {
  size_t domain;
  MerkleTreeVerifier merkle;
  Val mix;

  VerifyRoundInfo(ReadIopVal& iop, size_t inDomain)
      : domain(inDomain / kFriFold)
      , merkle("fri", iop, domain, kFriFold, kQueries, /*useExtension=*/true)
      , mix(iop.rngExtVal()) {}

  void verifyQuery(ReadIopVal& iop, Val* pos, Val* goal) const {
    // Compute which group we are in
    Val group = *pos & (domain - 1);
    Val qout = (*pos - group) / domain;
    // Get the column data
    std::vector<Val> data = merkle.verify(iop, group);
    // Check the existing goal
    eq(select(qout, data), *goal);
    // Compute the new goal + pos
    size_t rootPo2 = kRouRev[log2Ceil(kFriFold * domain)];
    Val invWK = dynamic_pow(rootPo2, group, domain);
    *goal = fold_eval(data, mix * invWK);
    *pos = group;
  }
};

} // End namespace

// Verify a FRI proof,
void friVerify(ReadIopVal& iop, size_t deg, InnerVerify inner) {
  size_t domain = deg * kInvRate;
  size_t origDomain = domain;
  std::vector<VerifyRoundInfo> rounds;
  // Prep the folding verfiers
  while (deg > kFriMinDegree) {
    rounds.emplace_back(iop, domain);
    domain /= kFriFold;
    deg /= kFriFold;
  }
  // Grab the final coeffs + commit
  std::vector<Val> finalCoeffs = iop.readExtVals(deg);
  auto digest = hash(finalCoeffs);
  iop.commit(digest);
  // Get the generator for the final polynomial evaluations
  Val gen = kRouFwd[log2Ceil(domain)];
  // Do queries
  for (size_t q = 0; q < kQueries; q++) {
    // Get a 'random' index.
    Val pos = iop.rngBits(log2Ceil(origDomain));
    // Do the 'inner' verification for this index
    Val goal = inner(iop, pos);
    // Verify the per-round proofs
    for (auto& round : rounds) {
      round.verifyQuery(iop, &pos, &goal);
    }
    // Do final verification
    Val x = dynamic_pow(gen, pos, deg * kInvRate);
    Val fx = poly_eval(finalCoeffs, x);
    eq(fx, goal);
  }
}

} // namespace zirgen::verify
