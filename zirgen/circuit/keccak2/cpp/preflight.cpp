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

#include "zirgen/circuit/keccak2/cpp/preflight.h"
#include "zirgen/circuit/keccak2/cpp/wrap_dsl.h"

#include <arpa/inet.h>
#include <array>
#include <cassert>
#include <iostream>

namespace zirgen::keccak2 {

namespace {

// Duplicate Internal implementations of Keccak + Sha because preflight needs
// all the details exposed.

#define ROTL64(x, y) (((x) << (y)) | ((x) >> ((sizeof(uint64_t) * 8) - (y))))

const uint64_t keccak_iota[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

const unsigned keccak_rho[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                                 27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};

const unsigned keccak_pi[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                                15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};

using keccak_t = std::array<uint64_t, 25>;
using theta_b_t = std::array<uint64_t, 5>;

theta_b_t theta_p1(const keccak_t& s) {
  theta_b_t b;
  for (unsigned i = 0; i < 5; i++) {
    b[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];
  }
  return b;
}

void theta_p2_rho_pi(keccak_t& s, const theta_b_t& bc) {
  for (unsigned i = 0; i < 5; i++) {
    uint64_t t = bc[(i + 4) % 5] ^ ROTL64(bc[(i + 1) % 5], 1);
    for (unsigned j = 0; j < 25; j += 5) {
      s[j + i] ^= t;
    }
  }
  uint64_t t1 = s[1];
  for (unsigned i = 0; i < 24; i++) {
    unsigned j = keccak_pi[i];
    uint64_t t2 = s[j];
    s[j] = ROTL64(t1, keccak_rho[i]);
    t1 = t2;
  }
}

void chi_iota(keccak_t& s, uint32_t round) {
  uint64_t t[5];
  for (unsigned j = 0; j < 25; j += 5) {
    for (unsigned i = 0; i < 5; i++) {
      t[i] = s[j + i];
    }
    for (unsigned i = 0; i < 5; i++) {
      s[j + i] ^= (~t[(i + 1) % 5]) & t[(i + 2) % 5];
    }
  }
  s[0] ^= keccak_iota[round];
}

uint32_t sha_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define ROTLEFT(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32 - (b))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x, 2) ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22))
#define EP1(x) (ROTRIGHT(x, 6) ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25))
#define SIG0(x) (ROTRIGHT(x, 7) ^ ROTRIGHT(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ ((x) >> 10))

#define SHA_ROUND_FUNC                                                                             \
  uint32_t t1 = h + EP1(e) + CH(e, f, g) + sha_k[i] + w[i];                                        \
  uint32_t t2 = EP0(a) + MAJ(a, b, c);                                                             \
  h = g;                                                                                           \
  g = f;                                                                                           \
  f = e;                                                                                           \
  e = d + t1;                                                                                      \
  d = c;                                                                                           \
  c = b;                                                                                           \
  b = a;                                                                                           \
  a = t1 + t2;

using sha_state = std::array<uint32_t, 8>;

struct sha_info {
  sha_info() {
    a.fill(0);
    e.fill(0);
    w.fill(0);
  }
  sha_info(const sha_state& state) {
    a.fill(0);
    e.fill(0);
    w.fill(0);
    for (size_t i = 0; i < 4; i++) {
      a[7 - i] = state[i];
      e[7 - i] = state[4 + i];
    }
  }
  std::array<uint32_t, 8> a;
  std::array<uint32_t, 8> e;
  std::array<uint32_t, 8> w;
};

std::vector<sha_info> compute_sha_infos(sha_state& state, const uint32_t* data) {
  uint32_t a = state[0];
  uint32_t b = state[1];
  uint32_t c = state[2];
  uint32_t d = state[3];
  uint32_t e = state[4];
  uint32_t f = state[5];
  uint32_t g = state[6];
  uint32_t h = state[7];
  uint32_t w[64];
  std::vector<sha_info> out;
  sha_info cur;
  for (size_t i = 0; i < 64; i++) {
    if (i < 16) {
      w[i] = htonl(data[i]);
    } else {
      w[i] = SIG1(w[i - 2]) + w[i - 7] + SIG0(w[i - 15]) + w[i - 16];
    }
    SHA_ROUND_FUNC;
    cur.a[i % 8] = a;
    cur.e[i % 8] = e;
    cur.w[i % 8] = w[i];
    if (i % 8 == 7) {
      out.push_back(cur);
    }
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
  out.emplace_back(state);
  return out;
}

sha_state sha_init = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

struct ControlState {
  uint8_t cycleType;
  uint8_t subType;
  uint8_t block;
  uint8_t round;
  uint32_t asWord() const {
    return uint32_t(cycleType) | (uint32_t(subType) << 8) | (uint32_t(block) << 16) |
           (uint32_t(round) << 24);
  }
  static ControlState Shutdown() { return ControlState{0, 0, 0, 0}; }
  static ControlState Read() { return ControlState{1, 0, 0, 0}; }
  static ControlState Expand(uint8_t subtype) { return ControlState{2, subtype, 0, 0}; }
  static ControlState Write() { return ControlState{3, 0, 0, 0}; }
  static ControlState Keccak0(uint8_t round) { return ControlState{4, 0, 0, round}; }
  static ControlState Keccak1(uint8_t round) { return ControlState{5, 0, 0, round}; }
  static ControlState Keccak2(uint8_t round) { return ControlState{6, 0, 0, round}; }
  static ControlState Keccak3(uint8_t round) { return ControlState{7, 0, 0, round}; }
  static ControlState Keccak4(uint8_t round) { return ControlState{8, 0, 0, round}; }
  static ControlState ShaIn(uint8_t block, uint8_t round) {
    return ControlState{9, 0, block, round};
  }
  static ControlState ShaOut(uint8_t block, uint8_t round) {
    return ControlState{9, 1, block, round};
  }
  static ControlState ShaNextBlockIn(uint8_t block) { return ControlState{10, 0, block, 0}; }
  static ControlState ShaNextBlockOut(uint8_t block) { return ControlState{10, 1, block, 0}; }
  static ControlState Init() { return ControlState{11, 0, 0, 0}; }
};

} // namespace

PreflightTrace preflightSegment(const std::vector<KeccakState>& inputs, size_t cycles) {
  auto li = getLayoutInfo();
  PreflightTrace ret;
  ret.preimages = inputs;
  uint32_t curPreimage = 0;
  uint32_t cycle = 0;
  auto addBits = [&](uint16_t col, uint32_t data, uint16_t len) {
    assert(len % 32 == 0);
    ret.scatter.push_back({data, cycle, col, len, 1});
  };
  auto addShorts = [&](uint16_t col, uint32_t data, uint16_t len) {
    assert(len % 2 == 0);
    ret.scatter.push_back({data, cycle, col, len, 16});
  };
  auto addCycle = [&](const ControlState& cstate, uint32_t bits, uint32_t kflat, uint32_t sflat) {
    uint32_t offset = ret.data.size();
    ret.data.push_back(cstate.asWord());
    ret.scatter.push_back({offset, cycle, uint16_t(li.control), 4, 8});
    uint32_t onehot = 1 << cstate.cycleType;
    ret.data.push_back(onehot);
    ret.scatter.push_back({offset + 1, cycle, uint16_t(li.ctypeOneHot), 12, 1});
    addBits(li.bits, bits, 800);
    addShorts(li.kflat, kflat, 100);
    addShorts(li.sflat, sflat, 16);
    ret.curPreimage.push_back(curPreimage);
    cycle++;
  };
  auto writeTheta = [&](const theta_b_t& theta) {
    uint32_t offset = ret.data.size();
    for (size_t i = 0; i < 5; i++) {
      ret.data.push_back(theta[i]);
      ret.data.push_back(theta[i] >> 32);
    }
    for (size_t i = 0; i < 20; i++) {
      ret.data.push_back(0);
    }
    return offset;
  };
  auto writeKeccak = [&](const keccak_t& s, bool high) {
    uint32_t offset = ret.data.size();
    for (size_t i = 0; i < 25; i++) {
      if (high) {
        ret.data.push_back(s[i] >> 32);
      } else {
        ret.data.push_back(s[i]);
      }
    }
    return offset;
  };
  auto writeKFlat = [&](std::vector<uint32_t>& data, const keccak_t& s) {
    data.clear();
    // Write in normal order
    for (size_t i = 0; i < 25; i++) {
      data.push_back(s[i]);
      data.push_back(s[i] >> 32);
    }
    uint32_t offset = ret.data.size();
    ret.data.insert(ret.data.end(), data.begin(), data.end());
    for (size_t i = 0; i < 64 - 50; i++) {
      data.push_back(0); // Pad out sha blocks
    }
    return offset;
  };
  auto writeShaState = [&](const sha_state& state) {
    uint32_t offset = ret.data.size();
    for (size_t i = 0; i < 8; i++) {
      ret.data.push_back(state[i]);
    }
    return offset;
  };
  auto writeShaInfo = [&](const sha_info& info) {
    uint32_t offset = ret.data.size();
    for (size_t i = 0; i < 8; i++) {
      ret.data.push_back(info.a[i]);
    }
    for (size_t i = 0; i < 8; i++) {
      ret.data.push_back(info.e[i]);
    }
    for (size_t i = 0; i < 8; i++) {
      ret.data.push_back(info.w[i]);
    }
    ret.data.push_back(0);
    return offset;
  };
  // 100 zeros @ offset zero (for whereever we need zero)
  uint32_t zeroOffset = ret.data.size();
  for (size_t i = 0; i < 100; i++) {
    ret.data.push_back(0);
  }
  // Initalize sha state (and current data offset)
  sha_state currentSha = sha_init;
  size_t sflatOffset = writeShaState(currentSha);
  // Do an initial 'init' cycle
  addCycle(ControlState::Init(), zeroOffset, zeroOffset, sflatOffset);
  // Do each permutation
  for (size_t input = 0; input < inputs.size(); input++) {
    KeccakState kstate = inputs[input];
    std::vector<uint32_t> data;
    // Do 'read' cycle
    size_t kflatOffset = writeKFlat(data, kstate);
    addCycle(ControlState::Read(), writeShaInfo(sha_info(currentSha)), kflatOffset, sflatOffset);
    curPreimage++;
    // Sha and write all for blocks
    for (size_t block = 0; block < 4; block++) {
      auto infos = compute_sha_infos(currentSha, data.data() + 16 * block);
      for (size_t i = 0; i < 8; i++) {
        addCycle(ControlState::ShaIn(block, i), writeShaInfo(infos[i]), kflatOffset, sflatOffset);
      }
      sflatOffset = writeShaState(currentSha);
      addCycle(
          ControlState::ShaNextBlockIn(block), writeShaInfo(infos[8]), kflatOffset, sflatOffset);
    }
    // Expand
    addCycle(ControlState::Expand(0), writeKeccak(kstate, false), kflatOffset, sflatOffset);
    addCycle(ControlState::Expand(1), writeKeccak(kstate, true), kflatOffset, sflatOffset);
    // Now do the Keccack cycles
    for (size_t round = 0; round < 24; round++) {
      auto theta = theta_p1(kstate);
      addCycle(ControlState::Keccak0(round), writeTheta(theta), kflatOffset, sflatOffset);
      theta_p2_rho_pi(kstate, theta);
      addCycle(ControlState::Keccak1(round), writeKeccak(kstate, false), kflatOffset, sflatOffset);
      addCycle(ControlState::Keccak2(round), writeKeccak(kstate, true), kflatOffset, sflatOffset);
      chi_iota(kstate, round);
      addCycle(ControlState::Keccak3(round), writeKeccak(kstate, false), kflatOffset, sflatOffset);
      addCycle(ControlState::Keccak4(round), writeKeccak(kstate, true), kflatOffset, sflatOffset);
    }
    // Do 'write' cycle
    kflatOffset = writeKFlat(data, kstate);
    addCycle(ControlState::Write(), writeShaInfo(sha_info(currentSha)), kflatOffset, sflatOffset);
    // Sha and write all for blocks
    for (size_t block = 0; block < 4; block++) {
      auto infos = compute_sha_infos(currentSha, data.data() + 16 * block);
      for (size_t i = 0; i < 8; i++) {
        addCycle(ControlState::ShaOut(block, i), writeShaInfo(infos[i]), kflatOffset, sflatOffset);
      }
      sflatOffset = writeShaState(currentSha);
      addCycle(
          ControlState::ShaNextBlockOut(block), writeShaInfo(infos[8]), kflatOffset, sflatOffset);
    }
  }
  // Do 'shudown' cycles until we are done
  while (cycle < cycles) {
    addCycle(ControlState::Shutdown(), zeroOffset, zeroOffset, sflatOffset);
  }

  return ret;
}

void applyPreflight(ExecutionTrace& exec, const PreflightTrace& preflight) {
  for (const auto& info : preflight.scatter) {
    uint32_t innerCount = 32 / info.bitPerElem;
    uint32_t mask = (1 << (info.bitPerElem)) - 1;
    for (size_t i = 0; i < info.count; i++) {
      uint32_t word = preflight.data[info.dataOffset + (i / innerCount)];
      size_t j = i % innerCount;
      uint32_t val = (word >> (j * info.bitPerElem)) & mask;
      // std::cout << "row = " << info.row << ", col = " << info.column + i << ", val = " << val <<
      // "\n";
      exec.data.set(info.row, info.column + i, val);
    }
  }
}

} // namespace zirgen::keccak2
