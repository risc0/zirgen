// Copyright 2025 RISC Zero, Inc.
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

#include "zirgen/circuit/keccak/cpp/preflight.h"
#include "zirgen/circuit/keccak/cpp/wrap_dsl.h"
#include "zirgen/compiler/zkp/poseidon2.h"

using cells_t = std::array<uint32_t, 24>;

#include <arpa/inet.h>
#include <array>
#include <cassert>
#include <iostream>

namespace zirgen::keccak {

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

struct ControlState {
  uint8_t cycleType;
  uint8_t subType;
  uint8_t round;
  uint32_t asWord() const {
    return uint32_t(cycleType) | (uint32_t(subType) << 8) | (uint32_t(round) << 16);
  }
  static ControlState Shutdown() { return ControlState{0, 0, 0}; }
  static ControlState Read() { return ControlState{1, 0, 0}; }
  static ControlState Expand(uint8_t subtype) { return ControlState{2, subtype, 0}; }
  static ControlState Write() { return ControlState{3, 0, 0}; }
  static ControlState Keccak0(uint8_t round) { return ControlState{4, 0, round}; }
  static ControlState Keccak1(uint8_t round) { return ControlState{5, 0, round}; }
  static ControlState keccak(uint8_t round) { return ControlState{6, 0, round}; }
  static ControlState Keccak3(uint8_t round) { return ControlState{7, 0, round}; }
  static ControlState Keccak4(uint8_t round) { return ControlState{8, 0, round}; }
  static ControlState Poseidon2In(uint8_t round) { return ControlState{9, 0, round}; }
  static ControlState Poseidon2Out(uint8_t round) { return ControlState{9, 1, round}; }
  static ControlState Init() { return ControlState{10, 0, 0}; }
};

} // namespace

PreflightTrace preflightSegment(const std::vector<KeccakState>& inputs, size_t cycles) {
  auto li = getLayoutInfo();
  PreflightTrace ret;
  ret.preimages = inputs;
  uint32_t curPreimage = 0;
  uint32_t cycle = 0;
  auto addBits = [&](uint16_t col, uint32_t data, uint16_t len) {
    ret.scatter.push_back({data, cycle, col, len, 1});
  };
  auto addBytes = [&](uint16_t col, uint32_t data, uint16_t len) {
    ret.scatter.push_back({data, cycle, col, len, 8});
  };
  auto addShorts = [&](uint16_t col, uint32_t data, uint16_t len) {
    ret.scatter.push_back({data, cycle, col, len, 16});
  };
  auto addWords = [&](uint16_t col, uint32_t data, uint16_t len) {
    ret.scatter.push_back({data, cycle, col, len, 32});
  };
  auto addCycle = [&](const ControlState& cstate,
                      uint32_t bits,
                      uint32_t kflat,
                      uint32_t pflat,
                      bool isBits = true) {
    uint32_t offset = ret.data.size();
    ret.data.push_back(cstate.asWord());
    uint32_t onehot = 1 << cstate.cycleType;
    ret.data.push_back(onehot);
    ret.data.push_back(cycle);
    addBytes(li.control, offset, 3);
    addBits(li.ctypeOneHot, offset + 1, 11);
    if (isBits) {
      addBits(li.bits, bits, 800);
    } else {
      addWords(li.bits, bits, 800);
    }
    addShorts(li.kflat, kflat, 100);
    addWords(li.pflat, pflat, 24);
    addWords(li.cycleNum, offset + 2, 1);
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
      data.push_back(0); // Pad out to 7 poseidon blocks
    }
    return offset;
  };
  auto writePFlat = [&](cells_t& cells) {
    uint32_t offset = ret.data.size();
    ret.data.insert(ret.data.end(), cells.begin(), cells.end());
    return offset;
  };

  // 100 zeros @ offset zero (for whereever we need zero)
  uint32_t zeroOffset = ret.data.size();
  for (size_t i = 0; i < 100; i++) {
    ret.data.push_back(0);
  }
  // Initalize sha state (and current data offset)
  cells_t currentP2 = {0};
  size_t pflatOffset = writePFlat(currentP2);
  // Do an initial 'init' cycle
  addCycle(ControlState::Init(), zeroOffset, zeroOffset, pflatOffset);
  // Do each permutation
  for (size_t input = 0; input < inputs.size(); input++) {
    KeccakState kstate = inputs[input];
    std::vector<uint32_t> data;
    // Do 'read' cycle
    size_t kflatOffset = writeKFlat(data, kstate);
    addCycle(ControlState::Read(), zeroOffset, kflatOffset, pflatOffset);
    curPreimage++;
    // Poseidon2 input
    for (size_t round = 0; round < 7; round++) {
      for (size_t i = 0; i < 8; i++) {
        uint32_t word = data[round * 8 + i];
        currentP2[2 * i] = word & 0xffff;
        currentP2[2 * i + 1] = word >> 16;
      }
      zirgen::poseidonSponge(currentP2);
      pflatOffset = writePFlat(currentP2);
      addCycle(ControlState::Poseidon2In(round), zeroOffset, kflatOffset, pflatOffset);
    }
    // Expand
    addCycle(ControlState::Expand(0), writeKeccak(kstate, false), kflatOffset, pflatOffset);
    addCycle(ControlState::Expand(1), writeKeccak(kstate, true), kflatOffset, pflatOffset);
    // Now do the Keccack cycles
    for (size_t round = 0; round < 24; round++) {
      auto theta = theta_p1(kstate);
      addCycle(ControlState::Keccak0(round), writeTheta(theta), kflatOffset, pflatOffset);
      theta_p2_rho_pi(kstate, theta);
      addCycle(ControlState::Keccak1(round), writeKeccak(kstate, false), kflatOffset, pflatOffset);
      addCycle(ControlState::keccak(round), writeKeccak(kstate, true), kflatOffset, pflatOffset);
      chi_iota(kstate, round);
      addCycle(ControlState::Keccak3(round), writeKeccak(kstate, false), kflatOffset, pflatOffset);
      addCycle(ControlState::Keccak4(round), writeKeccak(kstate, true), kflatOffset, pflatOffset);
    }
    // Do 'write' cycle
    kflatOffset = writeKFlat(data, kstate);
    addCycle(ControlState::Write(), zeroOffset, kflatOffset, pflatOffset);
    // Poseidon2 output
    for (size_t round = 0; round < 7; round++) {
      for (size_t i = 0; i < 8; i++) {
        uint32_t word = data[round * 8 + i];
        currentP2[2 * i] = word & 0xffff;
        currentP2[2 * i + 1] = word >> 16;
      }
      zirgen::poseidonSponge(currentP2);
      pflatOffset = writePFlat(currentP2);
      addCycle(ControlState::Poseidon2Out(round), zeroOffset, kflatOffset, pflatOffset);
    }
  }
  // Do 'shudown' cycles until we are done
  while (cycle < cycles) {
    addCycle(ControlState::Shutdown(), zeroOffset, zeroOffset, pflatOffset);
  }

  return ret;
}

void applyPreflight(ExecutionTrace& exec, const PreflightTrace& preflight) {
  for (const auto& info : preflight.scatter) {
    uint32_t innerCount = 32 / info.bitPerElem;
    uint32_t mask = (1 << (info.bitPerElem)) - 1;
    if (info.bitPerElem == 32) {
      mask = 0xffffffff;
    }
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

} // namespace zirgen::keccak
