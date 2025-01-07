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

#include <arpa/inet.h>

namespace zirgen::rv32im_v2 {

// 1 to 1 state from inst_sha
struct ShaState {
  static constexpr size_t FpCount = 7;  // Number of Fp values
  static constexpr size_t U32Count = 3; // Number of U32 value
  uint32_t stateInAddr;
  uint32_t stateOutAddr;
  uint32_t dataAddr;
  uint32_t count;
  uint32_t kAddr;
  uint32_t round;
  uint32_t nextState;
  uint32_t a;
  uint32_t e;
  uint32_t w;

  void write(std::vector<uint32_t>& out) {
    const uint32_t* data = reinterpret_cast<const uint32_t*>(this);
    for (size_t i = 0; i < sizeof(ShaState) / 4; i++) {
      out.push_back(data[i]);
    }
  }

  void read(const uint32_t* in, size_t count) {
    assert(count == sizeof(ShaState) / 4);
    uint32_t* data = reinterpret_cast<uint32_t*>(this);
    for (size_t i = 0; i < count; i++) {
      data[i] = in[i];
    }
  }
};

template <size_t size> struct RingBuffer {
  std::array<uint32_t, size> buf;
  uint32_t cur = 0;
  uint32_t back(size_t i) const { return buf[(size + cur - i) % size]; }
  void push(uint32_t val) {
    buf[cur] = val;
    cur++;
    cur %= size;
  }
};

#define ROTRIGHT(a, b) (((a) >> (b)) | ((a) << (32 - (b))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x, 2) ^ ROTRIGHT(x, 13) ^ ROTRIGHT(x, 22))
#define EP1(x) (ROTRIGHT(x, 6) ^ ROTRIGHT(x, 11) ^ ROTRIGHT(x, 25))
#define SIG0(x) (ROTRIGHT(x, 7) ^ ROTRIGHT(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x, 17) ^ ROTRIGHT(x, 19) ^ ((x) >> 10))

inline std::pair<uint32_t, uint32_t>
computeAE(const RingBuffer<68>& oldA, const RingBuffer<68>& oldE, uint32_t k, uint32_t w) {
  uint32_t a = oldA.back(1);
  uint32_t b = oldA.back(2);
  uint32_t c = oldA.back(3);
  uint32_t d = oldA.back(4);
  uint32_t e = oldE.back(1);
  uint32_t f = oldE.back(2);
  uint32_t g = oldE.back(3);
  uint32_t h = oldE.back(4);
  uint32_t t1 = h + EP1(e) + CH(e, f, g) + k + w;
  uint32_t t2 = EP0(a) + MAJ(a, b, c);
  e = d + t1;
  a = t1 + t2;
  return std::make_pair(a, e);
}

inline uint32_t computeW(const RingBuffer<16>& oldW) {
  return SIG1(oldW.back(2)) + oldW.back(7) + SIG0(oldW.back(15)) + oldW.back(16);
}

template <typename Context> void ShaECall(Context& context) {
  ShaState sha;
  sha.stateInAddr = context.load(MACHINE_REGS_WORD + REG_A0) / 4;
  sha.stateOutAddr = context.load(MACHINE_REGS_WORD + REG_A1) / 4;
  sha.dataAddr = context.load(MACHINE_REGS_WORD + REG_A2) / 4;
  sha.count = context.load(MACHINE_REGS_WORD + REG_A3) & 0xffff;
  sha.kAddr = context.load(MACHINE_REGS_WORD + REG_A4) / 4;
  sha.round = 0;
  sha.a = 0;
  sha.e = 0;
  sha.w = 0;
  uint32_t curState = STATE_SHA_ECALL;
  auto step = [&](uint32_t nextState) {
    sha.nextState = nextState;
    context.shaCycle(curState, sha);
    curState = nextState;
  };
  RingBuffer<68> oldA;
  RingBuffer<68> oldE;
  RingBuffer<16> oldW;
  for (size_t i = 0; i < 4; i++) {
    sha.round = i;
    step(STATE_SHA_LOAD_STATE);
    uint32_t leA = context.load(sha.stateInAddr + 3 - i);
    uint32_t leE = context.load(sha.stateInAddr + 7 - i);
    sha.a = htonl(leA);
    sha.e = htonl(leE);
    oldA.push(sha.a);
    oldE.push(sha.e);
    context.store(sha.stateOutAddr + 3 - i, leA);
    context.store(sha.stateOutAddr + 7 - i, leE);
  }
  while (sha.count != 0) {
    for (size_t i = 0; i < 16; i++) {
      sha.round = i;
      step(STATE_SHA_LOAD_DATA);
      uint32_t k = context.load(sha.kAddr + i);
      sha.w = htonl(context.load(sha.dataAddr));
      sha.dataAddr++;
      oldW.push(sha.w);
      auto ae = computeAE(oldA, oldE, k, sha.w);
      sha.a = ae.first;
      sha.e = ae.second;
      oldA.push(sha.a);
      oldE.push(sha.e);
    }
    for (size_t i = 0; i < 48; i++) {
      sha.round = i;
      step(STATE_SHA_MIX);
      uint32_t k = context.load(sha.kAddr + 16 + i);
      sha.w = computeW(oldW);
      oldW.push(sha.w);
      auto ae = computeAE(oldA, oldE, k, sha.w);
      sha.a = ae.first;
      sha.e = ae.second;
      oldA.push(sha.a);
      oldE.push(sha.e);
    }
    for (size_t i = 0; i < 4; i++) {
      sha.round = i;
      step(STATE_SHA_STORE_STATE);
      sha.a = oldA.back(4) + oldA.back(68);
      sha.e = oldE.back(4) + oldE.back(68);
      sha.w = 0;
      if (i == 3) {
        sha.count--;
      }
      oldA.push(sha.a);
      oldE.push(sha.e);
      context.store(sha.stateOutAddr + 3 - i, htonl(sha.a));
      context.store(sha.stateOutAddr + 7 - i, htonl(sha.e));
    }
  }
  sha.round = 0;
  step(STATE_DECODE);
}

} // namespace zirgen::rv32im_v2
