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

namespace zirgen::rv32im_v2 {

// 1 to 1 state from inst_p2
struct P2State {
  uint32_t hasState;
  uint32_t stateAddr;
  uint32_t bufOutAddr;
  uint32_t isElem;
  uint32_t checkOut;
  uint32_t loadTxType;

  uint32_t nextState;
  uint32_t subState;
  uint32_t bufInAddr;
  uint32_t count;
  uint32_t mode;

  std::array<uint32_t, 24> cells;
  std::array<uint32_t, 4> zcheck;

  void write(std::vector<uint32_t>& out) {
    zcheck.fill(0);
    const uint32_t* data = reinterpret_cast<const uint32_t*>(this);
    for (size_t i = 0; i < sizeof(P2State) / 4; i++) {
      out.push_back(data[i]);
    }
  }

  void read(const uint32_t* in, size_t count) {
    assert(count == sizeof(P2State) / 4);
    uint32_t* data = reinterpret_cast<uint32_t*>(this);
    for (size_t i = 0; i < count; i++) {
      data[i] = in[i];
    }
  }
};

/*
class P2ContextConcept {
  uint32_t load(uint32_t word);
  void store(uint32_t word, uint32_t val);
  void p2Cycle(uint32_t curState, const P2State& info);
};
*/

// Perform all but the first step of poseidon2
template <typename Context> void p2Rest(Context& context, P2State p2, uint32_t finalState) {
  uint32_t curState = p2.nextState;
  auto step = [&](uint32_t nextState, uint32_t subState) {
    p2.nextState = nextState;
    p2.subState = subState;
    context.p2Cycle(curState, p2);
    curState = nextState;
  };
  if (p2.hasState) { // If we have state, load it
    step(STATE_POSEIDON_LOAD_STATE, 0);
    for (size_t i = 0; i < 8; i++) {
      p2.cells[16 + i] = context.load(p2.stateAddr + i);
    }
  }
  while (p2.count) { // While we have data to process
    // Do load
    step(STATE_POSEIDON_LOAD_IN, 0);
    if (p2.isElem) {
      for (size_t i = 0; i < 16; i++) {
        if (i == 8) {
          step(STATE_POSEIDON_LOAD_IN, 1);
        }
        p2.cells[i] = context.load(p2.bufInAddr++);
      }
    } else {
      for (size_t i = 0; i < 8; i++) {
        uint32_t word = context.load(p2.bufInAddr++);
        p2.cells[2 * i] = word & 0xffff;
        p2.cells[2 * i + 1] = word >> 16;
      }
    }
    // Do the mix
    poseidonMultiplyByMExt(p2.cells);
    for (size_t i = 0; i < 4; i++) {
      step(STATE_POSEIDON_EXT_ROUND, i);
      poseidonDoExtRound(p2.cells, i);
    }
    step(STATE_POSEIDON_INT_ROUND, 0);
    poseidonDoIntRounds(p2.cells);
    for (size_t i = 4; i < 8; i++) {
      step(STATE_POSEIDON_EXT_ROUND, i);
      poseidonDoExtRound(p2.cells, i);
    }
    p2.count--;
  }
  step(STATE_POSEIDON_DO_OUT, 0);
  if (p2.checkOut) {
    for (size_t i = 0; i < 8; i++) {
      if (context.load(p2.bufOutAddr + i) != p2.cells[i]) {
        throw std::runtime_error("Poseidon2 check failed");
      }
    }
  } else {
    for (size_t i = 0; i < 8; i++) {
      context.store(p2.bufOutAddr + i, p2.cells[i]);
    }
  }
  p2.bufInAddr = 0;
  if (p2.hasState) {
    step(STATE_POSEIDON_STORE_STATE, 0);
    for (size_t i = 0; i < 8; i++) {
      context.store(p2.stateAddr + i, p2.cells[16 + i]);
    }
  }
  step(finalState, 0);
}

template <typename Context> void p2ECall(Context& context) {
  P2State p2;
  p2.cells.fill(0);
  p2.stateAddr = context.load(MACHINE_REGS_WORD + REG_A0) / 4;
  p2.bufInAddr = context.load(MACHINE_REGS_WORD + REG_A1) / 4;
  p2.bufOutAddr = context.load(MACHINE_REGS_WORD + REG_A2) / 4;
  uint32_t bitsAndCount = context.load(MACHINE_REGS_WORD + REG_A3);
  p2.hasState = (p2.stateAddr != 0);
  p2.isElem = (bitsAndCount & PFLAG_IS_ELEM) ? 1 : 0;
  p2.checkOut = (bitsAndCount & PFLAG_CHECK_OUT) ? 1 : 0;
  p2.count = bitsAndCount & 0xffff;
  p2.mode = 1;
  p2.loadTxType = TxKind::READ;
  p2.nextState = STATE_POSEIDON_ENTRY;
  p2.subState = 0;
  p2Rest(context, p2, STATE_DECODE);
}

inline uint32_t nodeIdxToAddr(uint32_t idx) {
  return 0x44000000 - idx * 8;
}

inline uint32_t nodeAddrToIdx(uint32_t addr) {
  return (0x44000000 - addr) / 8;
}

template <typename Context> void p2DoNode(Context& context, size_t idx, bool isRead) {
  P2State p2;
  p2.cells.fill(0);
  p2.hasState = 0;
  p2.stateAddr = 0;
  p2.bufOutAddr = nodeIdxToAddr(idx);
  p2.isElem = 1;
  p2.checkOut = isRead;
  p2.loadTxType = (isRead ? TxKind::PAGE_IN : TxKind::PAGE_OUT);
  p2.nextState = STATE_POSEIDON_PAGING;
  p2.subState = 0;
  p2.bufInAddr = nodeIdxToAddr(2 * idx + 1);
  p2.count = 1;
  p2.mode = (isRead ? 0 : 4);
  p2Rest(context, p2, STATE_POSEIDON_PAGING);
}

template <typename Context> void p2DoPage(Context& context, size_t page, bool isRead) {
  size_t idx = page + 4 * 1024 * 1024;
  P2State p2;
  p2.cells.fill(0);
  p2.hasState = 0;
  p2.stateAddr = 0;
  p2.bufOutAddr = nodeIdxToAddr(idx);
  p2.isElem = 0;
  p2.checkOut = isRead;
  p2.loadTxType = (isRead ? TxKind::PAGE_IN : TxKind::PAGE_OUT);
  p2.nextState = STATE_POSEIDON_PAGING;
  p2.subState = 0;
  p2.bufInAddr = page * 1024 / 4;
  p2.count = 1024 / 8 / 4;
  p2.mode = (isRead ? 1 : 3);
  p2Rest(context, p2, STATE_POSEIDON_PAGING);
}

template <typename Context> void p2ReadDone(Context& context) {
  P2State p2;
  p2.cells.fill(0);
  p2.hasState = 0;
  p2.stateAddr = 0;
  p2.bufOutAddr = 0x40000000;
  p2.isElem = 0;
  p2.checkOut = 0;
  p2.loadTxType = 0;
  p2.nextState = STATE_RESUME;
  p2.subState = 0;
  p2.bufInAddr = 0;
  p2.count = 0;
  p2.mode = 2;
  context.p2Cycle(STATE_POSEIDON_PAGING, p2);
}

template <typename Context> void p2WriteDone(Context& context) {
  P2State p2;
  p2.cells.fill(0);
  p2.hasState = 0;
  p2.stateAddr = 0;
  p2.bufOutAddr = 0x44000000;
  p2.isElem = 0;
  p2.checkOut = 0;
  p2.loadTxType = 0;
  p2.nextState = STATE_STORE_ROOT;
  p2.subState = 0;
  p2.bufInAddr = 0;
  p2.count = 0;
  p2.mode = 5;
  context.p2Cycle(STATE_POSEIDON_PAGING, p2);
}

template <typename Context> void p2PagingEntry(Context& context, uint32_t mode) {
  P2State p2;
  p2.cells.fill(0);
  p2.hasState = 0;
  p2.stateAddr = 0;
  p2.bufOutAddr = (mode ? 0x40000000 : 0x44000000);
  p2.isElem = 1;
  p2.checkOut = 1;
  p2.loadTxType = TxKind::PAGE_IN;
  p2.nextState = STATE_POSEIDON_PAGING;
  p2.subState = 0;
  p2.bufInAddr = 0;
  p2.count = 0;
  p2.mode = mode;
  context.p2Cycle(STATE_POSEIDON_ENTRY, p2);
}

} // namespace zirgen::rv32im_v2
