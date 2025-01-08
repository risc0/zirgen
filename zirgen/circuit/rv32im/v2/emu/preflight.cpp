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

#include "zirgen/circuit/rv32im/v2/emu/preflight.h"

#include <cstring>
#include <iostream>
#include <random>

#include "zirgen/circuit/rv32im/v2/emu/paging.h"
#include "zirgen/circuit/rv32im/v2/emu/r0vm.h"

namespace zirgen::rv32im_v2 {

namespace {

struct PreflightContext {
  PreflightTrace& trace;
  const Segment& segment;
  PagedMemory& pager;
  uint32_t pc = 0;
  uint32_t machineMode = 0;
  size_t curWrite = 0;
  size_t curRead = 0;
  uint32_t memCycle = 0;
  uint32_t userCycle = 0;
  uint32_t extraPtr = 0;
  uint32_t ecallPC = 0;
  uint32_t physCycles = 0;
  std::map<uint32_t, uint32_t> origValue;
  std::map<uint32_t, uint32_t> prevCycle;
  std::map<uint32_t, uint32_t> pageMemory;
  bool debug = false;

  PreflightContext(PreflightTrace& trace, const Segment& segment, PagedMemory& pager)
      : trace(trace), segment(segment), pager(pager) {
    for (const auto& kvp : segment.image.getKnownDigests()) {
      for (size_t i = 0; i < 8; i++) {
        pageMemory[nodeIdxToAddr(kvp.first) + i] = kvp.second.words[i];
      }
    }
  }

  void cycleComplete(uint32_t state, uint32_t pc, uint8_t major, uint8_t minor) {
    trace.cycles.emplace_back();
    auto& back = trace.cycles.back();
    back.state = state;
    back.pc = pc;
    back.machineMode = machineMode;
    back.major = major;
    back.minor = minor;
    back.padding = 0;
    back.memCycle = memCycle;
    back.userCycle = userCycle;
    back.extraPtr = extraPtr;
    back.diffCount[0] = 0;
    back.diffCount[1] = 0;

    memCycle = trace.txns.size();
    extraPtr = trace.extra.size();
  }

  void cycleCompleteInst(uint32_t state, uint32_t pc, InstType type) {
    if (type == InstType::EANY) {
      // Technically we need to switch on the machine mode *entering* the EANY
      if (trace.cycles.back().machineMode) {
        cycleComplete(state, pc, MajorType::ECALL0, ECallMinorType::MACHINE_ECALL);
      } else {
        cycleComplete(state, pc, MajorType::CONTROL0, ControlMinorType::USER_ECALL);
      }
    } else if (type == InstType::MRET) {
      cycleComplete(state, pc, MajorType::CONTROL0, ControlMinorType::MRET);
    } else {
      cycleComplete(state, pc, getMajor(type), getMinor(type));
    }
  }

  void cycleCompleteSpecial(uint32_t curState, uint32_t nextState, uint32_t pc) {
    cycleComplete(nextState, pc, 7 + curState / 8, curState % 8);
  }

  void resume() {
    cycleCompleteSpecial(STATE_RESUME, STATE_RESUME, pc);
    if (debug) {
      std::cout << trace.cycles.size() << " Resume\n";
    }
    for (size_t i = 0; i < 8; i++) {
      store(INPUT_WORD + i, 0);
    }
    cycleCompleteSpecial(STATE_RESUME, STATE_DECODE, pc);
  }
  void suspend() {
    pc = 0;
    cycleCompleteSpecial(STATE_SUSPEND, STATE_SUSPEND, 0);
    for (size_t i = 0; i < 8; i++) {
      load(OUTPUT_WORD + i);
    }
    machineMode = 3;
    cycleCompleteSpecial(STATE_SUSPEND, STATE_POSEIDON_ENTRY, 0);
  }
  void instruction(InstType type, const DecodedInst& decoded) {
    if (debug) {
      std::cout << trace.cycles.size() << " Type: " << instName(type) << "\n";
    }
    cycleCompleteInst(STATE_DECODE, pc, type);
    userCycle++;
    physCycles++;
  }
  void ecallCycle(uint32_t curState, uint32_t nextState, uint32_t s0, uint32_t s1, uint32_t s2) {
    if (debug) {
      std::cout << trace.cycles.size() << " ecallCycle\n";
    }
    trace.extra.push_back(s0);
    trace.extra.push_back(s1);
    trace.extra.push_back(s2);
    cycleCompleteSpecial(curState, nextState, pc);
    physCycles++;
  }
  void p2Cycle(uint32_t curState, P2State p2) {
    if (debug) {
      std::cout << trace.cycles.size() << " p2Cycle\n";
    }
    p2.write(trace.extra);
    cycleCompleteSpecial(curState, p2.nextState, pc);
    physCycles++;
  }
  void shaCycle(uint32_t curState, ShaState sha) {
    if (debug) {
      std::cout << trace.cycles.size() << " shaCycle\n";
    }
    sha.write(trace.extra);
    cycleCompleteSpecial(curState, sha.nextState, pc);
    physCycles++;
  }

  void trapRewind() {
    trace.txns.resize(memCycle);
    trace.extra.resize(extraPtr);
  }
  void trap(TrapCause cause) {
    // TODO:
    // cycleComplete(CycleType::CONTROL, ControlSubtype::TRAP, static_cast<uint32_t>(cause));
  }

  // Pass memory ops to pager + record
  uint32_t load(uint32_t word) {
    uint32_t val;
    if (word >= 0x40000000) {
      if (pageMemory.count(word)) {
        val = pageMemory.at(word);
      } else {
        throw std::runtime_error("Invalid load from page memory");
      }
    } else {
      val = pager.load(word);
    }
    MemoryTransaction txn;
    if (!origValue.count(word)) {
      origValue[word] = val;
    }
    txn.word = word;
    txn.cycle = 2 * trace.cycles.size();
    txn.val = val;
    txn.prevCycle = (prevCycle.count(word) ? prevCycle[word] : -1);
    txn.prevVal = val;
    prevCycle[word] = txn.cycle;
    trace.txns.push_back(txn);
    return val;
  }

  void store(uint32_t word, uint32_t val) {
    uint32_t prevVal;
    if (word >= 0x40000000) {
      if (!pageMemory.count(word)) {
        throw std::runtime_error("Invalid write to page memory");
      }
      prevVal = pageMemory[word];
      pageMemory[word] = val;
    } else {
      prevVal = pager.load(word);
      pager.store(word, val);
    }
    MemoryTransaction txn;
    txn.word = word;
    txn.cycle = 2 * trace.cycles.size() + 1;
    txn.val = val;
    txn.prevCycle = (prevCycle.count(word) ? prevCycle[word] : -1);
    txn.prevVal = prevVal;
    prevCycle[word] = txn.cycle;
    trace.txns.push_back(txn);
  }

  // Since hostWrites are ignored, we can return trash
  uint32_t hostPeek(uint32_t word) { return 0; }

  // Replay 'rlen'
  uint32_t write(uint32_t fd, const uint8_t* data, uint32_t len) {
    if (curWrite >= segment.writeRecord.size()) {
      throw std::runtime_error("Invalid segment");
    }
    return segment.writeRecord[curWrite++];
  }
  // Replay data
  uint32_t read(uint32_t fd, uint8_t* data, uint32_t len) {
    if (curRead >= segment.readRecord.size()) {
      throw std::runtime_error("Invalid segment");
    }
    if (segment.readRecord[curRead].size() > len) {
      throw std::runtime_error("Invalid segment");
    }
    size_t rlen = segment.readRecord[curRead].size();
    memcpy(data, segment.readRecord[curRead].data(), rlen);
    curRead++;
    return rlen;
  }

  uint32_t getDigestAddr(uint32_t idx) { return (1 << 30) + 8 * (2 * MEMORY_SIZE_PAGES - idx); }

  void readRoot() {
    size_t rootAddr = getDigestAddr(1);
    for (size_t i = 0; i < 8; i++) {
      load(rootAddr + i);
    }
    cycleCompleteSpecial(STATE_LOAD_ROOT, STATE_POSEIDON_ENTRY, 0);
  }
  void readNode(size_t idx) { p2DoNode(*this, idx, true); }
  void readPage(size_t page) { p2DoPage(*this, page, true); }
  void readDone() { p2ReadDone(*this); }

  void writeNode(size_t idx) { p2DoNode(*this, idx, false); }
  void writePage(size_t page) { p2DoPage(*this, page, false); }
  void writeDone() { p2WriteDone(*this); }
  void writeRoot() {
    size_t rootAddr = getDigestAddr(1);
    for (size_t i = 0; i < 8; i++) {
      load(rootAddr + i);
    }
    cycleCompleteSpecial(STATE_STORE_ROOT, STATE_CONTROL_TABLE, 0);
  }

  void doTables(size_t segmentSize) {
    for (size_t i = 16; i < 256; i += 16) {
      cycleCompleteSpecial(STATE_CONTROL_TABLE, STATE_CONTROL_TABLE, i);
    }
    machineMode = 1;
    for (size_t i = 0; i < 65536; i += 16) {
      cycleCompleteSpecial(STATE_CONTROL_TABLE, STATE_CONTROL_TABLE, i);
    }
    machineMode = 0;
    cycleCompleteSpecial(STATE_CONTROL_TABLE, STATE_CONTROL_DONE, 0);
    if (!segment.isTerminate) {
      if (trace.cycles.size() < segment.segmentThreshold) {
        throw std::runtime_error("Stopping segment too early");
      }
      size_t diff = trace.cycles.size() - segment.segmentThreshold;
      trace.cycles[diff / 2].diffCount[diff % 2]++;
    }
    machineMode = 1;
    cycleCompleteSpecial(STATE_CONTROL_DONE, STATE_CONTROL_DONE, 0);
    while (trace.cycles.size() < segmentSize) {
      cycleCompleteSpecial(STATE_CONTROL_DONE, STATE_CONTROL_DONE, 0);
    }
  }
};

} // namespace

PreflightTrace preflightSegment(const Segment& in, size_t segmentSize) {
  PreflightTrace ret;
  MemoryImage image(in.image);
  PagedMemory pager(image);
  PreflightContext preflightContext(ret, in, pager);
  R0Context<PreflightContext> r0Context(preflightContext);
  RV32Emulator<R0Context<PreflightContext>> emu(r0Context);

  // Do page in
  preflightContext.readRoot();
  auto pages = pager.readPaging();
  p2PagingEntry(preflightContext, 0);
  for (const auto& kvp : pages.nodes) {
    preflightContext.readNode(kvp.first);
  }
  preflightContext.machineMode = 1;
  for (const auto& kvp : pages.pages) {
    preflightContext.readPage(kvp.first);
  }
  preflightContext.machineMode = 2;
  preflightContext.readDone();
  preflightContext.physCycles = 0;

  // Run main execution
  r0Context.resume();
  while (preflightContext.physCycles < in.suspendCycle) {
    emu.step();
  }
  r0Context.suspend();

  // Do page out
  pages = pager.writePaging();
  pager.commit();
  p2PagingEntry(preflightContext, 3);
  for (auto it = pages.pages.rbegin(); it != pages.pages.rend(); ++it) {
    preflightContext.writePage(it->first);
  }
  preflightContext.machineMode = 4;
  for (auto it = pages.nodes.rbegin(); it != pages.nodes.rend(); ++it) {
    preflightContext.writeNode(it->first);
  }
  preflightContext.machineMode = 5;
  preflightContext.writeDone();
  preflightContext.machineMode = 0;
  preflightContext.writeRoot();

  // Do table reification
  ret.tableSplitCycle = ret.cycles.size();
  preflightContext.doTables(segmentSize);

  // Now, go back and update memory transactions to wrap around
  for (auto& txn : ret.txns) {
    if (static_cast<int>(txn.prevCycle) == -1) {
      // If first cycle for word, set to 'prevCycle' to final cycle
      txn.prevCycle = preflightContext.prevCycle[txn.word];
    } else {
      // Otherwise, compute cycle diff and another diff
      uint32_t diff = txn.cycle - txn.prevCycle;
      ret.cycles[diff / 2].diffCount[diff % 2]++;
    }

    // If last cycle, set final value to original value
    if (txn.cycle == preflightContext.prevCycle[txn.word]) {
      txn.val = preflightContext.origValue[txn.word];
    }
  }

  // Finally, update the polynomial logic for paging
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distr(0, risc0::Fp::P);
  ret.rng = risc0::FpExt(distr(gen), distr(gen), distr(gen), distr(gen));
  std::vector<risc0::FpExt> powers;
  risc0::FpExt cur(1);
  for (size_t i = 0; i < 17; i++) {
    powers.push_back(cur);
    cur = cur * ret.rng;
  }
  cur = risc0::FpExt(0);
  for (size_t i = 0; i < ret.cycles.size() - 1; i++) { // Skip last row, since I peek at row + 1
    if (ret.cycles[i].major != MajorType::POSEIDON0 && ret.cycles[i].major != MajorType::POSEIDON1)
      continue;
    uint32_t state = (ret.cycles[i].major - 7) * 8 + ret.cycles[i].minor;
    uint32_t extraOffset = ret.cycles[i + 1].extraPtr - 4;
    uint32_t loadTxType = ret.extra[ret.cycles[i].extraPtr + 5];
    switch (state) {
    case STATE_POSEIDON_LOAD_IN:
      cur = cur * powers[16];
      for (size_t mem = ret.cycles[i].memCycle; mem < ret.cycles[i + 1].memCycle; mem++) {
        size_t j = mem - ret.cycles[i].memCycle;
        auto txn = ret.txns[mem];
        int32_t coeffs[2] = {0, 0};
        switch (loadTxType) {
        case TxKind::READ:
          coeffs[1] = 1;
          break;
        case TxKind::PAGE_IN:
          coeffs[1] = txn.cycle - txn.prevCycle;
          break;
        case TxKind::PAGE_OUT:
          coeffs[0] = (txn.val & 0xffff) - (txn.prevVal & 0xffff);
          coeffs[1] = (txn.val >> 16) - (txn.prevVal >> 16);
          break;
        }
        if (coeffs[0] < 0) {
          coeffs[0] += risc0::Fp::P;
        }
        if (coeffs[1] < 0) {
          coeffs[1] += risc0::Fp::P;
        }
        cur = cur + powers[2 * j + 0] * risc0::FpExt(uint32_t(coeffs[0]));
        cur = cur + powers[2 * j + 1] * risc0::FpExt(uint32_t(coeffs[1]));
      }
    // Fallthrough is intentional
    case STATE_POSEIDON_EXT_ROUND:
    case STATE_POSEIDON_INT_ROUND:
      // Write to extra
      for (size_t i = 0; i < 4; i++) {
        ret.extra[extraOffset + i] = cur.elems[i].asUInt32();
      }
      break;
    default:
      cur = risc0::FpExt(0);
    }
  }

  std::cout << "Memory ops = " << ret.txns.size() << "\n";
  std::cout << "Trace size = " << ret.cycles.size() << "\n";
  return ret;
}

} // namespace zirgen::rv32im_v2
