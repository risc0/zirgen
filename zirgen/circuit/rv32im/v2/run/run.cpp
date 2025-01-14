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

#include <deque>
#include <iostream>

#include "risc0/core/elf.h"
#include "risc0/core/util.h"
#include "zirgen/circuit/rv32im/v2/emu/exec.h"
#include "zirgen/circuit/rv32im/v2/emu/preflight.h"
#include "zirgen/circuit/rv32im/v2/emu/r0vm.h"
#include "zirgen/circuit/rv32im/v2/run/wrap_dsl.h"

namespace zirgen::rv32im_v2 {

struct MemTxnKey {
  uint32_t addr;
  uint32_t cycle;
  uint32_t data;
  bool operator<(const MemTxnKey& rhs) const {
    if (addr != rhs.addr) {
      return addr < rhs.addr;
    }
    if (cycle != rhs.cycle) {
      return cycle < rhs.cycle;
    }
    return data < rhs.data;
  }
};

struct LookupTables {
  std::map<uint32_t, risc0::Fp> tableU8;
  std::map<uint32_t, risc0::Fp> tableU16;
  std::map<MemTxnKey, risc0::Fp> tableMem;
  std::map<risc0::Fp, risc0::Fp> tableCycle;

  void lookupDelta(risc0::Fp table, risc0::Fp index, risc0::Fp count) {
    uint32_t tableU32 = table.asUInt32();
    if (tableU32 == 0) {
      tableCycle[index] += count;
      return;
    }
    if (tableU32 != 8 && tableU32 != 16) {
      throw std::runtime_error("Invalid lookup table");
    }
    if (index.asUInt32() >= (1U << tableU32)) {
      std::cerr << "LOOKUP ERROR: table = " << table.asUInt32() << ", index = " << index.asUInt32()
                << "\n";
      throw std::runtime_error("u8/16 table error");
    }
    if (tableU32 == 8) {
      tableU8[index.asUInt32()] += count;
    } else {
      tableU16[index.asUInt32()] += count;
    }
  }

  risc0::Fp lookupCurrent(risc0::Fp table, risc0::Fp index) {
    uint32_t tableU32 = table.asUInt32();
    if (tableU32 != 8 && tableU32 != 16) {
      throw std::runtime_error("Invalid lookup table");
    }
    if (tableU32 == 8) {
      return tableU8[index.asUInt32()];
    } else {
      return tableU16[index.asUInt32()];
    }
  }

  void memoryDelta(uint32_t addr, uint32_t cycle, uint32_t data, risc0::Fp count) {
    tableMem[{addr, cycle, data}] += count;
  }

  void check() {
    for (const auto& kvp : tableU8) {
      if (kvp.second != 0) {
        std::cerr << "U8 entry " << kvp.first << ": " << kvp.second.asUInt32() << "\n";
        throw std::runtime_error("Table not zero");
      }
    }
    for (const auto& kvp : tableU16) {
      if (kvp.second != 0) {
        std::cerr << "U16 entry " << kvp.first << ": " << kvp.second.asUInt32() << "\n";
        throw std::runtime_error("Table not zero");
      }
    }
    for (const auto& kvp : tableMem) {
      if (kvp.second != 0) {
        std::cerr << "Nonzero memory entry: (" << kvp.first.addr << ", " << kvp.first.cycle << ", "
                  << kvp.first.data << ") = " << kvp.second.asUInt32() << "\n";
        throw std::runtime_error("Table not zero");
      }
    }
    for (const auto& kvp : tableCycle) {
      if (kvp.second != 0) {
        std::cerr << "Cycle entry " << kvp.first.asUInt32() << ": " << kvp.second.asUInt32()
                  << "\n";
        throw std::runtime_error("Table not zero");
      }
    }
  }
};

struct ReplayHandler : public StepHandler {
  ReplayHandler(const PreflightTrace& preflight, LookupTables& tables, size_t cycle)
      : preflight(preflight), tables(tables), cycle(cycle), which(0) {}

  std::pair<uint32_t, uint32_t> getMajorMinor() override {
    return std::make_pair(uint32_t(preflight.cycles[cycle].major),
                          uint32_t(preflight.cycles[cycle].minor));
  }

  MemoryTransaction getMemoryTxn(uint32_t addr) override {
    size_t memCycle = preflight.cycles[cycle].memCycle + which;
    const auto& txn = preflight.txns[memCycle];
    printf("getMemoryTxn(%lu, 0x%08x): txn(%u, 0x%08x, 0x%08x)\n",
           cycle,
           addr,
           txn.cycle,
           txn.word,
           txn.val);
    which++;
    if (txn.word != addr) {
      std::cerr << "txn.word = " << txn.word << ", addr = " << addr << "\n";
      throw std::runtime_error("memory peek not in replay");
    }
    return txn;
  }

  uint32_t readPrepare(uint32_t fd, uint32_t size) override {
    size_t memCycle = preflight.cycles[cycle].memCycle;
    return preflight.txns[memCycle + 3].val;
  }

  std::vector<uint8_t> readBytes(uint32_t count) override {
    // TODO
    return {0, 0, 0, 0};
  }

  uint32_t write(uint32_t fd, uint32_t addr, uint32_t size) override {
    size_t memCycle = preflight.cycles[cycle].memCycle;
    return preflight.txns[memCycle + 3].val;
  }

  std::vector<uint32_t> nextPagingIdx() override {
    size_t extraStart = preflight.cycles[cycle].extraPtr;
    size_t extraEnd = preflight.cycles[cycle + 1].extraPtr;
    size_t extraSize = extraEnd - extraStart;
    P2State p2;
    p2.read(&preflight.extra[extraStart], extraSize);
    uint32_t idx = nodeAddrToIdx(p2.bufOutAddr);
    return {idx, preflight.cycles[cycle].machineMode};
  }

  void lookupDelta(risc0::Fp table, risc0::Fp index, risc0::Fp count) override {
    tables.lookupDelta(table, index, count);
  }

  risc0::Fp lookupCurrent(risc0::Fp table, risc0::Fp index) override {
    return tables.lookupCurrent(table, index);
  }

  void memoryDelta(uint32_t addr, uint32_t cycle, uint32_t data, risc0::Fp count) override {
    tables.memoryDelta(addr, cycle, data, count);
  }

  uint32_t getDiffCount(uint32_t cycle) override {
    return preflight.cycles[cycle / 2].diffCount[cycle % 2];
  }

  const PreflightTrace& preflight;
  LookupTables& tables;
  size_t cycle;
  size_t which;
};

ExecutionTrace runSegment(const Segment& segment, size_t segmentSize) {
  auto rootIn = segment.image.getDigest(1);
  auto preflightTrace = preflightSegment(segment, segmentSize);
  size_t cycles = preflightTrace.cycles.size();
  std::cout << "**** TRACE cycle: " << cycles << "\n";
  std::cout << "Segment main cycle count: " << segment.suspendCycle << "\n";
  std::cout << "Segment paging count: " << segment.pagingCycles << "\n";
  ExecutionTrace trace(cycles, getDslParams());
  // Set globals:
  // TODO: Don't hardcode column numbers
  trace.global.set(37, segment.segmentThreshold);
  for (size_t i = 0; i < 8; i++) {
    // State in
    trace.global.set(38 + 2 * i, rootIn.words[i] & 0xffff);
    trace.global.set(38 + 2 * i + 1, rootIn.words[i] >> 16);
    // Input digest
    trace.global.set(0 + 2 * i, segment.input.words[i] & 0xffff);
    trace.global.set(0 + 2 * i + 1, segment.input.words[i] >> 16);
  }
  // Set RNG
  for (size_t i = 0; i < 4; i++) {
    trace.global.set(33 + i, preflightTrace.rng.elems[i]);
  }
  // Set isTerminate
  trace.global.set(16, segment.isTerminate);
  // Set stateful columns from 'top'
  for (size_t i = 0; i < cycles; i++) {
    std::cout << "Cycle: " << i << ", pc = " << preflightTrace.cycles[i].pc / 4
              << ", state = " << preflightTrace.cycles[i].state << "\n";
    trace.data.set(i, getCycleCol(), i);
    trace.data.set(i, getTopStateCol() + 0, preflightTrace.cycles[i].pc & 0xffff);
    trace.data.set(i, getTopStateCol() + 1, preflightTrace.cycles[i].pc >> 16);
    trace.data.set(i, getTopStateCol() + 2, preflightTrace.cycles[i].state);
    trace.data.set(i, getTopStateCol() + 3, preflightTrace.cycles[i].machineMode);
    size_t extraStart = preflightTrace.cycles[i].extraPtr;
    size_t extraEnd =
        (i == cycles - 1) ? preflightTrace.extra.size() : preflightTrace.cycles[i + 1].extraPtr;
    size_t extraSize = extraEnd - extraStart;
    if (extraSize == 3) {
      for (size_t j = 0; j < extraSize; j++) {
        trace.data.set(i, getEcall0StateCol() + j, preflightTrace.extra[extraStart + j]);
      }
    } else if (extraSize == sizeof(P2State) / 4) {
      for (size_t j = 0; j < extraSize; j++) {
        // std::cout << "  extra: " << preflightTrace.extra[extraStart + j] << "\n";
        trace.data.set(i, getPoseidonStateCol() + j, preflightTrace.extra[extraStart + j]);
      }
    } else if (extraSize == sizeof(ShaState) / 4) {
      for (size_t j = 0; j < ShaState::FpCount; j++) {
        trace.data.set(i, getShaStateCol() + j, preflightTrace.extra[extraStart + j]);
      }
      for (size_t j = 0; j < ShaState::U32Count; j++) {
        uint32_t val = preflightTrace.extra[extraStart + ShaState::FpCount + j];
        std::cout << "  SHA_WORD: " << val << "\n";
        for (size_t k = 0; k < 32; k++) {
          trace.data.set(i, getShaStateCol() + ShaState::FpCount + 32 * j + k, (val >> k) & 1);
        }
      }
    }
  }

  LookupTables tables;
  // for (size_t i = 0; i < preflightTrace.tableSplitCycle; i++) {
  for (size_t i = preflightTrace.tableSplitCycle; i-- > 0;) {
    std::cout << "Running cycle " << i << "\n";
    ReplayHandler memory(preflightTrace, tables, i);
    DslStep(memory, trace, i);
  }
  // for (size_t i = preflightTrace.tableSplitCycle; i < cycles; i++) {
  for (size_t i = cycles; i-- > preflightTrace.tableSplitCycle;) {
    std::cout << "Running cycle " << i << "\n";
    ReplayHandler memory(preflightTrace, tables, i);
    DslStep(memory, trace, i);
  }
  tables.check();
  // 'Randomize' mix
  for (size_t i = 0; i < trace.mix.getCols(); i++) {
    trace.mix.set(i, i * i + 77);
  }
  // Zero any undecided values in data
  trace.data.setUnset();
  // Do accum
  std::cout << "Doing accum\n";
  // Make final accum == 0
  for (size_t i = 0; i < 4; i++) {
    trace.accum.set(cycles - 1, trace.accum.getCols() - 4 + i, 0);
  }
  for (size_t i = 0; i < cycles; i++) {
    ReplayHandler memory(preflightTrace, tables, i);
    DslStepAccum(memory, trace, i);
  }
  // TODO: Check final is zero?
  std::cout << "Done\n";
  return trace;
}

} // namespace zirgen::rv32im_v2
