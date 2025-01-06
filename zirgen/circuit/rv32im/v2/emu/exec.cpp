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

#include "zirgen/circuit/rv32im/v2/emu/exec.h"

#include <cstring>
#include <iostream>

#include "zirgen/circuit/rv32im/v2/emu/paging.h"
#include "zirgen/circuit/rv32im/v2/emu/r0vm.h"

namespace zirgen::rv32im_v2 {

namespace {

struct ExecContext {
  HostIoHandler& upstream;
  PagedMemory& pager;
  Segment* segment;
  size_t pc = 0;
  size_t machineMode = 0;
  size_t userCycles = 0;
  size_t physCycles = 0;
  bool debug = false;

  ExecContext(HostIoHandler& upstream, PagedMemory& pager) : upstream(upstream), pager(pager) {}

  void resume() {}
  void suspend() {}
  void instruction(InstType type, const DecodedInst& decoded) {
    if (debug) {
      std::cout << "pc = " << pc << ", instType = " << instName(type) << "\n";
    }
    userCycles++;
    physCycles++;
  }
  void ecallCycle(uint32_t cur, uint32_t next, uint32_t s0, uint32_t s1, uint32_t s2) {
    physCycles++;
  }
  void p2Cycle(uint32_t cur, const P2State& state) {
    if (debug) {
      std::cout << "poseidon: " << state.nextState << "\n";
    }
    physCycles++;
  }
  void shaCycle(uint32_t cur, const ShaState& state) {
    if (debug) {
      std::cout << "sha: " << state.nextState << "\n";
    }
    physCycles++;
  }
  void trapRewind() {}
  void trap(TrapCause cause) {}

  uint32_t load(uint32_t word) { return pager.load(word); }
  void store(uint32_t word, uint32_t val) { pager.store(word, val); }
  uint32_t hostPeek(uint32_t word) { return pager.peek(word); }

  // For writes, just pass through, record rlen only
  uint32_t write(uint32_t fd, const uint8_t* data, uint32_t len) {
    uint32_t rlen = upstream.write(fd, data, len);
    segment->writeRecord.emplace_back(rlen);
    return rlen;
  }
  // Record what was read during execution so we can replay
  uint32_t read(uint32_t fd, uint8_t* data, uint32_t len) {
    size_t rlen = upstream.read(fd, data, len);
    auto& vec = segment->readRecord.emplace_back();
    vec.resize(rlen);
    memcpy(vec.data(), data, rlen);
    return rlen;
  }
};

} // namespace

std::vector<Segment> execute(
    MemoryImage& in, HostIoHandler& io, size_t segmentThreshold, size_t maxCycles, Digest input) {
  std::vector<Segment> ret;
  PagedMemory pager(in);
  ExecContext execContext(io, pager);
  R0Context<ExecContext> r0Context(execContext);
  RV32Emulator<R0Context<ExecContext>> emu(r0Context);
  ret.emplace_back();
  ret.back().input = input;
  execContext.segment = &ret.back();
  r0Context.resume();
  while (!r0Context.isDone() && execContext.userCycles < maxCycles) {
    if (execContext.physCycles + pager.getPagingCycles() >= segmentThreshold) {
      ret.back().suspendCycle = execContext.physCycles;
      ret.back().pagingCycles = pager.getPagingCycles();
      ret.back().segmentThreshold = segmentThreshold;
      r0Context.suspend();
      ret.back().image = pager.commit();
      ret.back().isTerminate = false;
      pager.clear();
      ret.emplace_back();
      ret.back().input = input;
      execContext.segment = &ret.back();
      r0Context.resume();
    }
    emu.step();
  }
  ret.back().suspendCycle = execContext.physCycles;
  ret.back().pagingCycles = pager.getPagingCycles();
  ret.back().segmentThreshold = segmentThreshold;
  r0Context.suspend();
  ret.back().image = pager.commit();
  ret.back().isTerminate = true;
  return ret;
}

void TestIoHandler::push_u32(uint32_t fd, uint32_t val) {
  for (size_t i = 0; i < 4; i++) {
    input[fd].push_back(val >> (i * 8));
  }
}

uint32_t TestIoHandler::pop_u32(uint32_t fd) {
  uint32_t ret = 0;
  for (size_t i = 0; i < 4; i++) {
    if (output[fd].empty()) {
      throw std::runtime_error("Out of data in pop_u32");
    }
    ret |= output[fd].front() << (i * 8);
    output[fd].pop_front();
  }
  return ret;
}

uint32_t TestIoHandler::write(uint32_t fd, const uint8_t* data, uint32_t len) {
  for (size_t i = 0; i < len; i++) {
    output[fd].push_back(data[i]);
  }
  return len;
}

uint32_t TestIoHandler::read(uint32_t fd, uint8_t* data, uint32_t len) {
  size_t rlen = std::min(len, uint32_t(input[fd].size()));
  for (size_t i = 0; i < rlen; i++) {
    data[i] = input[fd].front();
    input[fd].pop_front();
  }
  return rlen;
}

} // namespace zirgen::rv32im_v2
