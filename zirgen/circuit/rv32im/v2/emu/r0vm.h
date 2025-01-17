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

#pragma once

#include <stdexcept>

// Add r0 specific privledged ops to a context

#include "risc0/core/elf.h"
#include "risc0/core/util.h"

#include "zirgen/circuit/rv32im/shared/rv32im.h"
#include "zirgen/circuit/rv32im/v2/platform/constants.h"
#include "zirgen/compiler/zkp/poseidon2.h"

#include "zirgen/circuit/rv32im/v2/emu/p2.h"
#include "zirgen/circuit/rv32im/v2/emu/sha.h"

namespace zirgen::rv32im_v2 {

// A Risc0 specific execution context which proper privledged mode
// instructions and memory mapped registers.  This itself is templated
// on a way to read/write memory.
template <typename Context> struct R0Context {
  // Memory abstraction
  Context& context;
  // Are we done yet?
  bool done;
  // Exit code upon termination
  uint32_t exitCode;

  R0Context(Context& context) : context(context), done(false), exitCode(0) {}

  // Callback when instructions are decoded
  void instDecoded(InstType type, const DecodedInst& decoded) {}

  // Callback at end of normal cycles
  void endNormal(InstType type, const DecodedInst& decoded) { context.instruction(type, decoded); }

  // Manage PC
  uint32_t getPC() { return context.pc; }
  void setPC(uint32_t pc) { context.pc = pc; }

  // Manage registers
  uint32_t loadReg(uint32_t reg) {
    uint32_t base = context.machineMode ? MACHINE_REGS_WORD : USER_REGS_WORD;
    return context.load(base + reg);
  }
  void storeReg(uint32_t reg, uint32_t val) {
    uint32_t base = context.machineMode ? MACHINE_REGS_WORD : USER_REGS_WORD;
    if (reg == 0) {
      base += 64;
    }
    context.store(base + reg, val);
  }

  // Manage memory
  uint32_t loadMem(uint32_t word) { return context.load(word); }
  void storeMem(uint32_t word, uint32_t val) { context.store(word, val); }

  void resume() {
    context.pc = loadMem(SUSPEND_PC_WORD);
    context.machineMode = loadMem(SUSPEND_MODE_WORD);
    context.resume();
  }

  void suspend() {
    storeMem(SUSPEND_PC_WORD, context.pc);
    storeMem(SUSPEND_MODE_WORD, context.machineMode);
    context.suspend();
  }

  // Shared code used by both doTrap and doECALL
  void enterTrap(uint32_t addr) {
    if (context.machineMode) {
      throw std::runtime_error("Cannot trap in machine mode");
    }
    // Save PC + jump
    storeMem(MEPC_WORD, context.pc);
    context.pc = addr;
    // Set machine mode
    context.machineMode = true;
  }

  bool doTrap(TrapCause cause) {
    context.trapRewind();
    uint32_t dispatchAddr = loadMem(TRAP_DISPATCH_WORD + uint32_t(cause));
    if (dispatchAddr % 4 != 0 || dispatchAddr < KERNEL_START_ADDR) {
      throw std::runtime_error("Trap address invalid");
    }
    enterTrap(dispatchAddr);
    context.trap(cause);
    return false;
  }

  // Generic ecall handling for user mode
  bool doUserECALL() {
    uint32_t dispatchAddr = loadMem(ECALL_DISPATCH_WORD);
    if (dispatchAddr % 4 != 0 || dispatchAddr < KERNEL_START_ADDR) {
      return doTrap(TrapCause::ENVIRONMENT_CALL_FROM_U_MODE);
    }
    enterTrap(dispatchAddr);
    return true;
  }

  uint8_t hostPeekByte(uint32_t addr) {
    uint32_t word = addr / 4;
    uint32_t val = context.hostPeek(word);
    uint32_t shift = (addr & 0x3) * 8;
    return val >> shift;
  }

  void writeByte(uint32_t addr, uint8_t byte) {
    uint32_t word = addr / 4;
    uint32_t val = loadMem(word);
    uint32_t shift = (addr & 0x3) * 8;
    val &= ~(0xff << shift);
    val |= uint32_t(byte) << shift;
    storeMem(word, val);
  }

  bool doTerminate() {
    context.ecallCycle(STATE_MACHINE_ECALL, STATE_TERMINATE, 0, 0, 0);
    done = true;
    loadReg(REG_A0);
    loadReg(REG_A1);
    context.ecallCycle(STATE_TERMINATE, STATE_SUSPEND, 0, 0, 0);
    return false;
  }

  uint32_t nextState(uint32_t ptr, uint32_t rlen) {
    if (rlen == 0) {
      return STATE_DECODE;
    }
    if (ptr % 4 != 0) {
      return STATE_HOST_READ_BYTES;
    }
    if (rlen < 4) {
      return STATE_HOST_READ_BYTES;
    }
    return STATE_HOST_READ_WORDS;
  }

  bool doHostRead() {
    context.ecallCycle(STATE_MACHINE_ECALL, STATE_HOST_READ_SETUP, 0, 0, 0);
    uint32_t curState = STATE_HOST_READ_SETUP;
    uint32_t fd = loadReg(REG_A0);
    uint32_t ptr = loadReg(REG_A1);
    uint32_t len = loadReg(REG_A2);
    if (ptr + len < ptr) {
      throw std::runtime_error("Invalid wrapping host read");
    }
    if (len > 1024) {
      throw std::runtime_error("Invalid large host read");
    }
    uint32_t rlen = 0;
    std::vector<uint8_t> bytes(len);
    rlen = context.read(fd, bytes.data(), len);
    storeReg(REG_A0, rlen);
    uint32_t i = 0;
    if (rlen == 0) {
      context.pc += 4;
    }
    context.ecallCycle(curState, nextState(ptr, rlen), ptr / 4, ptr % 4, rlen);
    curState = nextState(ptr, rlen);
    while (rlen > 0 && ptr % 4 != 0) {
      writeByte(ptr, bytes[i]);
      ptr++;
      i++;
      rlen--;
      if (rlen == 0) {
        context.pc += 4;
      }
      context.ecallCycle(curState, nextState(ptr, rlen), ptr / 4, ptr % 4, rlen);
      curState = nextState(ptr, rlen);
    }
    while (rlen >= 4) {
      uint32_t words = std::min(rlen / 4, uint32_t(4));
      for (size_t j = 0; j < 4; j++) {
        if (j < words) {
          uint32_t word = 0;
          for (size_t k = 0; k < 4; k++) {
            word |= bytes[i + k] << (8 * k);
          }
          storeMem(ptr / 4, word);
          ptr += 4;
          i += 4;
          rlen -= 4;
        } else {
          storeMem(SAFE_WRITE_WORD, 0);
        }
      }
      if (rlen == 0) {
        context.pc += 4;
      }
      context.ecallCycle(curState, nextState(ptr, rlen), ptr / 4, ptr % 4, rlen);
      curState = nextState(ptr, rlen);
    }
    while (rlen > 0) {
      writeByte(ptr, bytes[i]);
      ptr++;
      i++;
      rlen--;
      if (rlen == 0) {
        context.pc += 4;
      }
      context.ecallCycle(curState, nextState(ptr, rlen), ptr / 4, ptr % 4, rlen);
      curState = nextState(ptr, rlen);
    }
    return false;
  }

  bool doHostWrite() {
    context.ecallCycle(STATE_MACHINE_ECALL, STATE_HOST_WRITE, 0, 0, 0);
    uint32_t fd = loadReg(REG_A0);
    uint32_t ptr = loadReg(REG_A1);
    uint32_t len = loadReg(REG_A2);
    if (ptr + len < ptr) {
      throw std::runtime_error("Invalid wrapping host write");
    }
    if (len > 1024) {
      // Technically, this is a bit silly since host writes are free
      // But we probably need some bound, so now it's consistent
      throw std::runtime_error("Invalid large host write");
    }
    uint32_t rlen = len;
    std::vector<uint8_t> bytes(len);
    for (size_t i = 0; i < len; i++) {
      bytes[i] = hostPeekByte(ptr + i);
    }
    rlen = context.write(fd, bytes.data(), len);
    storeReg(REG_A0, rlen);
    context.pc += 4;
    context.ecallCycle(STATE_HOST_WRITE, STATE_DECODE, 0, 0, 0);
    return false;
  }

  bool doPoseidon2() {
    // Bump PC
    context.pc += 4;
    context.ecallCycle(STATE_MACHINE_ECALL, STATE_POSEIDON_ENTRY, 0, 0, 0);
    p2ECall(context);
    return false;
  }

  bool doSha2() {
    // Bump PC
    context.pc += 4;
    context.ecallCycle(STATE_MACHINE_ECALL, STATE_SHA_ECALL, 0, 0, 0);
    ShaECall(context);
    return false;
  }

  // Machine mode ECALL, allow for overrides in subclasses
  bool doMachineECALL() {
    switch (loadReg(REG_A7)) {
    case HOST_ECALL_TERMINATE:
      return doTerminate();
    case HOST_ECALL_READ:
      return doHostRead();
    case HOST_ECALL_WRITE:
      return doHostWrite();
    case HOST_ECALL_POSEIDON2:
      return doPoseidon2();
    case HOST_ECALL_SHA2:
      return doSha2();
    default:
      throw std::runtime_error("unimplemented machine ECALL");
    }
  }

  // Context handlers start here
  bool isDone() { return done; }

  // Memory access checking
  bool checkInstLoad(uint32_t addr) {
    uint32_t word = addr / 4;
    if (word < ZERO_PAGE_END_WORD) {
      return false;
    }
    if (!context.machineMode && word >= KERNEL_START_WORD) {
      return false;
    }
    return true;
  }
  bool checkDataLoad(uint32_t addr) {
    uint32_t word = addr / 4;
    if (context.machineMode) {
      return true;
    }
    return word >= ZERO_PAGE_END_WORD && word < KERNEL_START_WORD;
  }
  bool checkDataStore(uint32_t addr) { return checkDataLoad(addr); }

  bool doECALL() {
    if (context.machineMode) {
      doMachineECALL();
      return false;
    } else {
      return doUserECALL();
    }
  }

  bool doMRET() {
    if (!context.machineMode) {
      throw std::runtime_error("Cannot MRET in user mode");
    }
    // load PC
    context.pc = loadMem(MEPC_WORD) + 4;
    // Set machine mode
    context.machineMode = false;
    return true;
  }
};

inline void loadWithKernel(std::map<uint32_t, uint32_t>& words,
                           const std::string& kernelElf,
                           const std::string& userElf) {
  auto kernelElfBytes = risc0::loadFile(kernelElf);
  auto userElfBytes = risc0::loadFile(userElf);
  // Set MEPC so MRET jumpts to start of user mode code
  uint32_t userEntry = risc0::loadElf(userElfBytes, words, ZERO_PAGE_END_WORD, KERNEL_START_WORD);
  words[MEPC_WORD] = userEntry - 4;
  // Put start info into the memory image
  uint32_t kernelEntry = risc0::loadElf(kernelElfBytes, words, KERNEL_START_WORD, KERNEL_END_WORD);
  words[SUSPEND_PC_WORD] = kernelEntry;
  words[SUSPEND_MODE_WORD] = 1;
}

inline void loadRaw(std::map<uint32_t, uint32_t>& words, const std::string& elf) {
  auto elfBytes = risc0::loadFile(elf);
  uint32_t entry = risc0::loadElf(elfBytes, words, ZERO_PAGE_END_WORD, KERNEL_END_WORD);
  words[SUSPEND_PC_WORD] = entry;
  words[SUSPEND_MODE_WORD] = 1;
}

} // namespace zirgen::rv32im_v2
