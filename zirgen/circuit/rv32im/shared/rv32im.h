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

// A very minimal and compile time configurable rv32im emulator
// Also: matches circuit very accurately

#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

// clang-format off
// One-to-one with the table in major_minor.zir
#define ALL_INSTS \
    TblEntry(0, 0, 0x33, 0x0, 0x00, ADD) \
    TblEntry(0, 1, 0x33, 0x0, 0x20, SUB) \
    TblEntry(0, 2, 0x33, 0x4, 0x00, XOR) \
    TblEntry(0, 3, 0x33, 0x6, 0x00, OR) \
    TblEntry(0, 4, 0x33, 0x7, 0x00, AND) \
    TblEntry(0, 5, 0x33, 0x2, 0x00, SLT) \
    TblEntry(0, 6, 0x33, 0x3, 0x00, SLTU) \
    TblEntry(0, 7, 0x13, 0x0,   -1, ADDI) \
    TblEntry(1, 0, 0x13, 0x4,   -1, XORI) \
    TblEntry(1, 1, 0x13, 0x6,   -1, ORI) \
    TblEntry(1, 2, 0x13, 0x7,   -1, ANDI) \
    TblEntry(1, 3, 0x13, 0x2,   -1, SLTI) \
    TblEntry(1, 4, 0x13, 0x3,   -1, SLTIU) \
    TblEntry(1, 5, 0x63, 0x0,   -1, BEQ) \
    TblEntry(1, 6, 0x63, 0x1,   -1, BNE) \
    TblEntry(1, 7, 0x63, 0x4,   -1, BLT) \
    TblEntry(2, 0, 0x63, 0x5,   -1, BGE) \
    TblEntry(2, 1, 0x63, 0x6,   -1, BLTU) \
    TblEntry(2, 2, 0x63, 0x7,   -1, BGEU) \
    TblEntry(2, 3, 0x6f,  -1,   -1, JAL) \
    TblEntry(2, 4, 0x67, 0x0,   -1, JALR) \
    TblEntry(2, 5, 0x37,  -1,   -1, LUI) \
    TblEntry(2, 6, 0x17,  -1,   -1, AUIPC) \
    TblEntry(3, 0, 0x33, 0x1, 0x00, SLL) \
    TblEntry(3, 1, 0x13, 0x1, 0x00, SLLI) \
    TblEntry(3, 2, 0x33, 0x0, 0x01, MUL) \
    TblEntry(3, 3, 0x33, 0x1, 0x01, MULH) \
    TblEntry(3, 4, 0x33, 0x2, 0x01, MULHSU) \
    TblEntry(3, 5, 0x33, 0x3, 0x01, MULHU) \
    TblEntry(4, 0, 0x33, 0x5, 0x00, SRL) \
    TblEntry(4, 1, 0x33, 0x5, 0x20, SRA) \
    TblEntry(4, 2, 0x13, 0x5, 0x00, SRLI) \
    TblEntry(4, 3, 0x13, 0x5, 0x20, SRAI) \
    TblEntry(4, 4, 0x33, 0x4, 0x01, DIV) \
    TblEntry(4, 5, 0x33, 0x5, 0x01, DIVU) \
    TblEntry(4, 6, 0x33, 0x6, 0x01, REM) \
    TblEntry(4, 7, 0x33, 0x7, 0x01, REMU) \
    TblEntry(5, 0, 0x03, 0x0,   -1, LB) \
    TblEntry(5, 1, 0x03, 0x1,   -1, LH) \
    TblEntry(5, 2, 0x03, 0x2,   -1, LW) \
    TblEntry(5, 3, 0x03, 0x4,   -1, LBU) \
    TblEntry(5, 4, 0x03, 0x5,   -1, LHU) \
    TblEntry(6, 0, 0x23, 0x0,   -1, SB) \
    TblEntry(6, 1, 0x23, 0x1,   -1, SH) \
    TblEntry(6, 2, 0x23, 0x2,   -1, SW) \
    TblEntry(7, 0, 0x73, 0x0, 0x00, EANY) \
    TblEntry(7, 1, 0x73, 0x0, 0x18, MRET)
// clang-format on

namespace zirgen {

#define TblEntry(major, minor, opcode, func3, func7, name) name = major * 8 + minor,
enum class InstType : uint8_t {
  ALL_INSTS INVALID = 255,
};
#undef TblEntry

inline uint8_t getMajor(InstType inst) {
  return static_cast<uint8_t>(inst) / 8;
}
inline uint8_t getMinor(InstType inst) {
  return static_cast<uint8_t>(inst) % 8;
}

#define TblEntry(major, minor, opcode, func3, func7, name)                                         \
  case InstType::name:                                                                             \
    return #name;
inline const char* instName(InstType inst) {
  switch (inst) {
    ALL_INSTS
  case InstType::INVALID:
    return "INVALID";
  default:
    return "<ERROR>";
  }
}
#undef TblEntry

enum class TrapCause {
  INSTRUCTION_ADDRESS_MISALIGNED = 0,
  INSTRUCTION_ACCESS_FAULT = 1,
  ILLEGAL_INSTRUCTION = 2,
  BREAKPOINT = 3,
  LOAD_ADDRESS_MISALIGNED = 4,
  LOAD_ACCESS_FAULT = 5,
  STORE_ADDRESS_MISALIGNED = 6,
  STORE_ACCESS_FAULT = 7,
  ENVIRONMENT_CALL_FROM_U_MODE = 8,
};

// Decomposed instruction
struct DecodedInst {
  DecodedInst(uint32_t inst)
      : inst(inst)
      , topBit((inst & 0x80000000) >> 31)
      , func7((inst & 0xfe000000) >> 25)
      , rs2((inst & 0x01f00000) >> 20)
      , rs1((inst & 0x000f8000) >> 15)
      , func3((inst & 0x00007000) >> 12)
      , rd((inst & 0x00000f80) >> 7)
      , opcode((inst & 0x0000007f)) {}

  uint32_t inst;
  uint32_t topBit;
  uint32_t func7;
  uint32_t rs2;
  uint32_t rs1;
  uint32_t func3;
  uint32_t rd;
  uint32_t opcode;
  uint32_t immB() const {
    return (topBit * 0xfffff000) | ((rd & 1) << 11) | ((func7 & 0x3f) << 5) | (rd & 0x1e);
  }
  uint32_t immI() const { return (topBit * 0xfffff000) | (func7 << 5) | rs2; }
  uint32_t immS() const { return (topBit * 0xfffff000) | (func7 << 5) | rd; }
  uint32_t immJ() const {
    return (topBit * 0xfff00000) | (rs1 << 15) | (func3 << 12) | ((rs2 & 1) << 11) |
           ((func7 & 0x3f) << 5) | (rs2 & 0x1e);
  }
  uint32_t immU() const { return inst & 0xfffff000; }
};

// RISC-V instruction are determined by 3 parts:
// - Opcode: 7 bits
// - Func3: 3 bits
// - Func7: 7 bits
// In many cases, func7 and/or func3 is ignored.  A stardard trick is to decode
// via a table, but a 17 bit lookup table destroys L1 cache.  Luckily for us,
// in practice the low 2 bits of opcode are always 11, so we can drop them, and
// also func7 is always either 0, 1, 0x20 or don't care, so we can reduce func7
// to 2 bits, which gets us to 10 bits, which is only 1k.

class FastDecodeTable {
public:
  FastDecodeTable() {
    table.resize(1 << 10, InstType::INVALID);
    addRV32IM();
  }

  InstType lookup(const DecodedInst& decoded) const {
    return table[map10(decoded.opcode, decoded.func3, decoded.func7)];
  }

private:
  std::vector<InstType> table;

  // Map to 10 bit format
  size_t map10(uint32_t opcode, int32_t func3, int32_t func7) const {
    uint32_t opHigh = opcode >> 2;
    // Map 0 -> 0, 1 -> 1, 0x20 -> 2, everything else to 3
    uint32_t func72bits = (func7 <= 1 ? func7 : (func7 == 0x20 ? 2 : 3));
    return (opHigh << 5) | (func72bits << 3) | func3;
  }

  void addInst(uint32_t opcode, int32_t func3, int32_t func7, InstType inst) {
    uint32_t opHigh = opcode >> 2;
    if (func3 < 0) {
      for (uint32_t f3 = 0; f3 < 8; f3++) {
        for (uint32_t f7b = 0; f7b < 4; f7b++) {
          table[(opHigh << 5) | (f7b << 3) | f3] = inst;
        }
      }
      return;
    }
    if (func7 < 0) {
      for (uint32_t f7b = 0; f7b < 4; f7b++) {
        table[(opHigh << 5) | (f7b << 3) | func3] = inst;
      }
      return;
    }
    table[map10(opcode, func3, func7)] = inst;
  }

  void addRV32IM() {
#define TblEntry(major, minor, opcode, func3, func7, name)                                         \
  addInst(opcode, func3, func7, InstType::name);
    ALL_INSTS
#undef TblEntry
  }
};

// The emulator class is templated on a 'context' object that allows
// other code to observe the and override various parts of the emulator.
// We provide a simple base version of this for testing
struct BaseContext {
  bool done = false;
  uint32_t pc;
  uint32_t regs[32];
  std::map<uint32_t, uint32_t> memory;

  BaseContext() : done(false), pc(0) {
    for (size_t i = 0; i < 32; i++) {
      regs[i] = 0;
    }
  }

  // Let emulator decide when to stop
  bool isDone() { return done; }

  // Memory access checking, defaults to allowing anything
  bool checkInstLoad(uint32_t addr) { return true; }
  bool checkDataLoad(uint32_t addr) { return true; }
  bool checkDataStore(uint32_t addr) { return true; }

  // Handle privledged instructions
  bool doECALL() {
    done = true; // ECALL induces termination
    return true; // end normally
  }

  bool doMRET() { throw std::runtime_error("Unimplemented"); }

  bool doTrap(TrapCause cause) { throw std::runtime_error("Unimplemented"); }

  // Callback when instructions are decoded
  void instDecoded(InstType type, const DecodedInst& decoded) {}

  // Callback when instructions end normally
  void endNormal(InstType type, const DecodedInst& decoded) {}

  // Manage PC
  uint32_t getPC() { return pc; }
  void setPC(uint32_t pc) { this->pc = pc; }

  // Manage registers
  uint32_t loadReg(uint32_t reg) { return regs[reg]; }
  void storeReg(uint32_t reg, uint32_t val) {
    if (reg) {
      regs[reg] = val;
    }
  }

  // Manage memory
  uint32_t loadMem(uint32_t word) { return memory[word]; }
  void storeMem(uint32_t word, uint32_t val) { memory[word] = val; }
};

template <typename Context> class RV32Emulator {
private:
  FastDecodeTable decodeTable;
  Context& context;

public:
  RV32Emulator(Context& context) : context(context) {}

  // Run for a bounded number of steps
  bool run(size_t maxSteps) {
    size_t curStep = 0;
    while (!context.isDone() && curStep < maxSteps) {
      step();
      curStep++;
    }
    return context.isDone();
  }

  // Do a single step
  void step() {
    uint32_t pc = context.getPC();
    if (!context.checkInstLoad(pc)) {
      context.doTrap(TrapCause::INSTRUCTION_ACCESS_FAULT);
      return;
    }
    uint32_t inst = context.loadMem(pc / 4);
    if ((inst & 0x3) != 0x3) {
      context.doTrap(TrapCause::ILLEGAL_INSTRUCTION);
      return;
    }
    DecodedInst decoded(inst);
    InstType type = decodeTable.lookup(decoded);
    context.instDecoded(type, decoded);
    bool ret;
    switch (getMajor(type)) {
    case 0:
    case 1:
    case 2:
      ret = stepMisc(type, decoded);
      break;
    case 3:
      ret = stepMul(type, decoded);
      break;
    case 4:
      ret = stepDiv(type, decoded);
      break;
    case 5:
      ret = stepLoad(type, decoded);
      break;
    case 6:
      ret = stepStore(type, decoded);
      break;
    case 7:
      ret = stepPriv(type, decoded);
      break;
    default:
      ret = context.doTrap(TrapCause::ILLEGAL_INSTRUCTION);
      break;
    }
    if (ret)
      context.endNormal(type, decoded);
  }

private:
  bool stepMisc(InstType type, const DecodedInst& decoded) {
    uint32_t pc = context.getPC();
    uint32_t newPC = pc + 4;
    uint32_t rd = decoded.rd;
    uint32_t out = 0;
    uint32_t rs1 = context.loadReg(decoded.rs1);
    uint32_t rs2 = context.loadReg(decoded.rs2);
    uint32_t immI = decoded.immI();
    auto br_cond = [&](bool cond) {
      rd = 0;
      if (cond) {
        newPC = pc + decoded.immB();
      }
    };
    switch (type) {
    case InstType::ADD:
      out = rs1 + rs2;
      break;
    case InstType::SUB:
      out = rs1 - rs2;
      break;
    case InstType::XOR:
      out = rs1 ^ rs2;
      break;
    case InstType::OR:
      out = rs1 | rs2;
      break;
    case InstType::AND:
      out = rs1 & rs2;
      break;
    case InstType::SLT:
      out = (int32_t(rs1) < int32_t(rs2));
      break;
    case InstType::SLTU:
      out = (rs1 < rs2);
      break;
    case InstType::ADDI:
      out = rs1 + immI;
      break;
    case InstType::XORI:
      out = rs1 ^ immI;
      break;
    case InstType::ORI:
      out = rs1 | immI;
      break;
    case InstType::ANDI:
      out = rs1 & immI;
      break;
    case InstType::SLTI:
      out = (int32_t(rs1) < int32_t(immI));
      break;
    case InstType::SLTIU:
      out = (rs1 < immI);
      break;
    case InstType::BEQ:
      br_cond(rs1 == rs2);
      break;
    case InstType::BNE:
      br_cond(rs1 != rs2);
      break;
    case InstType::BLT:
      br_cond(int32_t(rs1) < int32_t(rs2));
      break;
    case InstType::BGE:
      br_cond(int32_t(rs1) >= int32_t(rs2));
      break;
    case InstType::BLTU:
      br_cond(rs1 < rs2);
      break;
    case InstType::BGEU:
      br_cond(rs1 >= rs2);
      break;
    case InstType::JAL:
      out = pc + 4;
      newPC = pc + decoded.immJ();
      break;
    case InstType::JALR:
      out = pc + 4;
      newPC = rs1 + immI;
      break;
    case InstType::LUI:
      out = decoded.immU();
      break;
    case InstType::AUIPC:
      out = pc + decoded.immU();
      break;
    default:
      __builtin_unreachable();
    }
    if (newPC % 4 != 0) {
      return context.doTrap(TrapCause::INSTRUCTION_ADDRESS_MISALIGNED);
    }
    context.storeReg(rd, out);
    context.setPC(newPC);
    return true;
  }
  bool stepMul(InstType type, const DecodedInst& decoded) {
    uint32_t rs1 = context.loadReg(decoded.rs1);
    uint32_t rs2 = context.loadReg(decoded.rs2);
    uint32_t immI = decoded.immI();
    uint32_t out = 0;
    switch (type) {
    case InstType::SLL:
      out = rs1 << (rs2 & 0x1f);
      break;
    case InstType::SLLI:
      out = rs1 << (immI & 0x1f);
      break;
    case InstType::MUL:
      out = rs1 * rs2;
      break;
    case InstType::MULH:
      out = uint64_t(int64_t(int32_t(rs1)) * int64_t(int32_t(rs2))) >> 32;
      break;
    case InstType::MULHSU:
      out = uint64_t(int64_t(int32_t(rs1)) * int64_t(uint32_t(rs2))) >> 32;
      break;
    case InstType::MULHU:
      out = (uint64_t(rs1) * uint64_t(rs2)) >> 32;
      break;
    default:
      __builtin_unreachable();
    }
    context.storeReg(decoded.rd, out);
    context.setPC(context.getPC() + 4);
    return true;
  }
  bool stepDiv(InstType type, const DecodedInst& decoded) {
    uint32_t rs1 = context.loadReg(decoded.rs1);
    uint32_t rs2 = context.loadReg(decoded.rs2);
    uint32_t immI = decoded.immI();
    uint32_t out = 0;
    switch (type) {
    case InstType::SRL:
      out = rs1 >> (rs2 & 0x1f);
      break;
    case InstType::SRA:
      out = int32_t(rs1) >> (rs2 & 0x1f);
      break;
    case InstType::SRLI:
      out = rs1 >> (immI & 0x1f);
      break;
    case InstType::SRAI:
      out = int32_t(rs1) >> (immI & 0x1f);
      break;
    case InstType::SLLI:
      out = rs1 << (immI & 0x1f);
      break;
    case InstType::DIV:
      if (rs1 == 0x80000000 && rs2 == 0xffffffff) {
        out = rs1;
      } else {
        out = (rs2 == 0 ? -1 : int32_t(rs1) / int32_t(rs2));
      }
      break;
    case InstType::DIVU:
      out = (rs2 == 0 ? -1 : rs1 / rs2);
      break;
    case InstType::REM:
      if (rs1 == 0x80000000 && rs2 == 0xffffffff) {
        out = 0;
      } else {
        out = (rs2 == 0 ? rs1 : int32_t(rs1) % int32_t(rs2));
      }
      break;
    case InstType::REMU:
      out = (rs2 == 0 ? rs1 : rs1 % rs2);
      break;
    default:
      __builtin_unreachable();
    }
    context.storeReg(decoded.rd, out);
    context.setPC(context.getPC() + 4);
    return true;
  }

  bool stepLoad(InstType type, const DecodedInst& decoded) {
    uint32_t rs1 = context.loadReg(decoded.rs1);
    uint32_t addr = rs1 + decoded.immI();
    if (!context.checkDataLoad(addr)) {
      return context.doTrap(TrapCause::LOAD_ACCESS_FAULT);
    }
    uint32_t data = context.loadMem(addr / 4);
    uint32_t out = 0;
    uint32_t shift = 8 * (addr & 3);
    switch (type) {
    case InstType::LB:
      out = (data >> shift) & 0xff;
      if (out & 0x80) {
        out |= 0xffffff00;
      }
      break;
    case InstType::LH:
      if (addr & 0x1) {
        return context.doTrap(TrapCause::LOAD_ADDRESS_MISALIGNED);
      }
      out = (data >> shift) & 0xffff;
      if (out & 0x8000) {
        out |= 0xffff0000;
      }
      break;
    case InstType::LW:
      if (addr & 0x3) {
        return context.doTrap(TrapCause::LOAD_ADDRESS_MISALIGNED);
      }
      out = data;
      break;
    case InstType::LBU:
      out = (data >> shift) & 0xff;
      break;
    case InstType::LHU:
      if (addr & 0x1) {
        return context.doTrap(TrapCause::LOAD_ADDRESS_MISALIGNED);
      }
      out = (data >> shift) & 0xffff;
      break;
    default:
      __builtin_unreachable();
    }
    context.storeReg(decoded.rd, out);
    context.setPC(context.getPC() + 4);
    return true;
  }

  bool stepStore(InstType type, const DecodedInst& decoded) {
    uint32_t rs1 = context.loadReg(decoded.rs1);
    uint32_t rs2 = context.loadReg(decoded.rs2);
    uint32_t addr = rs1 + decoded.immS();
    uint32_t shift = 8 * (addr & 3);
    if (!context.checkDataStore(addr)) {
      return context.doTrap(TrapCause::STORE_ACCESS_FAULT);
    }
    uint32_t data = context.loadMem(addr / 4);
    switch (type) {
    case InstType::SB:
      data ^= data & (0xff << shift);
      data |= (rs2 & 0xff) << shift;
      break;
    case InstType::SH:
      if (addr & 0x1) {
        return context.doTrap(TrapCause::STORE_ADDRESS_MISALIGNED);
      }
      data ^= data & (0xffff << shift);
      data |= (rs2 & 0xffff) << shift;
      break;
    case InstType::SW:
      if (addr & 0x3) {
        return context.doTrap(TrapCause::STORE_ADDRESS_MISALIGNED);
      }
      data = rs2;
      break;
    default:
      __builtin_unreachable();
    }
    context.storeMem(addr / 4, data);
    context.setPC(context.getPC() + 4);
    return true;
  }

  bool stepPriv(InstType type, const DecodedInst& decoded) {
    switch (type) {
    case InstType::EANY:
      switch (decoded.rs2) {
      case 0:
        return context.doECALL();
      case 1:
        return context.doTrap(TrapCause::BREAKPOINT);
      default:
        return context.doTrap(TrapCause::ILLEGAL_INSTRUCTION);
      }
    case InstType::MRET:
      return context.doMRET();
    default:
      __builtin_unreachable();
    }
    return true;
  }
};

} // namespace zirgen
