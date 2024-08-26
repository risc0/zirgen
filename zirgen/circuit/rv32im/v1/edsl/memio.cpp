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

#include "zirgen/circuit/rv32im/v1/edsl/compute.h"

#include "zirgen/circuit/rv32im/v1/edsl/top.h"

namespace zirgen::rv32im_v1 {

MemIOCycleImpl::MemIOCycleImpl(RamHeader ramHeader) : ram(ramHeader, 5) {}

void MemIOCycleImpl::set(Top top) {
  // Get some basic state data
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  Val curPC = BACK(1, body->pc->get());
  Val userMode = BACK(1, body->userMode->get());

  // We should always be in 'decode'
  eqz(BACK(1, body->nextMajor->get()) - MajorType::kMuxSize);

  // Read and decode the next instruction
  U32Val inst = readInst->doRead(cycle, curPC / 4);
  decoder->set(inst);

  // Pick minor select (also non-det)
  NONDET {
    Val minor =
        doExtern("getMinor", "", 1, std::vector<Val>(inst.bytes.begin(), inst.bytes.end()))[0];
    minorSelect->set(minor);
  }

  // Nondeterministically set imm
  NONDET{
#define OPI(id, mnemonic, opc, f3, f7, immFmt, isRead, is8Bit, is16Bit, signExt)                   \
  IF(minorSelect->at(id % kMinorMuxSize)) { immReg->set(decoder->imm##immFmt()); }
#include "zirgen/circuit/rv32im/v1/platform/rv32im.inl"
  }

  // Load RS1 + RS2
  U32Val rs1 = readRS1->doRead(cycle, kRegisterOffset - 32 * userMode + decoder->rs1());
  U32Val rs2 = readRS2->doRead(cycle, kRegisterOffset - 32 * userMode + decoder->rs2());
  U32Val imm = immReg->get();

  // Make sure not to write when rd = x0
  rdZero->set(decoder->rd());

  // Add rd + rs1 and split the low bytes into parts
  NONDET {
    Val byte0 = rs1.bytes[0] + imm.bytes[0];
    lowBits->set(byte0 & 3);
    carry[0]->set((byte0 & 256) / 256);
    Val rem = byte0 - lowBits - 256 * carry[0];
    bytes[0]->setExact(rem);
    check6->setExact(rem / 4);
    Val byte1 = rs1.bytes[1] + imm.bytes[1] + carry[0];
    bytes[1]->setExact(byte1 & 0xff);
    carry[1]->setExact((byte1 & 256) / 256);
    Val byte2 = rs1.bytes[2] + imm.bytes[2] + carry[1];
    bytes[2]->setExact(byte2 & 0xff);
    carry[2]->setExact((byte2 & 256) / 256);
    Val byte3 = rs1.bytes[3] + imm.bytes[3] + carry[2];
    checkHigh[0]->set(byte3 & 3);
    checkHigh[1]->set((byte3 & 12) / 4);
    carry[3]->set((byte3 & 256) / 256);
  }
  // This verifies that bytes[0] hold at most 6 bits, and ends in 00
  eq(bytes[0], 4 * check6);
  // This verifies that the first two bytes sum decomposes as desired
  eq(rs1.bytes[0] + imm.bytes[0], 256 * carry[0] + bytes[0] + lowBits);
  // Check byte1
  eq(rs1.bytes[1] + imm.bytes[1] + carry[0], 256 * carry[1] + bytes[1]);
  // Check byte2
  eq(rs1.bytes[2] + imm.bytes[2] + carry[1], 256 * carry[2] + bytes[2]);
  // Check byte3
  eq(rs1.bytes[3] + imm.bytes[3] + carry[2], 256 * carry[3] + 4 * checkHigh[1] + checkHigh[0]);
  // Make sure to disallow user memory IO from system RAM (top 1/4 of ram)
  Val top2 = checkHigh[1];
  eqz(top2 * (1 - top2) * (2 - top2));
  // Also, disallow anything above the first half in user mode
  eqz(body->userMode * top2 * (1 - top2));

  // OK, now I can finally compute the actual nice memory addr
  Val addr = checkHigh[1] * (1 << 24) + checkHigh[0] * (1 << 22) + bytes[2] * (1 << 14) +
             bytes[1] * (1 << 6) + check6;
  U32Val loaded = readMem->doRead(cycle, addr);
  XLOG("  imm=%w, rs1=x%u -> %w, rs2=x%u -> %w, Addr = %10x, lowBits = %u, loaded = %w",
       imm,
       decoder->rs1(),
       rs1,
       decoder->rs2(),
       rs2,
       addr,
       lowBits,
       loaded);

  body->pc->set(curPC + 4);
  body->nextMajor->set(MajorType::kMuxSize);

#define OPI(id, mnemonic, opc, f3, f7, immFmt, isRead, is8Bit, is16Bit, signExt)                   \
  {                                                                                                \
    uint32_t is32Bit = 1 - is8Bit - is16Bit;                                                       \
    uint32_t addrMask = is32Bit * 0 + is16Bit * 2 + is8Bit * 3;                                    \
    uint32_t count = 4 - addrMask;                                                                 \
    IF(minorSelect->at(id % kMinorMuxSize)) {                                                      \
      IF(is32Bit) { eq(lowBits->at(0), 1); }                                                       \
      IF(is16Bit) { eq(lowBits->at(0) + lowBits->at(2), 1); }                                      \
      if (isRead) {                                                                                \
        for (size_t i = 0; i < 4; i++) {                                                           \
          if ((i & addrMask) != i) {                                                               \
            continue;                                                                              \
          }                                                                                        \
          IF(lowBits->at(i)) { highByte->set(loaded.bytes[i + 3 - addrMask]); }                    \
        }                                                                                          \
        NONDET {                                                                                   \
          highBit->setExact((highByte & 0x80) / 0x80);                                             \
          lowBits2->setExact((highByte & 0x7f) * 2);                                               \
        }                                                                                          \
        eqz(highBit*(1 - highBit));                                                                \
        eq(highByte, highBit * 0x80 + lowBits2 / 2);                                               \
        Val fillByte = signExt ? 255 * highBit : 0;                                                \
        U32Val extended = {0, 0, 0, 0};                                                            \
        for (size_t i = 0; i < count; i++) {                                                       \
          for (size_t j = 0; j < 4; j++) {                                                         \
            if ((j & addrMask) != j) {                                                             \
              continue;                                                                            \
            }                                                                                      \
            extended.bytes[i] = extended.bytes[i] + lowBits->at(j) * loaded.bytes[j + i];          \
          }                                                                                        \
        }                                                                                          \
        for (size_t i = count; i < 4; i++) {                                                       \
          extended.bytes[i] = fillByte;                                                            \
        }                                                                                          \
        buffer->set(extended);                                                                     \
        extended = buffer->get();                                                                  \
        XLOG("  fillByte = %4x, extended: %w", fillByte, extended);                                \
        IF(1 - rdZero->isZero()) {                                                                 \
          write->doWrite(cycle, kRegisterOffset - 32 * userMode + decoder->rd(), extended);        \
        }                                                                                          \
        IF(rdZero->isZero()) { write->doNOP(); }                                                   \
      } else {                                                                                     \
        highByte->setExact(0);                                                                     \
        highBit->setExact(0);                                                                      \
        lowBits2->setExact(0);                                                                     \
        U32Val writeVal;                                                                           \
        for (size_t i = 0; i < count; i++) {                                                       \
          for (size_t j = 0; j < 4; j++) {                                                         \
            if ((j & addrMask) != j) {                                                             \
              continue;                                                                            \
            }                                                                                      \
            writeVal.bytes[i + j] =                                                                \
                lowBits->at(j) * rs2.bytes[i] + (1 - lowBits->at(j)) * loaded.bytes[i + j];        \
          }                                                                                        \
        }                                                                                          \
        XLOG("  writeVal = %w", writeVal);                                                         \
        write->doWrite(cycle, addr, writeVal);                                                     \
      }                                                                                            \
      eq(decoder->opcode(), opc * 4 + 3);                                                          \
      if (f3 != -1) {                                                                              \
        eq(decoder->func3(), f3);                                                                  \
      }                                                                                            \
      if (f7 != -1) {                                                                              \
        eq(decoder->func7(), f7);                                                                  \
      }                                                                                            \
      immReg->set(decoder->imm##immFmt());                                                         \
    }                                                                                              \
  }
#include "zirgen/circuit/rv32im/v1/platform/rv32im.inl"
}

} // namespace zirgen::rv32im_v1
