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

#include "zirgen/circuit/rv32im/v2/platform/constants.h"

#include <array>
#include <cstdint>

namespace zirgen::rv32im_v2 {

struct BigIntState {
  uint32_t pc;
  // uint32_t polyOp;
  std::array<uint32_t, 16> bytes{};
  // std::array<uint32_t, 4> poly;
  // std::array<uint32_t, 4> term;
  // std::array<uint32_t, 4> tot;
  uint32_t nextState;
};

struct BigIntInstruction {
  uint32_t polyOp;
  uint32_t memOp;
  int coeff;
  uint32_t reg;
  uint32_t offset;

  static BigIntInstruction decode(uint32_t insn) {
    return BigIntInstruction{
        .memOp = insn >> 28 & 0x0f,
        .polyOp = insn >> 24 & 0x0f,
        .coeff = int(insn >> 21 & 0x07) - 4,
        .reg = insn >> 16 & 0x1f,
        .offset = insn & 0xffff,
    };
  }
};

struct BigInt {
  template <typename Context> static void ecall(Context& ctx) {
    BigIntState bigint;
    bigint.pc = ctx.load(MACHINE_REGS_WORD + REG_T2) / 4 - 1;
    bigint.nextState = STATE_BIGINT_STEP;
    ctx.bigintCycle(STATE_BIGINT_ECALL, STATE_BIGINT_STEP, bigint);

    bigint.pc += 1;
    bigint.nextState = STATE_DECODE;
    uint32_t insn = ctx.load(bigint.pc);
    auto decoded = BigIntInstruction::decode(insn);

    uint32_t addr = ctx.load(MACHINE_REGS_WORD + decoded.reg) / 4 + decoded.offset * 4;
    for (size_t i = 0; i < 4; i++) {
      ctx.store(addr + i, 0);
    }

    ctx.bigintCycle(STATE_BIGINT_STEP, STATE_DECODE, bigint);
  }
};

} // namespace zirgen::rv32im_v2
