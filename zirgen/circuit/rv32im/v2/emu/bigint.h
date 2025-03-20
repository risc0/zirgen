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

#include "zirgen/Dialect/BigInt/Bytecode/decode.h"
#include "zirgen/Dialect/BigInt/Bytecode/file.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/IR/Eval.h"

#include <array>
#include <cstdint>

namespace zirgen::rv32im_v2 {

struct BigIntState {
  uint32_t isEcall;
  uint32_t mode;
  uint32_t pc;
  uint32_t polyOp;
  uint32_t coeff;
  std::array<uint32_t, 16> bytes{};
  uint32_t nextState;
};

struct BigIntInstruction {
  uint32_t polyOp;
  uint32_t memOp;
  uint32_t coeff;
  uint32_t reg;
  uint32_t offset;

  static BigIntInstruction decode(uint32_t insn) {
    return BigIntInstruction{
        .polyOp = insn >> 24 & 0x0f,
        .memOp = insn >> 28 & 0x0f,
        .coeff = insn >> 21 & 0x07,
        .reg = insn >> 16 & 0x1f,
        .offset = insn & 0xffff,
    };
  }
};

namespace {

template <typename Context> struct BigIntIO : public zirgen::BigInt::BigIntIO {
  Context& ctx;
  std::map<uint32_t, uint32_t>& polyWitness;

  BigIntIO(Context& ctx, std::map<uint32_t, uint32_t>& polyWitness)
      : ctx(ctx), polyWitness(polyWitness) {}

  llvm::APInt load(uint32_t arena, uint32_t offset, uint32_t count) override {
    uint32_t regVal = ctx.hostPeek(MACHINE_REGS_WORD + arena);
    uint32_t addr = regVal + offset * 16;
    uint32_t baseWord = addr / 4;
    std::vector<uint64_t> limbs64;
    for (size_t i = 0; i < count; i++) {
      std::array<uint32_t, 4> words;
      for (size_t j = 0; j < 4; j++) {
        words[j] = ctx.hostPeek(baseWord + i * 4 + j);
      }
      limbs64.push_back(uint64_t(words[0]) | ((uint64_t(words[1])) << 32));
      limbs64.push_back(uint64_t(words[2]) | ((uint64_t(words[3])) << 32));
    }
    llvm::APInt val(count * 128, limbs64);
    llvm::errs() << "Load, arena=" << arena << ", offset=" << offset << "\n";
    llvm::errs() << "  Addr = " << addr << "\n";
    llvm::errs() << "  ";
    val.print(llvm::errs(), false);
    llvm::errs() << "\n";
    return val;
  }

  void store(uint32_t arena, uint32_t offset, uint32_t count, llvm::APInt val) override {
    uint32_t regVal = ctx.hostPeek(MACHINE_REGS_WORD + arena);
    uint32_t addr = regVal + offset * 16;
    uint32_t baseWord = addr / 4;
    llvm::errs() << "Store, arena=" << arena << ", offset=" << offset << "\n";
    llvm::errs() << "  Addr = " << addr << "\n";
    llvm::errs() << "  ";
    val.print(llvm::errs(), false);
    llvm::errs() << "\n";
    val = val.zext(count * 128);
    for (size_t i = 0; i < count * 4; i++) {
      polyWitness[baseWord + i] = val.extractBitsAsZExtValue(32, i * 32);
    }
  }
};

} // namespace

struct BigInt {
  BigIntState state;
  std::map<uint32_t, uint32_t> polyWitness;

  template <typename Context> static void ecall(Context& ctx) {
    printf("BigInt::ecall\n");

    BigInt bigint;
    bigint.state.isEcall = 1;
    bigint.state.mode = ctx.load(MACHINE_REGS_WORD + REG_T0);
    bigint.state.pc = ctx.load(MACHINE_REGS_WORD + REG_T2) / 4 - 1;
    bigint.state.polyOp = 0;
    bigint.state.coeff = 0;
    bigint.state.nextState = STATE_BIGINT_STEP;
    ctx.bigintCycle(STATE_BIGINT_ECALL, STATE_BIGINT_STEP, bigint.state);

    uint32_t blobAddr = ctx.hostPeek(MACHINE_REGS_WORD + REG_A0) / 4;
    uint32_t bibcAddr = ctx.hostPeek(MACHINE_REGS_WORD + REG_T1) / 4;
    uint32_t bibcSize = ctx.hostPeek(blobAddr);

    std::vector<uint32_t> code;
    for (size_t i = 0; i < bibcSize; i++) {
      code.push_back(ctx.hostPeek(bibcAddr + i));
    }

    // Deserialize
    zirgen::BigInt::Bytecode::Program prog;
    zirgen::BigInt::Bytecode::read(prog, code.data(), code.size() * 4);

    // Build a module + func
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<zirgen::BigInt::BigIntDialect>();
    mlir::MLIRContext mlirContext(registry);
    mlirContext.loadAllAvailableDialects();
    auto loc = mlir::UnknownLoc::get(&mlirContext);
    auto module = mlir::ModuleOp::create(loc);
    auto func = zirgen::BigInt::Bytecode::decode(module, prog);

    // evaluate the program to compute the polyWitness
    BigIntIO io(ctx, bigint.polyWitness);
    zirgen::BigInt::eval(func, io, false);

    while (bigint.state.nextState == STATE_BIGINT_STEP) {
      bigint.step(ctx);
    }
  }

  template <typename Context> void step(Context& ctx) {
    state.pc += 1;
    uint32_t insn = ctx.load(state.pc);
    auto decoded = BigIntInstruction::decode(insn);
    uint32_t addr = ctx.load(MACHINE_REGS_WORD + decoded.reg) / 4 + decoded.offset * 4;

    printf("BigInt::step(0x%08x, %u, %u)\n", state.pc, decoded.polyOp, decoded.memOp);

    switch (decoded.memOp) {
    case 0: { // read
      for (size_t i = 0; i < 4; i++) {
        uint32_t word = ctx.load(addr + i);
        for (size_t j = 0; j < 4; j++) {
          state.bytes[i * 4 + j] = (word >> (j * 8)) & 0xff;
        }
      }
    } break;
    case 1: { // write
      for (size_t i = 0; i < 4; i++) {
        uint32_t word = polyWitness[addr + i];
        for (size_t j = 0; j < 4; j++) {
          uint32_t byte = (word >> (j * 8)) & 0xff;
          state.bytes[i * 4 + j] = byte;
        }
        ctx.store(addr + i, word);
      }
    } break;
    case 2: { // check
      for (size_t i = 0; i < 4; i++) {
        uint32_t word = polyWitness[addr + i];
        for (size_t j = 0; j < 4; j++) {
          uint32_t byte = (word >> (j * 8)) & 0xff;
          state.bytes[i * 4 + j] = byte;
        }
      }
    } break;
    }

    bool isLast = !state.isEcall && !decoded.polyOp;
    if (isLast) {
      state.nextState = STATE_DECODE;
    } else {
      state.nextState = STATE_BIGINT_STEP;
    }

    state.isEcall = 0;
    state.polyOp = decoded.polyOp;
    state.coeff = decoded.coeff;

    ctx.bigintCycle(STATE_BIGINT_STEP, state.nextState, state);
  }
};

} // namespace zirgen::rv32im_v2
