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

#include "zirgen/components/mux.h"
#include "zirgen/components/onehot.h"

namespace zirgen::recursion {

enum class OpType {
  MICRO,
  MACRO,
  POSEIDON2_LOAD,
  POSEIDON2_FULL,
  POSEIDON2_PARTIAL,
  POSEIDON2_STORE,
  CHECKED_BYTES,
  COUNT,
};

constexpr size_t OP_TYPE_COUNT = size_t(OpType::COUNT);

enum class MacroOpcode {
  NOP,
  WOM_INIT,
  WOM_FINI,
  BIT_AND_ELEM,
  BIT_OP_SHORTS,
  SHA_INIT,
  SHA_FINI,
  SHA_LOAD,
  SHA_MIX,
  SET_GLOBAL,
  COUNT,
};

constexpr size_t MACRO_OPCODE_COUNT = size_t(MacroOpcode::COUNT);

enum class MicroOpcode {
  CONST,
  ADD,
  SUB,
  MUL,
  INV,
  EQ,
  READ_IOP_HEADER,
  READ_IOP_BODY,
  MIX_RNG,
  SELECT,
  EXTRACT,
  COUNT,
};

constexpr size_t MICRO_OPCODE_COUNT = size_t(MicroOpcode::COUNT);

struct MicroInstImpl : public CompImpl<MicroInstImpl> {
  MicroInstImpl();

  Reg opcode;
  std::vector<Reg> operands;
};
using MicroInst = Comp<MicroInstImpl>;

struct MicroInstsImpl : public CompImpl<MicroInstsImpl> {
  MicroInstsImpl();

  std::vector<MicroInst> insts;
};
using MicroInsts = Comp<MicroInstsImpl>;

struct MacroInstImpl : public CompImpl<MacroInstImpl> {
  MacroInstImpl();

  OneHot<MACRO_OPCODE_COUNT> opcode;
  // Operands for the macro op. Each macro op has three operands.
  std::vector<Reg> operands;
};
using MacroInst = Comp<MacroInstImpl>;

struct Poseidon2MemInstImpl : CompImpl<Poseidon2MemInstImpl> {
  Poseidon2MemInstImpl();

  Reg doMont;
  // `keepState`: When loading, add new loaded values to the existing state
  Reg keepState;
  // `keepUpperState`: When loading, keep the final 8 values & overwrite others
  Reg keepUpperState;
  // When the following instruction is Poseidon2Full, set `prepFull` to 1 to run
  // the initial constants & linear layer as part of this instruction
  Reg prepFull;
  OneHot<3> group;
  std::vector<Reg> inputs;
};
using Poseidon2MemInst = Comp<Poseidon2MemInstImpl>;

struct Poseidon2FullInstImpl : CompImpl<Poseidon2FullInstImpl> {
  Poseidon2FullInstImpl();

  OneHot<4> cycle;
};
using Poseidon2FullInst = Comp<Poseidon2FullInstImpl>;

struct Poseidon2InstImpl : CompImpl<Poseidon2InstImpl> {};
using Poseidon2Inst = Comp<Poseidon2InstImpl>;

struct CheckedBytesInstImpl : CompImpl<CheckedBytesInstImpl> {
  CheckedBytesInstImpl();

  Reg evalPoint;
  Reg keepCoeffs;
  Reg keepUpperState;
  Reg prepFull;
};
using CheckedBytesInst = Comp<CheckedBytesInstImpl>;

struct CodeImpl : public CompImpl<CodeImpl> {
  CodeImpl();

  Reg writeAddr;
  OneHot<OP_TYPE_COUNT> select;
  Mux<MicroInsts,
      MacroInst,
      Poseidon2MemInst,
      Poseidon2FullInst,
      Poseidon2Inst,
      Poseidon2MemInst,
      CheckedBytesInst>
      inst;
};

using Code = Comp<CodeImpl>;

constexpr size_t kDigestShorts = kDigestWords * 2;
constexpr size_t kOutDigests = 2;
constexpr size_t kOutSize = kOutDigests * kDigestShorts;
constexpr size_t kCodeSize = 23;
constexpr size_t kDataSize = 128;
constexpr size_t kMixSize = 20;
constexpr size_t kAccumSize = 12;
constexpr size_t kRecursionPo2 = 18;
constexpr size_t kAllowedCodeMerkleDepth = 8; // allows for 2^8 allowed code roots.
constexpr size_t kNumRollup = 2;

} // namespace zirgen::recursion
