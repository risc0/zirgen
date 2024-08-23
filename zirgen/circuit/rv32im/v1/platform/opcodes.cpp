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

#include "zirgen/circuit/rv32im/v1/platform/opcodes.h"

#include <iostream>
#include <vector>

namespace zirgen::rv32im_v1 {

namespace {

struct OpcodeTable {
  struct DecodeSwitch {
    bool isLeaf = true;
    OpcodeInfo leafData;
    std::vector<DecodeSwitch> lower;
  };

  void addOpcode(
      uint32_t major, uint32_t minor, const char* mnemonic, int opcode, int func3, int func7) {
    OpcodeInfo leafData = {mnemonic, major, minor};
    if (opcode == -1) {
      throw std::runtime_error("Opcode must be specified");
    }
    DecodeSwitch& stage1 = decode[opcode];
    if (func3 == -1) {
      stage1.leafData = leafData;
      return;
    }
    stage1.isLeaf = false;
    stage1.lower.resize(8);
    DecodeSwitch& stage2 = stage1.lower[func3];
    if (func7 == -1) {
      stage2.leafData = leafData;
      return;
    }
    stage2.isLeaf = false;
    stage2.lower.resize(128);
    DecodeSwitch& stage3 = stage2.lower[func7];
    stage3.leafData = leafData;
  }

  OpcodeTable() {
    decode.resize(32);
#define OPC(idx, mnemonic, opcode, func3, func7, ...)                                              \
  addOpcode(idx / 8, idx % 8, #mnemonic, opcode, func3, func7);
#define OPI(idx, mnemonic, opcode, func3, func7, ...)                                              \
  addOpcode(idx / 8, idx % 8, #mnemonic, opcode, func3, func7);
#define OPM(idx, mnemonic, opcode, func3, func7, ...)                                              \
  addOpcode(idx / 8, idx % 8, #mnemonic, opcode, func3, func7);
#define OPD(idx, mnemonic, opcode, func3, func7, ...)                                              \
  addOpcode(idx / 8, idx % 8, #mnemonic, opcode, func3, func7);
#include "zirgen/circuit/rv32im/v1/platform/rv32im.inl"
    addOpcode(MajorType::kECall, 0, "ecall", 0b11100, 0, 0);
  }

  OpcodeInfo resolve(uint32_t inst) {
    if ((inst & 3) != 3) {
      return OpcodeInfo();
    }
    uint32_t opcode = (inst & 0x7c) >> 2;
    DecodeSwitch& stage1 = decode[opcode];
    if (stage1.isLeaf) {
      return stage1.leafData;
    }
    uint32_t func3 = (inst & 0x7000) >> 12;
    DecodeSwitch& stage2 = stage1.lower[func3];
    if (stage2.isLeaf) {
      return stage2.leafData;
    }
    uint32_t func7 = inst >> 25;
    DecodeSwitch& stage3 = stage2.lower[func7];
    return stage3.leafData;
  }

  std::vector<DecodeSwitch> decode;
};

} // namespace

OpcodeInfo getOpcodeInfo(uint32_t inst) {
  static OpcodeTable table;
  return table.resolve(inst);
}

} // namespace zirgen::rv32im_v1
