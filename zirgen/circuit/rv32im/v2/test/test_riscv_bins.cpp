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

#include <iostream>

#include "zirgen/circuit/rv32im/v2/platform/constants.h"
#include "zirgen/circuit/rv32im/v2/run/run.h"

using namespace zirgen::rv32im_v2;

void runOneTest(const std::string& name) {
  std::string kernel = "zirgen/circuit/rv32im/shared/test/" + name;
  size_t cycles = 10000;

  TestIoHandler io;

  // Load image
  auto image = MemoryImage::fromRawElf(kernel);
  // Do executions
  auto segments = execute(image, io, cycles, cycles);
  // Do 'run' (preflight + expansion)
  for (const auto& segment : segments) {
    runSegment(segment, cycles);
  }
}

std::vector<std::string> names = {
    "add",  "addi", "and",  "andi",   "auipc", "beq",   "bge",  "bgeu", "blt",  "bltu",
    "bne",  "div",  "divu", "jal",    "jalr",  "lb",    "lbu",  "lh",   "lhu",  "lui",
    "lw",   "mul",  "mulh", "mulhsu", "mulhu", "or",    "ori",  "rem",  "remu", "sb",
    "sh",   "sll",  "slli", "slt",    "slti",  "sltiu", "sltu", "sra",  "srai", "srl",
    "srli", "sub",  "sw",   "test",   "xor",   "xori"};

const std::string kernelName = "zirgen/circuit/rv32im/shared/test/rem";

int main() {
  for (const std::string& name : names) {
    runOneTest(name);
  }
}