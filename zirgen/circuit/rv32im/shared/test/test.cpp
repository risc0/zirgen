// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "risc0/core/elf.h"
#include "risc0/core/util.h"
#include "zirgen/circuit/rv32im/shared/rv32im.h"

#include <assert.h>
#include <iostream>

using namespace risc0;
using namespace zirgen;

void runTest(const std::string& name) {
  BaseContext context;
  auto elfBytes = loadFile("zirgen/circuit/rv32im/shared/test/" + name);
  context.pc = loadElf(elfBytes, context.memory);
  RV32Emulator<BaseContext> emu(context);
  std::cout << "Running test on " << name << "\n";
  emu.run(5000);
  if (!context.isDone()) {
    std::cout << "Ran for more than 5000 cycles\n";
    throw std::runtime_error("BAD\n");
  }
}

int main() {
  runTest("add");
  runTest("sub");
  runTest("xor");
  runTest("or");
  runTest("and");
  runTest("slt");
  runTest("sltu");
  runTest("addi");
  runTest("xori");
  runTest("ori");
  runTest("andi");
  runTest("slti");
  runTest("sltiu");
  runTest("beq");
  runTest("bne");
  runTest("blt");
  runTest("bge");
  runTest("bltu");
  runTest("bgeu");
  runTest("jal");
  runTest("jalr");
  runTest("lui");
  runTest("auipc");
  runTest("sll");
  runTest("slli");
  runTest("mul");
  runTest("mulh");
  runTest("mulhsu");
  runTest("mulhu");
  runTest("srl");
  runTest("sra");
  runTest("srli");
  runTest("srai");
  runTest("div");
  runTest("divu");
  runTest("rem");
  runTest("remu");
  runTest("lb");
  runTest("lh");
  runTest("lw");
  runTest("lbu");
  runTest("lhu");
  runTest("sb");
  runTest("sh");
  runTest("sw");
}
