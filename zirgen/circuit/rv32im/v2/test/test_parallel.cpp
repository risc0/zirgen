// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include <iostream>

#include "zirgen/circuit/rv32im/v2/run/run.h"

using namespace zirgen::rv32im_v2;

const std::string kernelName = "zirgen/circuit/rv32im/v2/kernel/kernel";
const std::string progName = "zirgen/circuit/rv32im/v2/emu/test/guest";

int main() {
  size_t cycles = 20000;
  TestIoHandler io;
  io.push_u32(0, 5);

  // Load image
  auto image = MemoryImage::fromElfs(kernelName, progName);
  // Do execution
  auto segments = execute(image, io, cycles, cycles);
  // Do 'run' (preflight + expansion)
  for (const auto& segment : segments) {
    runSegment(segment);
  }
}
