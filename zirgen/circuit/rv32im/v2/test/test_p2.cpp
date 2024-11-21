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

const std::string kernelName = "zirgen/circuit/rv32im/v2/test/test_p2_kernel";

int main() {
  size_t cycles = 100000;
  TestIoHandler io;

  auto entry = 0x10000;
  auto pc = entry / 4;

  auto image = MemoryImage::fromWords({
      {pc + 0, 0x1234b337}, // lui x6, 0x0001234b
      {pc + 1, 0xf387e3b7}, // lui x7, 0x000f387e
      {pc + 2, 0x007302b3}, // add x5, x6, x7
      {pc + 3, 0x000045b7}, // lui x11, 0x00000004
      {pc + 4, 0x00000073}, // ecall
      {SUSPEND_PC_WORD, entry},
      {SUSPEND_MODE_WORD, 1},
  });

  std::cout << image.getDigest(0x400100) << std::endl;
  std::cout << image.getDigest(1) << std::endl;

  // Load image
  // auto image = MemoryImage::fromRawElf(kernelName);
  // Do executions
  auto segments = execute(image, io, cycles, cycles);
  // Do 'run' (preflight + expansion)
  for (const auto& segment : segments) {
    runSegment(segment);
  }
}
