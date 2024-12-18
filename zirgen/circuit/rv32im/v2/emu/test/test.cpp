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

#include "zirgen/circuit/rv32im/v2/emu/exec.h"
#include "zirgen/circuit/rv32im/v2/emu/image.h"
#include "zirgen/circuit/rv32im/v2/emu/preflight.h"

using namespace zirgen::rv32im_v2;

int main() {
  std::string path = "zirgen/circuit/rv32im/v2/";
  auto image = MemoryImage::fromElfs(path + "kernel/kernel", path + "emu/test/guest");
  TestIoHandler io;
  io.push_u32(0, 1000);
  auto segments = execute(image, io, 1000000, 64 * 1024 * 1024);
  std::cout << "Made " << segments.size() << " segments\n";
  if (io.output.size() >= 4) {
    uint32_t result = io.pop_u32(1);
    std::cout << "Result = " << result << "\n";
  }
  auto ptrace = preflightSegment(segments[0], 1000000 + 2000);
  return 0;
}
