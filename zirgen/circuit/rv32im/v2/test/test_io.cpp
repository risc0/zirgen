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

#include <iostream>

#include "zirgen/circuit/rv32im/v2/platform/constants.h"
#include "zirgen/circuit/rv32im/v2/run/run.h"

using namespace zirgen::rv32im_v2;

const std::string kernelName = "zirgen/circuit/rv32im/v2/test/test_io_kernel";

// Allows reads of any size, fill with a pattern to check in kernel
struct RandomReadSizeHandler : public HostIoHandler {
  uint32_t write(uint32_t fd, const uint8_t* data, uint32_t len) override { return len; }
  uint32_t read(uint32_t fd, uint8_t* data, uint32_t len) override {
    std::cout << "DOING READ OF SIZE " << len << "\n";
    for (size_t i = 0; i < len; i++) {
      data[i] = i;
    }
    return len;
  }
};

int main() {
  size_t cycles = 100000;
  RandomReadSizeHandler io;

  // Load image
  auto image = MemoryImage::fromRawElf(kernelName);
  // Do executions
  auto segments = execute(image, io, cycles, cycles);
  // Do 'run' (preflight + expansion)
  for (const auto& segment : segments) {
    std::cout << "HEY, doing a segment!\n";
    runSegment(segment, cycles + 1000);
  }
  std::cout << "What a fine day\n";
}
