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

#include "zirgen/circuit/rv32im/v2/run/run.h"

using namespace zirgen::rv32im_v2;

const std::string kernelName = "zirgen/circuit/rv32im/v2/kernel/kernel";
const std::string progName = "zirgen/circuit/rv32im/v2/emu/test/guest";

int main() {
  try {
    size_t threshold = 16000;
    size_t segmentSize = 16384;
    size_t maximum = 100000;
    TestIoHandler io;
    io.push_u32(0, 5);

    // Load image
    auto image = MemoryImage::fromElfs(kernelName, progName);
    // Do execution
    auto segments = execute(image, io, threshold, maximum);
    // Do 'run' (preflight + expansion)
    for (const auto& segment : segments) {
      runSegment(segment, segmentSize);
    }
  } catch (std::exception& ex) {
    printf("Exception: %s\n", ex.what());
  }
}
