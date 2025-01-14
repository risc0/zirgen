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

#include "risc0/core/log.h"
#include "zirgen/circuit/rv32im/v2/platform/constants.h"
#include "zirgen/circuit/rv32im/v2/run/run.h"

using namespace zirgen::rv32im_v2;

int main(int argc, char* argv[]) {
  risc0::setLogLevel(2);
  if (argc < 2) {
    LOG(1, "usage: risc0-simulate <elf>");
    exit(1);
  }

  LOG(1, "File = " << argv[1]);
  try {
    size_t cycles = 10000;

    TestIoHandler io;

    // Load image
    auto image = MemoryImage::fromRawElf(argv[1]);
    // Do executions
    auto segments = execute(image, io, cycles, cycles);
    // Do 'run' (preflight + expansion)
    for (const auto& segment : segments) {
      runSegment(segment, cycles);
    }
  } catch (const std::runtime_error& err) {
    LOG(1, "Failed: " << err.what());
    exit(1);
  }
  return 0;
}
