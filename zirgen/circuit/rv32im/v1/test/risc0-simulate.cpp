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

#include "risc0/core/elf.h"
#include "risc0/core/log.h"
#include "risc0/core/util.h"
#include "zirgen/circuit/rv32im/v1/test/runner.h"

using namespace zirgen;
using namespace zirgen::rv32im_v1;

int main(int argc, char* argv[]) {
  risc0::setLogLevel(2);
  if (argc < 3) {
    LOG(1, "usage: risc0-simulate <elf> <max_cycles>");
    exit(1);
  }
  LOG(1, "File = " << argv[1] << ", cycles: " << argv[2]);
  try {
    auto file = risc0::loadFile(argv[1]);
    std::map<uint32_t, uint32_t> image;
    uint32_t entryPoint = risc0::loadElf(file, image);
    Runner runner(atoi(argv[2]), image, entryPoint);
    runner.run();
    if (!runner.done()) {
      throw std::runtime_error("Didn't get to ecall");
    }
  } catch (const std::runtime_error& err) {
    LOG(1, "Failed: " << err.what());
    exit(1);
  }
  return 0;
}
