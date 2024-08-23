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

#include <gtest/gtest.h>

namespace zirgen::rv32im_v1 {

TEST(KERNEL_ABI, Basic) {
  risc0::setLogLevel(5);

  std::map<uint32_t, uint32_t> image;
  auto kernel_file = risc0::loadFile("zirgen/circuit/rv32im/v1/test/test-kernel");
  uint32_t kernelEntryPoint = risc0::loadElf(kernel_file, image, 0x08000000 / 4, 0x0c000000 / 4);

  auto user_file = risc0::loadFile("zirgen/circuit/rv32im/v1/test/user-guest");
  uint32_t userEntryPoint = risc0::loadElf(user_file, image, 0x00000000 / 4, 0x08000000 / 4);
  image[0x0BFFFF00 / 4] = userEntryPoint;

  Runner runner(64 * 1024, image, kernelEntryPoint);
  runner.run();
  runner.done();
}

} // namespace zirgen::rv32im_v1
