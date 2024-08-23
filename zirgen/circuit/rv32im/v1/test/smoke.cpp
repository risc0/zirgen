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

#include "zirgen/circuit/rv32im/v1/test/runner.h"

#include <gtest/gtest.h>

namespace zirgen::rv32im_v1 {

TEST(RISCV, Smoke) {
  std::map<uint32_t, uint32_t> image;
  image[0x1000] = 0x00000793; // r15 = 0
  image[0x1001] = 0x00100513; // r10 = 1
  image[0x1002] = 0x40000593; // r11 = 0x400
  image[0x1003] = 0x00000713; // r14 = 0
  image[0x1004] = 0x00000073; // ecall
  Runner runner(32 * 1024, image, 0x4000);
  runner.run();
  runner.done();
}

} // namespace zirgen::rv32im_v1
