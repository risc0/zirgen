// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
