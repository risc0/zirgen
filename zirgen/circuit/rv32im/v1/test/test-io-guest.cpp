// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include <cstdint>

#include "guest.h"

extern "C" void start() {
  uint32_t input[2];
  uint32_t len = sys_io(input, 2, 0, 0, 0);
  if (len != 8 || input[0] != 0x01020304 || input[1] != 0x05060708) {
    fail();
  }
  sys_halt();
}
