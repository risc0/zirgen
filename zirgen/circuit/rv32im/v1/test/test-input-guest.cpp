// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include <cstdint>

#include "guest.h"

// result of `shaHash("Hello")`
constexpr uint32_t expected[8] = {0xb38d5f18, //
                                  0x25fe7122,
                                  0xfca661f5,
                                  0x262e8b93,
                                  0x30ec0643,
                                  0x8051da4e,
                                  0x4876d107,
                                  0x69193826};

extern "C" void start() {
  for (int i = 0; i < 8; i++) {
    if (sys_input(i) != expected[i]) {
      fail();
    }
  }
  sys_halt();
}
