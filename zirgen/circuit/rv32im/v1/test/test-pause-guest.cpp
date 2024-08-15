// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include <cstdint>

#include "guest.h"

extern "C" void start() {
  sys_log("1");
  sys_pause();
  sys_log("2");
  sys_halt();
}
