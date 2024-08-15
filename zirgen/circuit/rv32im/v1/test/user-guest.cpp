// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include <unistd.h>

int main() {
  uint32_t x = 0;
  uint32_t tot = 0;
  read(0, &x, sizeof(uint32_t));
  for (uint32_t i = 0; i < x; i++) {
    for (uint32_t j = 0; j < x; j++) {
      tot += i * j % 17;
      tot %= 107;
    }
  }
  write(1, &tot, sizeof(uint32_t));
  return 0;
}
