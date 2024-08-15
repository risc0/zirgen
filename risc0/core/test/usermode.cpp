// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include <unistd.h>

int main() {
  unsigned char buf[10];
  read(0, buf, 10);
  int tot = 0;
  for (size_t i = 0; i < 10; i++) {
    tot += buf[i];
  }
  if (tot == 45) {
    return 0;
  }
  return tot;
}
