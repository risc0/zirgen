// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/keccak2/cpp/run.h"

int main() {
  using namespace zirgen::keccak2;
  std::vector<KeccakState> inputs;
  KeccakState test;
  test.fill(0);
  inputs.push_back(test);
  runSegment(inputs);
}
