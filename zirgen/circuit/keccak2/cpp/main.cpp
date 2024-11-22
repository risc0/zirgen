// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/keccak2/cpp/run.h"
#include "zirgen/compiler/zkp/sha256.h"

int main() {
  zirgen::Digest digest = zirgen::impl::initState();
  std::vector<uint32_t> data(16, 0);
  data[0] = 0;
  zirgen::impl::compress(digest, data.data());
  using namespace zirgen::keccak2;
  std::vector<KeccakState> inputs;
  KeccakState test;
  test.fill(0);
  inputs.push_back(test);
  runSegment(inputs);
}
