// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/circuit/recursion/top.h"

namespace zirgen::recursion {

struct Runner {
  using Polynomial = llvm::SmallVector<uint64_t, 4>;

  struct RecursionExternHandler;

  Runner();

  // Setup
  void setup(llvm::ArrayRef<uint32_t> code, llvm::ArrayRef<uint32_t> proof);
  void setRepro(llvm::ArrayRef<uint32_t> toRepro);
  // Run all the stages
  void run();
  // Clear things when done
  void done();

  void runStage(size_t stage);
  void setMix();

  Module module;
  std::unique_ptr<WomExternHandler> handler;
  size_t cycles;
  std::vector<Polynomial> code;
  std::vector<Polynomial> out;
  std::vector<Polynomial> data;
  std::vector<Polynomial> mix;
  std::vector<Polynomial> accum;
  std::vector<llvm::MutableArrayRef<Polynomial>> args;
};

} // namespace zirgen::recursion
