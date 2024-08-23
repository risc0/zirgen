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
