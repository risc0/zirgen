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

// Disable these tests to prevent dependencies on old RISC0 repo
#if 0

#include <gtest/gtest.h>

#include "risc0/zkp/prove/merkle.h"
#include "zirgen/circuit/verify/merkle.h"

using namespace zirgen;
using namespace zirgen::verify;

TEST(verify, merkle) {
  size_t rows = 1024;
  size_t cols = 30;
  size_t queries = 10;

  // Make the merkle tree + queries
  auto leavesAccel = risc0::AccelSlice<risc0::Fp>::allocate(rows * cols);
  {
    risc0::AccelWriteOnlyLock lock(leavesAccel);
    for (size_t i = 0; i < rows * cols; i++) {
      lock[i] = static_cast<int>(i);
    }
  }
  risc0::MerkleTreeProver prover(leavesAccel, rows, cols, queries);
  risc0::WriteIOP wiop;
  prover.commit(wiop);
  for (size_t i = 0; i < queries; i++) {
    size_t idx = wiop.generate() % rows;
    auto col = prover.prove(wiop, idx);
  }
  auto proof = wiop.getProof();

  Module module;
  module.addFunc<1>("check_merkle", {ioparg()}, [&](ReadIopVal iop) {
    auto verifier = MerkleTreeVerifier(iop, rows, cols, queries);
    for (size_t i = 0; i < 5; i++) {
      Val x = iop.rngBits(10);
      auto vals = verifier.verify(iop, x);
    }
  });
  module.optimize();
  auto func = module.getModule().lookupSymbol<mlir::func::FuncOp>("check_merkle");
  Interpreter interp;
  risc0::ReadIopVal riop(proof.data(), proof.size());
  interp.setIOP(func.getArgument(0), &riop);
  // interp.setDebug(true);
  // module.dump();
  if (failed(  interp.runBlock(func.front())))
    throw(std::runtime_error("Unable to evaluate block"));
}

#endif
