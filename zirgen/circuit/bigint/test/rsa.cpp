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

#include "zirgen/circuit/bigint/rsa.h"
#include "zirgen/circuit/bigint/test/bibc.h"

#include <gtest/gtest.h>

using namespace zirgen;
using namespace zirgen::BigInt::test;

TEST_F(BibcTest, RSA256) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("rsa_256", builder);
  BigInt::makeRSAChecker(builder, func.getLoc(), 256);
  lower();

  llvm::APInt N(64, 101);
  llvm::APInt S(64, 32766);
  llvm::APInt M(64, 53);
  EXPECT_EQ(M, BigInt::RSA(N, S));
  std::vector<llvm::APInt> inputs = {N, S, M};

  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}

TEST_F(BibcTest, RSA3072) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("rsa_3072", builder);
  BigInt::makeRSAChecker(builder, func.getLoc(), 3072);
  lower();

  llvm::APInt N(64, 22764235167642101);
  llvm::APInt S(64, 10116847215);
  llvm::APInt M(64, 14255570451702775);
  EXPECT_EQ(M, BigInt::RSA(N, S));
  std::vector<llvm::APInt> inputs = {N, S, M};

  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}
