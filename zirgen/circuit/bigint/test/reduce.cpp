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

#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/circuit/bigint/test/bibc.h"

#include <gtest/gtest.h>

using namespace zirgen;
using namespace zirgen::BigInt::test;

namespace {

void makeReduceTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  auto lhs = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto rhs = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto expected = builder.create<BigInt::DefOp>(loc, bits, 2, true);

  auto result = builder.create<BigInt::ReduceOp>(loc, lhs, rhs);
  auto diff = builder.create<BigInt::SubOp>(loc, result, expected);
  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

} // namespace

TEST_F(BibcTest, Reduce8) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("reduce_8", builder);
  makeReduceTest(builder, func.getLoc(), 8);
  lower();

  auto inputs = apints({"7", "2", "1"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}

TEST_F(BibcTest, Reduce128) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("reduce_128", builder);
  makeReduceTest(builder, func.getLoc(), 128);
  lower();

  auto inputs = apints({"8003", "4002", "4001"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}
