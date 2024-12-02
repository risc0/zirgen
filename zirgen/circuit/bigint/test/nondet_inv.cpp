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

void makeNondetInvTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  auto inp = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto prime = builder.create<BigInt::DefOp>(loc, bits, 1, true, bits - 1);
  auto expected = builder.create<BigInt::DefOp>(loc, bits, 2, true);

  // Construct constants
  mlir::Type oneType = builder.getIntegerType(1);    // a `1` is bitwidth 1
  auto oneAttr = builder.getIntegerAttr(oneType, 1); // value 1
  auto one = builder.create<BigInt::ConstOp>(loc, oneAttr);

  auto inv = builder.create<BigInt::NondetInvOp>(loc, inp, prime);
  auto prod = builder.create<BigInt::MulOp>(loc, inp, inv);
  auto reduced = builder.create<BigInt::ReduceOp>(loc, prod, prime);
  auto expect_zero = builder.create<BigInt::SubOp>(loc, reduced, one);
  builder.create<BigInt::EqualZeroOp>(loc, expect_zero);
  auto result_match = builder.create<BigInt::SubOp>(loc, inv, expected);
  builder.create<BigInt::EqualZeroOp>(loc, result_match);
}

} // namespace

TEST_F(BibcTest, NondetInv8) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("nondet_inv_8", builder);
  makeNondetInvTest(builder, func.getLoc(), 8);
  lower();

  auto inputs = apints({"4", "3", "1"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}

TEST_F(BibcTest, NondetInv128) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("nondet_inv_128", builder);
  makeNondetInvTest(builder, func.getLoc(), 128);
  lower();

  auto inputs = apints({"100E", "0BB9", "03D9"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}
