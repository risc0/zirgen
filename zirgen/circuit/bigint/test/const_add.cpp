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

void makeConstAddTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  size_t const_bits = 16;
  auto inp = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  auto expected = builder.create<BigInt::DefOp>(loc, std::max(bits, const_bits) + 1, 1, true);

  // Construct constant
  mlir::Type fortysevensType = builder.getIntegerType(const_bits);
  auto fortysevensAttr = builder.getIntegerAttr(fortysevensType, 0x4747); // value 0x4747
  auto fortysevens = builder.create<BigInt::ConstOp>(loc, fortysevensAttr);

  auto result = builder.create<BigInt::AddOp>(loc, inp, fortysevens);
  auto diff = builder.create<BigInt::SubOp>(loc, result, expected);
  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

} // namespace

TEST_F(BibcTest, ConstAdd8) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("const_add_8", builder);
  makeConstAddTest(builder, func.getLoc(), 8);

  auto inputs = apints({"1", "4748"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}

TEST_F(BibcTest, ConstAdd16) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("const_add_16", builder);
  makeConstAddTest(builder, func.getLoc(), 16);

  auto inputs = apints({"2", "4749"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}

TEST_F(BibcTest, ConstAdd128) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("const_add_128", builder);
  makeConstAddTest(builder, func.getLoc(), 128);

  auto inputs = apints({"1", "4748"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}
