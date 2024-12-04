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

void makeConstOneTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits) {
  auto expected = builder.create<BigInt::DefOp>(loc, bits, 0, true);
  mlir::Type oneType = builder.getIntegerType(8, false); // unsigned 8 bit
  auto oneAttr = builder.getIntegerAttr(oneType, 1);     // value 1
  auto one = builder.create<BigInt::ConstOp>(loc, oneAttr);
  auto diff = builder.create<BigInt::SubOp>(loc, one, expected);
  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

} // namespace

TEST_F(BibcTest, ConstOne8) {
  mlir::OpBuilder builder(ctx);
  auto func = makeFunc("const_one_8", builder);
  makeConstOneTest(builder, func.getLoc(), 8);

  auto inputs = apints({"1"});
  ZType a, b;
  AB(func, inputs, a, b);
  EXPECT_EQ(a, b);
}
