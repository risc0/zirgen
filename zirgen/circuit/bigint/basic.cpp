// Copyright 2025 RISC Zero, Inc.
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

#include "zirgen/circuit/bigint/basic.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"

using namespace mlir;

namespace zirgen::BigInt {

void genMul(mlir::OpBuilder& builder, mlir::Location loc, size_t bitwidth) {
  auto lhs = builder.create<BigInt::LoadOp>(loc, bitwidth, 11, 0);
  auto rhs = builder.create<BigInt::LoadOp>(loc, bitwidth, 12, 0);
  auto prod = builder.create<BigInt::MulOp>(loc, lhs, rhs);

  // Construct the constant 1
  mlir::Type oneType = builder.getIntegerType(8);
  auto oneAttr = builder.getIntegerAttr(oneType, 1); // value 1
  auto one = builder.create<BigInt::ConstOp>(loc, oneAttr);

  auto result = builder.create<BigInt::NondetQuotOp>(loc, prod, one);
  auto diff = builder.create<BigInt::SubOp>(loc, prod, result);
  builder.create<BigInt::EqualZeroOp>(loc, diff);

  builder.create<BigInt::StoreOp>(loc, result, 13, 0);
}

} // namespace zirgen::BigInt
