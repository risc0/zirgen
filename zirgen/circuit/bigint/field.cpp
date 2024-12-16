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

#include "zirgen/circuit/bigint/field.h"

namespace zirgen::BigInt::field {

// Prime field operations

Value modAdd(mlir::OpBuilder builder, mlir::Location loc, Value lhs, Value rhs, Value prime) {
  auto sum = builder.create<BigInt::AddOp>(loc, lhs, rhs);
  auto result = builder.create<BigInt::ReduceOp>(loc, sum, prime);
  return result;
}

Value modInv(mlir::OpBuilder builder, mlir::Location loc, Value inp, Value prime) {
  return builder.create<BigInt::InvOp>(loc, inp, prime);
}

Value modMul(mlir::OpBuilder builder, mlir::Location loc, Value lhs, Value rhs, Value prime) {
  auto prod = builder.create<BigInt::MulOp>(loc, lhs, rhs);
  auto result = builder.create<BigInt::ReduceOp>(loc, prod, prime);
  return result;
}

Value modSub(mlir::OpBuilder builder, mlir::Location loc, Value lhs, Value rhs, Value prime) {
  auto diff = builder.create<BigInt::SubOp>(loc, lhs, rhs);
  // True statements can fail to prove if a ReduceOp is given negative inputs; thus, add `prime`
  // to ensure all normalized inputs can produce an answer
  auto diff_aug = builder.create<BigInt::AddOp>(loc, diff, prime);
  auto result = builder.create<BigInt::ReduceOp>(loc, diff_aug, prime);
  return result;
}

// Full programs, including I/O

// Finite Field arithmetic
//
// These functions accelerate finite field arithmetic
//  - The `Mod` versions are for prime order fields
//  - Versions for finite extensions of prime fields are planned as future work
//
// We do not use integer quotients in these functions, so minBits does not give us performance gains
// and we therefore do not require the prime to be full bitwidth, enabling simpler generalization
// (i.e., there's no need to make sure the bitwidth is minimal for your use case)

void genModAdd(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth) {
  auto lhs = builder.create<BigInt::LoadOp>(loc, bitwidth, 11, 0);
  auto rhs = builder.create<BigInt::LoadOp>(loc, bitwidth, 12, 0);
  auto prime = builder.create<BigInt::LoadOp>(loc, bitwidth, 13, 0);
  auto result = BigInt::field::modAdd(builder, loc, lhs, rhs, prime);
  builder.create<BigInt::StoreOp>(loc, result, 14, 0);
}

void genModInv(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth) {
  auto inp = builder.create<BigInt::LoadOp>(loc, bitwidth, 11, 0);
  auto prime = builder.create<BigInt::LoadOp>(loc, bitwidth, 12, 0);
  auto result = BigInt::field::modInv(builder, loc, inp, prime);
  builder.create<BigInt::StoreOp>(loc, result, 13, 0);
}

void genModMul(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth) {
  auto lhs = builder.create<BigInt::LoadOp>(loc, bitwidth, 11, 0);
  auto rhs = builder.create<BigInt::LoadOp>(loc, bitwidth, 12, 0);
  auto prime = builder.create<BigInt::LoadOp>(loc, bitwidth, 13, 0);
  auto result = BigInt::field::modMul(builder, loc, lhs, rhs, prime);
  builder.create<BigInt::StoreOp>(loc, result, 14, 0);
}

void genModSub(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth) {
  auto lhs = builder.create<BigInt::LoadOp>(loc, bitwidth, 11, 0);
  auto rhs = builder.create<BigInt::LoadOp>(loc, bitwidth, 12, 0);
  auto prime = builder.create<BigInt::LoadOp>(loc, bitwidth, 13, 0);
  auto result = BigInt::field::modSub(builder, loc, lhs, rhs, prime);
  builder.create<BigInt::StoreOp>(loc, result, 14, 0);
}

} // namespace zirgen::BigInt::field
