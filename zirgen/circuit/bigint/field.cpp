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

// Extension field operations

llvm::SmallVector<Value, 3> extAdd(mlir::OpBuilder builder,
                                   mlir::Location loc,
                                   llvm::SmallVector<Value, 3> lhs,
                                   llvm::SmallVector<Value, 3> rhs,
                                   Value prime) {
  auto deg = lhs.size();
  assert(rhs.size() == deg);
  llvm::SmallVector<Value, 3> result(deg);

  for (size_t i = 0; i < deg; i++) {
    auto sum = builder.create<BigInt::AddOp>(loc, lhs[i], rhs[i]);
    result[i] = builder.create<BigInt::ReduceOp>(loc, sum, prime);
  }
  return result;
}

// Deg2 extfield mul with irreducible polynomial x^2+1
// (ax+b)(cx+d) == acxx-ac(xx+1) + (ad+bc)x + bd == (ad+bc)x + bd-ac
// This is a more optimized algorithm specialized to the x^2+1 polynomial;
// you could also use the degree 2 extMul code for this, but it is generally slower
llvm::SmallVector<Value, 3> extXXOneMul(mlir::OpBuilder builder,
                                        mlir::Location loc,
                                        llvm::SmallVector<Value, 3> lhs,
                                        llvm::SmallVector<Value, 3> rhs,
                                        Value prime,
                                        Value primesqr) {
  assert(lhs.size() == 2);
  assert(rhs.size() == 2);
  llvm::SmallVector<Value, 3> result(2);

  auto ad = builder.create<BigInt::MulOp>(loc, lhs[1], rhs[0]);
  auto bc = builder.create<BigInt::MulOp>(loc, lhs[0], rhs[1]);
  result[1] = builder.create<BigInt::AddOp>(loc, ad, bc);
  result[1] = builder.create<BigInt::ReduceOp>(loc, result[1], prime);

  auto bd = builder.create<BigInt::MulOp>(loc, lhs[0], rhs[0]);
  auto ac = builder.create<BigInt::MulOp>(loc, lhs[1], rhs[1]);
  result[0] = builder.create<BigInt::SubOp>(loc, bd, ac);
  result[0] = builder.create<BigInt::AddOp>(loc, result[0], primesqr);
  result[0] = builder.create<BigInt::ReduceOp>(loc, result[0], prime);

  return result;
}

llvm::SmallVector<Value, 3> extMul(mlir::OpBuilder builder,
                                   mlir::Location loc,
                                   llvm::SmallVector<Value, 3> lhs,
                                   llvm::SmallVector<Value, 3> rhs,
                                   llvm::SmallVector<Value, 3> monic_irred_poly,
                                   Value prime) {
  // Here `monic_irred_poly` is the coefficients a_i such that x^n - sum_i a_i x^i = 0
  auto deg = lhs.size();
  // Note: The field is not an extension field if deg <= 1
  assert(deg > 1);
  assert(rhs.size() == deg);
  assert(monic_irred_poly.size() == deg);
  llvm::SmallVector<Value, 3> result(2 * deg - 1);
  llvm::SmallVector<bool, 2> first_write(2 * deg - 1, true);

  // Compute product of polynomials
  for (size_t i = 0; i < deg; i++) {
    for (size_t j = 0; j < deg; j++) {
      size_t idx = i + j;
      auto prod = builder.create<BigInt::MulOp>(loc, lhs[i], rhs[j]);
      auto reduced_prod = builder.create<BigInt::ReduceOp>(loc, prod, prime);
      if (first_write[idx]) {
        result[idx] = reduced_prod;
        first_write[idx] = false;
      } else {
        result[idx] = builder.create<BigInt::AddOp>(loc, result[idx], reduced_prod);
        result[idx] = builder.create<BigInt::ReduceOp>(loc, result[idx], prime);
      }
    }
  }
  // Reduce using the monic irred polynomial of the extension field
  for (size_t i = 2 * deg - 2; i >= deg; i--) {
    for (size_t j = 0; j < deg; j++) {
      auto prod = builder.create<BigInt::MulOp>(loc, result[i], monic_irred_poly[j]);
      result[i - deg + j] = builder.create<BigInt::AddOp>(loc, result[i - deg + j], prod);
      result[i - deg + j] = builder.create<BigInt::ReduceOp>(loc, result[i - deg + j], prime);
    }
    // No need to zero out result[i], it will just get dropped
  }
  // Result's degree is just `deg`, drop the coefficients beyond that
  result.truncate(deg);

  return result;
}

llvm::SmallVector<Value, 3> extSub(mlir::OpBuilder builder,
                                   mlir::Location loc,
                                   llvm::SmallVector<Value, 3> lhs,
                                   llvm::SmallVector<Value, 3> rhs,
                                   Value prime) {
  auto deg = lhs.size();
  assert(rhs.size() == deg);
  llvm::SmallVector<Value, 3> result(deg);

  for (size_t i = 0; i < deg; i++) {
    // auto diff = builder.create<BigInt::SubOp>(loc, lhs[i], rhs[i]);
    auto diff = builder.create<BigInt::SubOp>(loc, lhs[i], rhs[i]);
    // Add `prime` due to the same reason as in modSub
    auto diff_aug = builder.create<BigInt::AddOp>(loc, diff, prime);
    result[i] = builder.create<BigInt::ReduceOp>(loc, diff_aug, prime);
  }
  return result;
}

// Full programs, including I/O

// Finite Field arithmetic
//
// These functions accelerate finite field arithmetic
//  - The `Mod` versions are for prime order fields
//  - The `FieldExt` versions are for simple extensions
//    - Every finite extension of a finite field is simple, so in a sense this covers every finite
//      field, but to use these functions you must represent the extension as the adjunction of a
//      primitive element to a prime order field, which is not always convenient (i.e. when you have
//      a tower of extensions)
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

// Extension fields we use are most commonly degree 2
void genExtFieldAdd(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth, size_t degree) {
  assert(bitwidth % 128 == 0); // Bitwidth must be an even number of 128-bit chunks
  size_t chunkwidth = bitwidth / 128;
  llvm::SmallVector<Value, 3> lhs(degree);
  llvm::SmallVector<Value, 3> rhs(degree);
  for (size_t i = 0; i < degree; i++) {
    lhs[i] = builder.create<BigInt::LoadOp>(loc, bitwidth, 11, i * chunkwidth);
    rhs[i] = builder.create<BigInt::LoadOp>(loc, bitwidth, 12, i * chunkwidth);
  }
  auto prime = builder.create<BigInt::LoadOp>(loc, bitwidth, 13, 0);
  auto result = BigInt::field::extAdd(builder, loc, lhs, rhs, prime);
  for (size_t i = 0; i < degree; i++) {
    builder.create<BigInt::StoreOp>(loc, result[i], 14, i * chunkwidth);
  }
}

void genExtFieldMul(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth, size_t degree) {
  assert(bitwidth % 128 == 0); // Bitwidth must be an even number of 128-bit chunks
  size_t chunkwidth = bitwidth / 128;
  llvm::SmallVector<Value, 3> lhs(degree);
  llvm::SmallVector<Value, 3> rhs(degree);
  llvm::SmallVector<Value, 3> monic_irred_poly(degree);
  for (size_t i = 0; i < degree; i++) {
    lhs[i] = builder.create<BigInt::LoadOp>(loc, bitwidth, 11, i * chunkwidth);
    rhs[i] = builder.create<BigInt::LoadOp>(loc, bitwidth, 12, i * chunkwidth);
    monic_irred_poly[i] = builder.create<BigInt::LoadOp>(loc, bitwidth, 13, i * chunkwidth);
  }
  auto prime = builder.create<BigInt::LoadOp>(loc, bitwidth, 14, 0);
  auto result = BigInt::field::extMul(builder, loc, lhs, rhs, monic_irred_poly, prime);
  for (size_t i = 0; i < degree; i++) {
    builder.create<BigInt::StoreOp>(loc, result[i], 15, i * chunkwidth);
  }
}

void genExtFieldSub(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth, size_t degree) {
  assert(bitwidth % 128 == 0); // Bitwidth must be an even number of 128-bit chunks
  size_t chunkwidth = bitwidth / 128;
  llvm::SmallVector<Value, 3> lhs(degree);
  llvm::SmallVector<Value, 3> rhs(degree);
  for (size_t i = 0; i < degree; i++) {
    lhs[i] = builder.create<BigInt::LoadOp>(loc, bitwidth, 11, i * chunkwidth);
    rhs[i] = builder.create<BigInt::LoadOp>(loc, bitwidth, 12, i * chunkwidth);
  }
  auto prime = builder.create<BigInt::LoadOp>(loc, bitwidth, 13, 0);
  auto result = BigInt::field::extSub(builder, loc, lhs, rhs, prime);
  for (size_t i = 0; i < degree; i++) {
    builder.create<BigInt::StoreOp>(loc, result[i], 14, i * chunkwidth);
  }
}

void genExtFieldXXOneMul(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth) {
  assert(bitwidth % 128 == 0); // Bitwidth must be an even number of 128-bit chunks
  size_t chunkwidth = bitwidth / 128;
  llvm::SmallVector<Value, 3> lhs(2);
  llvm::SmallVector<Value, 3> rhs(2);
  for (size_t i = 0; i < 2; i++) {
    lhs[i] = builder.create<BigInt::LoadOp>(loc, bitwidth, 11, i * chunkwidth);
    rhs[i] = builder.create<BigInt::LoadOp>(loc, bitwidth, 12, i * chunkwidth);
  }
  auto prime = builder.create<BigInt::LoadOp>(loc, bitwidth, 13, 0);
  auto primesqr = builder.create<BigInt::LoadOp>(loc, 2 * bitwidth, 14, 0);
  auto result = BigInt::field::extXXOneMul(builder, loc, lhs, rhs, prime, primesqr);
  for (size_t i = 0; i < 2; i++) {
    builder.create<BigInt::StoreOp>(loc, result[i], 15, i * chunkwidth);
  }
}

} // namespace zirgen::BigInt::field
