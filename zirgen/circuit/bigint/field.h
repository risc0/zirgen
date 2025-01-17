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

#include <memory>

#include "zirgen/Dialect/BigInt/IR/BigInt.h"

using namespace mlir;

namespace zirgen::BigInt::field {

// Finite Field arithmetic
//
// These functions accelerate finite field arithmetic
//  - The `Mod` versions are for prime order fields
//  - The `FieldExt` versions are for simple extensions
//    - Every finite extension of a finite field is simple, so in a sense this covers every finite
//      field, but to use these functions you must represent the extension as the adjunction of a
//      primitive element to a prime order field, which is not always convenient (i.e. when you have
//      a tower of extensions)
//  - The `ExtFieldXXOne` version of multiply is for specifically the field extension with
//    irreducible polynomial `x^2 + 1` (i.e., extension by the square root of negative one)
//
// We do not use integer quotients in these functions, so minBits does not give us performance gains
// and we therefore do not require the prime to be full bitwidth, enabling simpler generalization
// (i.e., there's no need to make sure the bitwidth is minimal for your use case)

// Full programs, including I/O
void genModAdd(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth);
void genModInv(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth);
void genModMul(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth);
void genModSub(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth);
void genExtFieldAdd(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth, size_t degree);
void genExtFieldMul(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth, size_t degree);
void genExtFieldXXOneMul(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth);
void genExtFieldSub(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth, size_t degree);

// Prime field arithmetic (aka modular arithmetic)
Value modAdd(mlir::OpBuilder builder, mlir::Location loc, Value lhs, Value rhs, Value prime);
Value modInv(mlir::OpBuilder builder, mlir::Location loc, Value inp, Value prime);
Value modMul(mlir::OpBuilder builder, mlir::Location loc, Value lhs, Value rhs, Value prime);
Value modSub(mlir::OpBuilder builder, mlir::Location loc, Value lhs, Value rhs, Value prime);

// Extension field arithmetic
// Extension fields we use are most commonly degree 2
llvm::SmallVector<Value, 3> extAdd(mlir::OpBuilder builder,
                                   mlir::Location loc,
                                   llvm::SmallVector<Value, 3> lhs,
                                   llvm::SmallVector<Value, 3> rhs,
                                   Value prime);
llvm::SmallVector<Value, 3> extMul(mlir::OpBuilder builder,
                                   mlir::Location loc,
                                   llvm::SmallVector<Value, 3> lhs,
                                   llvm::SmallVector<Value, 3> rhs,
                                   llvm::SmallVector<Value, 3> monic_irred_poly,
                                   Value prime);
llvm::SmallVector<Value, 3> extXXOneMul(mlir::OpBuilder builder,
                                        mlir::Location loc,
                                        llvm::SmallVector<Value, 3> lhs,
                                        llvm::SmallVector<Value, 3> rhs,
                                        Value prime,
                                        Value primesqr);
llvm::SmallVector<Value, 3> extSub(mlir::OpBuilder builder,
                                   mlir::Location loc,
                                   llvm::SmallVector<Value, 3> lhs,
                                   llvm::SmallVector<Value, 3> rhs,
                                   Value prime);

} // namespace zirgen::BigInt::field
