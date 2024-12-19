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

#include <memory>

#include "zirgen/Dialect/BigInt/IR/BigInt.h"

using namespace mlir;

namespace zirgen::BigInt::field {

// Finite Field arithmetic
//
// These functions accelerate finite field arithmetic
//  - The `Mod` versions are for prime order fields
//  - Versions for finite extensions of prime fields are planned as future work
//
// We do not use integer quotients in these functions, so minBits does not give us performance gains
// and we therefore do not require the prime to be full bitwidth, enabling simpler generalization
// (i.e., there's no need to make sure the bitwidth is minimal for your use case)

// Full programs, including I/O
void genModAdd(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth);
void genModInv(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth);
void genModMul(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth);
void genModSub(mlir::OpBuilder builder, mlir::Location loc, size_t bitwidth);

// Prime field arithmetic (aka modular arithmetic)
Value modAdd(mlir::OpBuilder builder, mlir::Location loc, Value lhs, Value rhs, Value prime);
Value modInv(mlir::OpBuilder builder, mlir::Location loc, Value inp, Value prime);
Value modMul(mlir::OpBuilder builder, mlir::Location loc, Value lhs, Value rhs, Value prime);
Value modSub(mlir::OpBuilder builder, mlir::Location loc, Value lhs, Value rhs, Value prime);

} // namespace zirgen::BigInt::field
