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

// Prime field arithmetic (aka modular arithmetic)
Value modAdd(mlir::OpBuilder builder, mlir::Location loc, Value lhs, Value rhs, Value prime);
Value modInv(mlir::OpBuilder builder, mlir::Location loc, Value inp, Value prime);
Value modMul(mlir::OpBuilder builder, mlir::Location loc, Value lhs, Value rhs, Value prime);
Value modSub(mlir::OpBuilder builder, mlir::Location loc, Value lhs, Value rhs, Value prime);

// Extension field arithmetic
// Extension fields we use are most commonly degree 2
// TODO: ^ Hence the use of 2 in the SmallVectors ... but is this true?
llvm::SmallVector<Value, 2> extAdd(mlir::OpBuilder builder, mlir::Location loc, llvm::SmallVector<Value, 2> lhs, llvm::SmallVector<Value, 2> rhs, Value prime);
llvm::SmallVector<Value, 2> extMul(mlir::OpBuilder builder, mlir::Location loc, llvm::SmallVector<Value, 2> lhs, llvm::SmallVector<Value, 2> rhs, Value prime, llvm::SmallVector<Value, 2> monic_irred_poly);
llvm::SmallVector<Value, 2> extSub(mlir::OpBuilder builder, mlir::Location loc, llvm::SmallVector<Value, 2> lhs, llvm::SmallVector<Value, 2> rhs, Value prime);

} // namespace zirgen::BigInt::field
