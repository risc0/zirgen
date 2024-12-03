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
    auto result = builder.create<BigInt::ReduceOp>(loc, diff, prime);
    return result;
}

// Extension field operations

llvm::SmallVector<Value, 2> extAdd(mlir::OpBuilder builder, mlir::Location loc, llvm::SmallVector<Value, 2> lhs, llvm::SmallVector<Value, 2> rhs, Value prime) {
    auto deg = lhs.size();
    assert(rhs.size() == deg);
    llvm::SmallVector<Value, 2> result(deg);

    for (size_t i = 0; i < deg; i++) {
        auto sum = builder.create<BigInt::AddOp>(loc, lhs[i], rhs[i]);
        result[i] = builder.create<BigInt::ReduceOp>(loc, sum, prime);
    }
    return result;
}

llvm::SmallVector<Value, 2> extMul(mlir::OpBuilder builder, mlir::Location loc, llvm::SmallVector<Value, 2> lhs, llvm::SmallVector<Value, 2> rhs, Value prime, llvm::SmallVector<Value, 2> monic_irred_poly) {
    // TODO: Annoying to have a SmallVector output that needs to be deg - 1 bigger than the inputs; I think that means all should be 3...
    // TODO: We could have a simplified version for nth roots x^n - a
    // Here `monic_irred_poly` is the coefficients a_i such that x^n - sum_i a_i x^i = 0
    auto deg = lhs.size();
    // Note: The field is not an extension field if deg <= 1
    assert(deg > 1);
    assert(rhs.size() == deg);
    assert(monic_irred_poly.size() == deg);
    llvm::SmallVector<Value, 2> result(2 * deg - 1);
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
        auto sum = builder.create<BigInt::AddOp>(loc, lhs[i], rhs[i]);
        result[i] = builder.create<BigInt::ReduceOp>(loc, sum, prime);
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

llvm::SmallVector<Value, 2> extSub(mlir::OpBuilder builder, mlir::Location loc, llvm::SmallVector<Value, 2> lhs, llvm::SmallVector<Value, 2> rhs, Value prime) {
    auto deg = lhs.size();
    assert(rhs.size() == deg);
    llvm::SmallVector<Value, 2> result(deg);

    for (size_t i = 0; i < deg; i++) {
        // auto diff = builder.create<BigInt::SubOp>(loc, lhs[i], rhs[i]);
        auto diff = builder.create<BigInt::SubOp>(loc, lhs[i], rhs[i]);
        result[i] = builder.create<BigInt::ReduceOp>(loc, diff, prime);
    }
    return result;
}



} // namespace zirgen::BigInt::field
