// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include <memory>

#include "zirgen/Dialect/BigInt/IR/BigInt.h"

using namespace mlir;

namespace zirgen::BigInt {

void makeIsOddTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeConstAddTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeConstAddAltTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeConstMulTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeAddTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeConstZeroTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeConstOneTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeConstTwoByteTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeSubTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeMulTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);
void makeReduceTest(mlir::OpBuilder builder, mlir::Location loc, size_t bits);

} // namespace zirgen::BigInt
