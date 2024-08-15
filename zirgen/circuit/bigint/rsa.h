// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/APInt.h"

namespace zirgen::BigInt {

void makeRSA(mlir::OpBuilder builder, mlir::Location loc, size_t bits);

llvm::APInt RSA(llvm::APInt N, llvm::APInt S);

} // namespace zirgen::BigInt
