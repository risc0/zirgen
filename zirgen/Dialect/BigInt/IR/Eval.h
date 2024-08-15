// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/compiler/zkp/poseidon2.h"

namespace zirgen::BigInt {

using BytePoly = std::vector<int32_t>;

struct EvalOutput {
  std::array<uint32_t, 4> z;
  std::vector<BytePoly> constantWitness;
  std::vector<BytePoly> publicWitness;
  std::vector<BytePoly> privateWitness;

  void print(llvm::raw_ostream& os) const;
};

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os, const EvalOutput& output) {
  output.print(os);
  return os;
}

BytePoly fromAPInt(llvm::APInt value, size_t coeffs);
Digest computeDigest(std::vector<BytePoly> witness, size_t groupSize = 3);

EvalOutput eval(mlir::func::FuncOp inFunc, llvm::ArrayRef<llvm::APInt> witnessValues);

} // namespace zirgen::BigInt
