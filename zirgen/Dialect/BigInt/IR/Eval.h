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

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/compiler/zkp/poseidon2.h"

namespace zirgen::BigInt {

using BytePoly = std::vector<int32_t>;

struct EvalOutput {
  // This extension element is not in Montgomery form
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
