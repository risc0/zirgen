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

#include "mlir/IR/Operation.h"
#include "zirgen/Dialect/Zll/IR/IR.h"

namespace zirgen {

// Analyses the powers of poly_mix needed to calculate the validity
// polynomial.  Since these are the same for every cycle, we can
// precompute them.
class MixPowAnalysis {
public:
  // Analyzes the polynomial mix powers needed for the contents of the
  // given function, which should be a function which calculates the
  // value of the validity polynomial.
  MixPowAnalysis(mlir::Operation* funcOp);

  // Returns the mix power that should be multipled for the given operation
  size_t getMixPow(mlir::Operation* op) {
    assert((llvm::isa<Zll::AndEqzOp, Zll::AndCondOp>(op)));
    return mixPows.at(op);
  }

  // Returns the index into the array returnd by getPowersNeeded() of
  // the mix power needed by the given operation.
  size_t getMixPowIndex(mlir::Operation* op) { return mixPowIndex.at(getMixPow(op)); }

  // Returns an array of all the powers of poly_mix that are needed
  // for the analysed function.
  llvm::ArrayRef<size_t> getPowersNeeded() { return powsNeeded; }

  // Returns any called functions encountered
  llvm::ArrayRef<mlir::func::FuncOp> getCalledFuncs() { return calledFuncs; }

private:
  size_t processChain(mlir::Value val,
                      size_t offset,
                      llvm::SmallVector<mlir::func::CallOp> callStack = {});

  llvm::DenseMap<mlir::Operation*, size_t> mixPows;
  llvm::SmallVector<size_t> powsNeeded;
  llvm::DenseMap<size_t, size_t> mixPowIndex;
  llvm::DenseMap<mlir::Operation*, size_t> useCount;
  llvm::SmallVector<mlir::func::FuncOp> calledFuncs;
};

} // namespace zirgen
