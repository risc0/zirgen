// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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

private:
  size_t processChain(mlir::Value val, size_t offset);

  llvm::DenseMap<mlir::Operation*, size_t> mixPows;
  llvm::SmallVector<size_t> powsNeeded;
  llvm::DenseMap<size_t, size_t> mixPowIndex;
  llvm::DenseMap<mlir::Operation*, size_t> useCount;
};

} // namespace zirgen
