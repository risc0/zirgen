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

#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace zirgen {

class BogoCycleAnalysis {
public:
  // Return the number of bogo-cycles that the given operation uses.
  // This is a rough indication of how much different operations cost,
  // relative to each other.  A bogo-cycle does not have any absolute
  // meaning; it only has meaning relative to other bogo-cycle counts.
  //
  // This is not deterministic so should not be depended upon when
  // generating code.
  double getBogoCycles(mlir::Operation* op);

  // Print out statistics for the given operation and anything inside of it
  // if the op-stqts
  void printStatsIfRequired(mlir::Operation* topOp, llvm::raw_ostream& os);

private:
  template <typename OpT, size_t K, typename FpT> double getOrCalcBogoCycles();

  // Bogo cycles measured for various operations
  std::map<std::pair</*opName=*/mlir::StringLiteral, /*fieldK=*/size_t>, double> bogoCycles;
};

void registerOpStatsCLOptions();

} // namespace zirgen
