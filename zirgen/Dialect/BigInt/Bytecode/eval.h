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

// Do not include any LLVM or MLIR headers!
// This is meant to be a standalone reimplementation of the canonical
// eval function implemented in zirgen/Dialect/BigInt/IR/Eval.h, without
// depending on MLIR, so it can be ported to rust and run in the host.

#include <array>
#include <cstdint>
#include <vector>
#include "zirgen/compiler/zkp/digest.h"
#include "zirgen/Dialect/BigInt/Bytecode/bibc.h"
#include "zirgen/Dialect/BigInt/Bytecode/bqint.h"

namespace zirgen::BigInt::Bytecode {

using BytePoly = std::vector<int32_t>;

struct EvalOutput {
  std::array<uint32_t, 4> z;
  std::vector<BytePoly> constantWitness;
  std::vector<BytePoly> publicWitness;
  std::vector<BytePoly> privateWitness;
};

BytePoly fromBQInt(BQInt value, size_t coeffs);
Digest computeDigest(std::vector<BytePoly> witness, size_t groupSize = 3);

EvalOutput eval(const Program &func, std::vector<BQInt> &witnessValues);

} // namespace zirgen::BigInt::Bytecode
