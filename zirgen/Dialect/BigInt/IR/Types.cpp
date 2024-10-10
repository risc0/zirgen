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

#include "risc0/fp/fp.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"
#include "zirgen/Dialect/BigInt/IR/Types.h.inc"

using namespace mlir;

namespace zirgen::BigInt {

LogicalResult BigIntType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 size_t coeffs,
                                 size_t maxPos,
                                 size_t maxNeg,
                                 size_t minBits) {
  if (maxNeg > 0 && minBits > 0) {
    return emitError() << "BigInts with positive minBits must be positive: maxNeg: " << maxNeg
                       << ", minBits: " << minBits;
  }
  // TODO: Think through whether maxPos / maxNeg can ever overflow their attribute type, which would
  // cause problems here
  if (maxPos + maxNeg >= risc0::Fp::P) {
    return emitError() << "Cannot create BigInt with coefficients overflowing BabyBear: maxPos: "
                       << maxPos << " + maxNeg: " << maxNeg;
  }
  return success();
}

} // namespace zirgen::BigInt
