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


#include "zirgen/Dialect/BigInt/IR/BigInt.h"

#include "zirgen/Dialect/BigInt/IR/Types.h.inc"

using namespace mlir;

namespace zirgen::BigInt {

LogicalResult BigIntType::verify(function_ref<InFlightDiagnostic()> emitError, size_t coeffs, size_t maxPos, size_t maxNeg, size_t minBits) {
  // TODO: Do this more nicely (i.e. more precise bound)
  return maxPos + maxNeg < ((uint64_t)1 << 31) ?
    success() :
    emitError() << "Cannot create BigInt with coefficients overflowing BabyBear: maxPos: " << maxPos << " + maxNeg: " << maxNeg;
}

} // namespace zirgen::BigInt
