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

#include "zirgen/components/iszero.h"

namespace zirgen {

Val IsZeroImpl::set(Val val) {
  NONDET {
    isZeroBit->set(isz(val));
    invVal->set(inv(val));
  }
  // The following IF statements generate constraints to prove that isZeroBit was set correctly
  // based on val. Two constraints are generated: c1(isZeroBit, val) = isZeroBit * val
  IF(isZeroBit) { eqz(val); }
  // c2(isZeroBit, val) = (1 - isZeroBit) * (val*invVal - 1)
  IF(1 - isZeroBit) { eq(val * invVal, 1); }
  // Each constraint must evaluate to 0, which enforces the following logic:
  // If isZeroBit is 1, c1 enforces that val == 0.
  // If isZeroBit is 0, c2 enforces that val is non-zero.
  // The logic behind c2 is that if val is non-zero, we can multiply and get something non-zero.
  // If isZeroBit is neither 0 nor 1, either c1 or c2 will fail and proof generation will abort.
  return isZeroBit;
}

Val IsZeroImpl::isZero() {
  return isZeroBit;
}

} // namespace zirgen
