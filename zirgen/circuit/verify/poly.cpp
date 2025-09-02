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

#include "zirgen/circuit/verify/poly.h"

namespace zirgen::verify {

Val poly_eval(const std::vector<Val>& coeffs, Val x) {
  ScopedLocation loc;

  Val tot = 0;
  Val mul = 1;
  for (size_t i = 0; i < coeffs.size(); i++) {
    tot = tot + coeffs[i] * mul;
    mul = mul * x;
  }
  return tot;
}

} // namespace zirgen::verify
