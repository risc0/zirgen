// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/verify/poly.h"

namespace zirgen::verify {

Val poly_eval(const std::vector<Val>& coeffs, Val x) {
  Val tot = 0;
  Val mul = 1;
  for (size_t i = 0; i < coeffs.size(); i++) {
    tot = tot + coeffs[i] * mul;
    mul = mul * x;
  }
  return tot;
}

} // namespace zirgen::verify
