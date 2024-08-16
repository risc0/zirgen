// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/compiler/edsl/edsl.h"

namespace zirgen::verify {

Val poly_eval(const std::vector<Val>& coeffs, Val x);

} // namespace zirgen::verify
