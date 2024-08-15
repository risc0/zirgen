// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/compiler/r1cs/r1csfile.h"
#include "zirgen/compiler/r1cs/wtnsfile.h"

namespace zirgen::R1CS {

void validate(const r1csfile::System& sys, const wtnsfile::Witness& wtns);

} // namespace zirgen::R1CS
