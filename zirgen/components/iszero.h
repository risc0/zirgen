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

#include "zirgen/components/bits.h"
#include "zirgen/components/reg.h"

namespace zirgen {

class IsZeroImpl : public CompImpl<IsZeroImpl> {
public:
  // Sets a register value
  Val set(Val val);
  // Checks whether the register value is 0
  Val isZero();

private:
  // Records whether the register value is 0
  Bit isZeroBit;
  // Records the finite field multiplicative inverse of the register value
  Reg invVal;
};

using IsZero = Comp<IsZeroImpl>;

} // namespace zirgen
