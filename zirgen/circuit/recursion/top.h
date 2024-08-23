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

#include "zirgen/circuit/recursion/checked_bytes.h"
#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/recursion/macro.h"
#include "zirgen/circuit/recursion/micro.h"
#include "zirgen/circuit/recursion/poseidon2.h"

namespace zirgen::recursion {

struct TopImpl : public CompImpl<TopImpl> {
  TopImpl();
  void set();

  Code code = Label("code");
  WomHeader womHeader;
  Mux<MicroOps,
      MacroOp,
      Poseidon2Load,
      Poseidon2Full,
      Poseidon2Partial,
      Poseidon2Store,
      CheckedBytes>
      mux;
};
using Top = Comp<TopImpl>;

} // namespace zirgen::recursion
