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

#include "zirgen/circuit/verify/wrap.h"

using zirgen::Val;
using zirgen::verify::CircuitBase;
using zirgen::verify::MixState;

#include "zirgen/circuit/recursion/impl.h"
#include "zirgen/circuit/recursion/poly_edsl.cpp"
#include "zirgen/circuit/recursion/taps.cpp"
#include "zirgen/compiler/codegen/protocol_info_const.h"

namespace zirgen::verify {

class CircuitInterfaceRecursion : public circuit::recursion::CircuitImpl {
public:
  CircuitInterfaceRecursion() { add_taps(); }

  ProtocolInfo get_circuit_info() const { return RECURSION_CIRCUIT_INFO; };
};

std::unique_ptr<CircuitInterface> getInterfaceRecursion() {
  return std::make_unique<CircuitInterfaceRecursion>();
}

} // namespace zirgen::verify
