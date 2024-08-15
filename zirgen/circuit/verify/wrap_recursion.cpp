// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
