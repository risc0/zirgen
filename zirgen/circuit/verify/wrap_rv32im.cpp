// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/verify/wrap.h"

using zirgen::Val;
using zirgen::verify::CircuitBase;
using zirgen::verify::MixState;

#include "zirgen/circuit/rv32im/v1/edsl/impl.h"
#include "zirgen/circuit/rv32im/v1/edsl/poly_edsl.cpp"
#include "zirgen/circuit/rv32im/v1/edsl/taps.cpp"
#include "zirgen/compiler/codegen/protocol_info_const.h"

namespace zirgen::verify {

class CircuitInterfaceRV32IM : public circuit::rv32im::CircuitImpl {
public:
  CircuitInterfaceRV32IM() { add_taps(); }

  ProtocolInfo get_circuit_info() const { return RV32IM_CIRCUIT_INFO; };
};

std::unique_ptr<CircuitInterface> getInterfaceRV32IM() {
  return std::make_unique<CircuitInterfaceRV32IM>();
}

} // namespace zirgen::verify
