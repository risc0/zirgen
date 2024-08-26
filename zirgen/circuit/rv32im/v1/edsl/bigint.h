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

#include "zirgen/circuit/rv32im/v1/edsl/decode.h"
#include "zirgen/circuit/rv32im/v1/edsl/global.h"
#include "zirgen/circuit/rv32im/v1/platform/constants.h"
#include "zirgen/components/ram.h"

namespace zirgen::rv32im_v1 {

class TopImpl;
using Top = Comp<TopImpl>;

// Big integer arithmatic component supporting modular multiplication of 256-bit numbers.
//
// Big integers are represented to the guest as 8x32-bit limbs, with limbs ordered from least to
// most significant. In this circuit, BigInts are represented as 32x8-bit arrays ordered from least
// to most significant. Note that in rv32im, the packed memory representations are identical.
//
// Assumptions:
// * When the modulus input N is non-zero, at least one of the inputs x and y must be less N.
//   Otherwise there may be no valid witness.
// * When the modulus input is zero, the multiplication result must not overflow the bigint.
//   Otherwise there will not be a valid witness.
//   This PR adds to the RISC-V circuit a BigInt arithmatic component.
//
// See the description on https://github.com/risc0/risczero-wip/pull/200 for more detail.

class BigIntCycleImpl : public CompImpl<BigIntCycleImpl> {
public:
  BigIntCycleImpl(RamHeader ramHeader);
  void set(Top top);

  // Callback for checks that occur during the _builtin_verify phase.
  // Verifies the correctness of the multiply operation.
  void onVerify();

  // Registers for reading from and writing to RAM.
  // Supports reading or writing 4 words per cycle (e.g. 256-bits per 2 cycles).
  RamBody ram;
  std::array<RamReg, BigInt::kIoSize> io;
  Reg ioAddr;

  // Control registers.
  // BigInt component has 5 stages.
  // 0: Read input and output addresses.
  // 1: Read N value and set multiplier inputs for q and N.
  // 2: Read x value and constrain multiplication result r.
  // 3: Read y value and set multiplier inputs for x and y.
  // 4: Write output z value and constrain final multiplier result.
  OneHot<BigInt::kStages> stage;
  Bit stageOffset;
  Bit mulActive;
  Bit finalize;

  // Registers allocated for enforcing byte constraints.
  // 16 bytes are allocated (which is the max* available after I/O).
  // (* Actually there are up to 17 available, but that's a quirk.)
  std::vector<ByteReg> bytes;

  // Registers for the multiplier input and output.
  // 32 registers are allocated.
  std::vector<Reg> mulBuffer;

  // Registers for the high carries used in output normalization.
  // 8 registers are allocated to store the high carries. Low carries are in bytes.
  std::vector<Reg> carryHi;

  // FpExt register allocated in the mix buffer.
  // Provides a random challenge point for multiply equality check.
  FpExtReg mix;
};
using BigIntCycle = Comp<BigIntCycleImpl>;

} // namespace zirgen::rv32im_v1
