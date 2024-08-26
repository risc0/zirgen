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

#include "zirgen/circuit/recursion/checked_bytes.h"

using namespace zirgen::recursion::poseidon2;

namespace zirgen::recursion {

CheckedBytesImpl::CheckedBytesImpl(Code code, WomHeader header)
    : body(Label("wom_body"), header, 2, 4) {}

CheckedBytesImpl::CheckedVals
CheckedBytesImpl::getPowersOfZ(FpExtReg tmp2, FpExtReg tmp4, FpExtReg tmp10, FpExt in) {
  CheckedVals out;
  out[0] = FpExt(Val(1));
  out[1] = in;
  NONDET {
    // Do this nondet, then verify in top.  This is a "green
    // wire"-style patch to fix the checked bytes functionality in the
    // existing circuit without invalidating proofs generated from the broken
    // circuit.
    //
    // If we added this constraint inline, it would increase the
    // exponent of poly_mix in all subsequent constraints, which would
    // invalidate any proofs that depend on them.
    //
    // So, we register a callback to the _green_wire phase, which
    // generates the constraint after all others.  Since the
    // checked_bytes mux arm is a constant 0 if checked_bytes isn't
    // used, this additional constraint will evaluate to be zero, and
    // be fully compatible with any other proofs that don't use
    // checked_bytes.
    tmp2->set(in * in);
    registerCallback("_green_wire", &CheckedBytesImpl::addPowersOfZConstraint);
  }
  out[2] = tmp2->get();
  out[3] = in * out[2];
  tmp4->set(out[2] * out[2]);
  out[4] = tmp4->get();
  out[5] = out[4] * in;
  out[6] = out[4] * out[2];
  out[7] = out[4] * out[2] * in;
  out[8] = out[4] * out[4];
  out[9] = out[4] * out[4] * in;
  tmp10->set(out[4] * out[4] * out[2]);
  out[10] = tmp10->get();
  out[11] = out[10] * in;
  out[12] = out[10] * out[2];
  out[13] = out[10] * out[3];
  out[14] = out[10] * out[4];
  out[15] = out[10] * out[4] * in;
  return out;
}

void CheckedBytesImpl::addPowersOfZConstraint() {
  // Verify checked-bytes power.  This is a "green wire"-style patch
  // to fix the checked bytes functionality in the existing circuit
  // without invalidating proofs from the broken circuit.
  FpExt zPt = eval_point->data();
  eq(power2->get(), zPt * zPt);
}

void CheckedBytesImpl::set(Code code, Val writeAddr) {
  auto inst = code->inst->at<size_t(OpType::CHECKED_BYTES)>();
  FpExt zPt = eval_point->doRead(inst->evalPoint->get());
  std::vector<Val> readVals;
  CheckedVals powersOfZ = getPowersOfZ(power2, power4, power10, zPt);
  FpExt out = Val(0);
  NONDET {
    auto vals = doExtern("readCoefficients", "", 16, {});
    for (size_t i = 0; i < 16; i++) {
      Val coeff = vals[i];
      lowestBits[i]->set(coeff & 0b00000011);
      midLoBits[i]->set(coeff & 0b00001100);
      midHiBits[i]->set(coeff & 0b00110000);
      highestBits[i]->set(coeff & 0b11000000);
    }
  }
  CellVals in;
  for (size_t i = 0; i < 16; i++) {
    auto lowest = lowestBits[i]->get();
    auto midLo = midLoBits[i]->get();
    auto midHi = midHiBits[i]->get();
    auto highest = highestBits[i]->get();
    // Check that each alleged bit pair is one of its four possible legal values
    eqz(lowest * (lowest - 1) * (lowest - 2) * (lowest - 3));
    eqz(midLo * (midLo - 4) * (midLo - 8) * (midLo - 12));
    eqz(midHi * (midHi - 16) * (midHi - 32) * (midHi - 48));
    eqz(highest * (highest - 64) * (highest - 128) * (highest - 192));
    // Make the coeff via the sum of the bit pairs
    auto coeff = lowest + midLo + midHi + highest;
    in[i] = inst->keepCoeffs * 256 * UNCHECKED_BACK(1, output[i]->get()) + coeff;
    // Update the polynomial
    out = out + FpExt(coeff) * powersOfZ[i];
  }
  XLOG("CHECKED_BYTES_EVAL: evalPt(%u), keepCoeffs(%u), keepUpperState(%u), prepFull(%u)",
       inst->evalPoint,
       inst->keepCoeffs,
       inst->keepUpperState,
       inst->prepFull);
  evaluation->doWrite(writeAddr, out.getElems());

  // Maybe copy final 1/3rd of state across
  for (size_t i = 16; i < 24; i++) {
    in[i] = inst->keepUpperState * UNCHECKED_BACK(1, output[i]->get());
  }
  // Now, set output
  CellVals mult = mulMExt(in);
  for (size_t i = 0; i < CELLS; i++) {
    Val out;
    out = inst->prepFull * mult[i] + (1 - inst->prepFull) * in[i]; // Maybe mulMExt
    out = out + inst->prepFull * ROUND_CONSTANTS[i];               // Maybe add constant
    output[i]->set(out);
  }
}

} // namespace zirgen::recursion
