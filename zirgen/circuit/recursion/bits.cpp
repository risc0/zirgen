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

#include "zirgen/circuit/recursion/bits.h"

namespace zirgen::recursion {

BitOpShortsImpl::BitOpShortsImpl(WomHeader header) : body(Label("wom_body"), header, 3, 3) {}

void BitOpShortsImpl::set(MacroInst inst, Val writeAddr) {
  XLOG("BIT_OP_SHORTS, reading from [%u, %u], type=%u",
       inst->operands[0],
       inst->operands[1],
       inst->operands[2]);
  inA->doRead(inst->operands[0]);
  inB->doRead(inst->operands[1]);
  XLOG("  A = [%x, %x], B = [%x, %x]",
       inA->data()[0],
       inA->data()[1],
       inB->data()[0],
       inB->data()[1]);
  std::array<Val, 4> outO = {0, 0, 0, 0};
  for (size_t i = 0; i < 2; i++) {
    Val totA = 0;
    Val totB = 0;
    Val totO = 0;
    for (size_t j = 0; j < 16; j++) {
      Val bit = 1 << j;
      NONDET {
        bitsA[i * 16 + j]->set((inA->data()[i] & bit) / bit);
        bitsB[i * 16 + j]->set((inB->data()[i] & bit) / bit);
      }
      totA = totA + bitsA[i * 16 + j] * bit;
      totB = totB + bitsB[i * 16 + j] * bit;
      totO = totO + bitsA[i * 16 + j] * bitsB[i * 16 + j] * bit;
    }
    eq(totA, inA->data()[i]);
    eq(totB, inB->data()[i]);
    outO[i] = totO;
  }
  IF(inst->operands[2]) {
    // AND and combine [a, b, 0, 0] & [c, d, 0, 0] -> [(a & c) + ((b & d) << 16), 0, 0, 0]
    out->doWrite(writeAddr, {outO[1] * 65536 + outO[0], 0, 0, 0});
    XLOG("  AND Result = %e", out->data());
  }
  IF(1 - inst->operands[2]) {
    // XORs and returns 2 shorts: [a, b, 0, 0] ^ [c, d, 0, 0] -> [a ^ c, b ^ d, 0, 0]
    out->doWrite(writeAddr,
                 {inA->data()[0] + inB->data()[0] - 2 * outO[0],
                  inA->data()[1] + inB->data()[1] - 2 * outO[1],
                  0,
                  0});
    XLOG("  XOR Result = %e", out->data());
  }
}

BitAndElemImpl::BitAndElemImpl(WomHeader header) : body(Label("wom_body"), header, 3, 3) {
  for (size_t i = 0; (1U << i) < Zll::kFieldPrimeDefault; ++i) {
    bitsA.push_back(Bit(Label("bits_a", i)));
    bitsB.push_back(Bit(Label("bits_b", i)));
  }
}

void BitAndElemImpl::set(MacroInst inst, Val writeAddr) {
  inA->doRead(inst->operands[0]);
  inB->doRead(inst->operands[1]);
  Val totA = 0;
  Val totB = 0;
  Val totO = 0;
  for (size_t j = 0; j < bitsA.size(); j++) {
    size_t bit = 1 << j;

    NONDET {
      bitsA[j]->set((inA->data()[0] & bit) / bit);
      bitsB[j]->set((inB->data()[0] & bit) / bit);
    }
    totA = totA + bitsA[j] * bit;
    totB = totB + bitsB[j] * bit;
    totO = totO + bitsA[j] * bitsB[j] * bit;
  }
  eq(totA, inA->data()[0]);
  eq(totB, inB->data()[0]);

  out->doWrite(writeAddr, {totO, 0, 0, 0});
  XLOG("BIT_AND_ELEM, reading from [%u, %u],  A = [%x, %x], B = [%x, %x], type = %u, Out=%x, "
       "Result=%e",
       inst->operands[0],
       inst->operands[1],
       inA->data()[0],
       inA->data()[1],
       inB->data()[0],
       inB->data()[1],
       inst->operands[2],
       totO,
       out->data());
}

} // namespace zirgen::recursion
