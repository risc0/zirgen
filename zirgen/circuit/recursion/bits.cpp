// Copyright 2025 RISC Zero, Inc.
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
  // Compute the bit-composition of the extension elem limbs, and the bitwise AND.
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
      // Incrementally compute the bitwise AND.
      totO = totO + bitsA[i * 16 + j] * bitsB[i * 16 + j] * bit;
    }
    // Confirm that the inputs are indeed 16-bit limbs.
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
    // NOTE: a ^ b == a + b - 2 * (a & b)
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
  // The running sums after bit 26, i.e. the integer formed by the low 27 bits of each input.  These
  // are reused by verifyCanonical so that the canonicality check adds no new subexpressions to the
  // constraint polynomial (the recursion programs that verify this circuit are close to their cycle
  // budget).
  Val lowA = 0;
  Val lowB = 0;
  for (size_t j = 0; j < bitsA.size(); j++) {
    size_t bit = 1 << j;

    NONDET {
      bitsA[j]->set((inA->data()[0] & bit) / bit);
      bitsB[j]->set((inB->data()[0] & bit) / bit);
    }
    totA = totA + bitsA[j] * bit;
    totB = totB + bitsB[j] * bit;
    totO = totO + bitsA[j] * bitsB[j] * bit;
    if (j == 26) {
      lowA = totA;
      lowB = totB;
    }
  }
  eq(totA, inA->data()[0]);
  eq(totB, inB->data()[0]);
  // The equalities above hold in the field, so they only fix the decomposition modulo P.  Require
  // the decompositions to be canonical so that the output is a function of the inputs.
  verifyCanonical(bitsA, lowA, topA);
  verifyCanonical(bitsB, lowB, topB);

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

void BitAndElemImpl::verifyCanonical(const std::vector<Bit>& bits, Val low27, Twit top) {
  // Baby Bear's prime is P = 0x78000001 = 2^30 + 2^29 + 2^28 + 2^27 + 1 < 2^31.  A 31-bit integer
  // is >= P exactly when bits 27-30 are all set and at least one of bits 0-26 is set.  Note that
  // bit 31 does not exist here, since we only decompose into 31 bits.
  static_assert(Zll::kFieldPrimeDefault == 0x78000001, "Canonicality check assumes Baby Bear");
  assert(bits.size() == 31);
  // Sum of the top 4 bits (0-4).  `low27` is the integer formed by bits 0-26, which is < 2^27 < P,
  // so as a field element it is zero exactly when all of the low 27 bits are zero.
  Val top4 = 0;
  for (size_t i = 27; i < 31; i++) {
    top4 = top4 + bits[i];
  }
  // Set the witness to top4, unless top4 == 4, in which case set it to 0.  Because `top` is a
  // twit (constrained to 0-3) it can never equal 4, so the prover cannot cancel the first factor
  // below when the top 4 bits are all set.
  NONDET {
    Val top4is4 = isz(top4 - 4);
    IF(top4is4) { top->set(0); }
    IF(1 - top4is4) { top->set(top4); }
  }
  // If top4 != 4, an honest witness makes the first factor zero.  If top4 == 4, the first factor
  // is nonzero, so low27 must be zero, i.e. the low 27 bits are all zero and the value is exactly
  // 2^30 + 2^29 + 2^28 + 2^27 = P - 1.  Either way the value is < P.
  eqz((top4 - top) * low27);
}

} // namespace zirgen::recursion
