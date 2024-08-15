// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "u32.h"

namespace zirgen {

U32RegImpl::U32RegImpl(llvm::StringRef source) {
  for (size_t i = 0; i < 4; i++) {
    bytes.emplace_back(Label("byte", i), source);
  }
}

U32Val U32Val::underflowProtect() {
  return U32Val(0x100, 0xff, 0xff, 0xff);
}

void eq(U32Val a, U32Val b, SourceLoc loc) {
  OverrideLocation local(loc);
  for (size_t i = 0; i < 4; i++) {
    eq(a.bytes[i], b.bytes[i]);
  }
}

void U32RegImpl::setZero() {
  for (size_t i = 0; i < 4; i++) {
    bytes[i]->set(0);
  }
}

void U32RegImpl::set(U32Val in) {
  for (size_t i = 0; i < 4; i++) {
    bytes[i]->set(in.bytes[i]);
  }
}

void U32RegImpl::setWithFactor(U32Val in, Val factor) {
  for (size_t i = 0; i < 4; i++) {
    bytes[i]->set(in.bytes[i] * factor);
  }
}

U32Val U32RegImpl::get() {
  U32Val out;
  for (size_t i = 0; i < 4; i++) {
    out.bytes[i] = bytes[i];
  }
  return out;
}

U32Val U32Val::fromFlat(Val in) {
  Val b0 = in & 0xFF;
  Val b1 = in & 0xFF00;
  Val b2 = in & 0xFF0000;
  Val b3 = in - b0 - b1 - b2;

  U32Val result;
  result.bytes[0] = b0;
  result.bytes[1] = b1 / (1 << 8);
  result.bytes[2] = b2 / (1 << 16);
  result.bytes[3] = b3 / (1 << 24);

  return result;
}

Val U32RegImpl::getSmallSigned() {
  Val pos = bytes[0] + (bytes[1] * (1 << 8)) + (bytes[2] * (1 << 16));
  // We presume bytes[3] is either 0 or 255, and if it's 255, we interpret
  // the value as negative
  return pos + (bytes[3] / 255) * (Val(0) - (1 << 24));
}

Val U32RegImpl::getSmallUnsigned() {
  return bytes[0] + (bytes[1] * (1 << 8)) + (bytes[2] * (1 << 16)) + (bytes[3] * (1 << 24));
}

std::vector<Val> U32RegImpl::toVals() {
  std::vector<Val> out;
  for (size_t i = 0; i < 4; i++) {
    out.push_back(bytes[i]);
  }
  return out;
}

void U32RegImpl::setFromVals(std::vector<Val> vals) {
  for (size_t i = 0; i < 4; i++) {
    bytes[i]->set(vals[i]);
  }
}

U32Val operator+(U32Val a, U32Val b) {
  U32Val out;
  for (size_t i = 0; i < 4; i++) {
    out.bytes[i] = a.bytes[i] + b.bytes[i];
  }
  return out;
}

U32Val operator-(U32Val a, U32Val b) {
  U32Val out;
  for (size_t i = 0; i < 4; i++) {
    out.bytes[i] = a.bytes[i] - b.bytes[i];
  }
  return out;
}

U32Val operator*(Val scalar, U32Val a) {
  U32Val out;
  for (size_t i = 0; i < 4; i++) {
    out.bytes[i] = scalar * a.bytes[i];
  }
  return out;
}

U32Val operator&(U32Val a, U32Val b) {
  U32Val out;
  for (size_t i = 0; i < 4; i++) {
    out.bytes[i] = a.bytes[i] & b.bytes[i];
  }
  return out;
}

void TopBitImpl::set(U32Val in) {
  NONDET {
    topBit->set((in.bytes[3] & 0x80) / 128);
    lowBits2->setExact((in.bytes[3] & 0x7f) * 2);
  }
  eq(in.bytes[3], topBit * 128 + lowBits2 / 2);
  low31 = in;
  low31.bytes[3] = lowBits2 / 2;
}

Val TopBitImpl::getHighBit() {
  return topBit;
}

U32Val TopBitImpl::getLow31() {
  return low31;
}

void IsZeroU32Impl::set(U32Val in) {
  low->set(in.bytes[0] + 256 * in.bytes[1]);
  high->set(in.bytes[2] + 256 * in.bytes[3] + 65536 * (1 - low->isZero()));
}

Val IsZeroU32Impl::isZero() {
  return high->isZero();
}

void U32NormalizeImpl::set(U32Val in) {
  Val low16 = in.bytes[0] + 256 * in.bytes[1];
  carry[0]->set(normed[1]->set(normed[0]->set(low16)));
  Val high16 = carry[0] + in.bytes[2] + 256 * in.bytes[3];
  carry[1]->set(normed[3]->set(normed[2]->set(high16)));
}

U32Val U32NormalizeImpl::getNormed() {
  return U32Val(normed[0], normed[1], normed[2], normed[3]);
}

Val U32NormalizeImpl::getCarry() {
  return carry[1];
}

U32Po2Impl::U32Po2Impl() {
  this->registerCallback("_builtin_verify", &U32Po2Impl::onVerify);
}

void U32Po2Impl::onVerify() {
  // Nicely name the four case of the top two bits
  Val topIs0 = (1 - bits[4]) * (1 - bits[3]);
  Val topIs1 = (1 - bits[4]) * bits[3];
  Val topIs2 = bits[4] * (1 - bits[3]);
  Val topIs3 = bits[4] * bits[3];
  // Get the po2 value
  U32Val po2 = out->get();
  // Check that all the non-matching bytes are zero
  IF(1 - topIs0) { eqz(po2.bytes[0]); }
  IF(1 - topIs1) { eqz(po2.bytes[1]); }
  IF(1 - topIs2) { eqz(po2.bytes[2]); }
  IF(1 - topIs3) { eqz(po2.bytes[3]); }
  // Get the byte in question
  Val byte =
      topIs0 * po2.bytes[0] + topIs1 * po2.bytes[1] + topIs2 * po2.bytes[2] + topIs3 * po2.bytes[3];
  // Verify it matches the low 3 bits
  eq(byte, (1 + 15 * bits[2]) * (1 + 3 * bits[1]) * (1 + bits[0]));
}

void U32Po2Impl::set(Val in) {
  NONDET {
    for (size_t i = 0; i < 5; i++) {
      bits[i]->set((in & (1 << i)) / (1 << i));
    }
    Val byte = (1 + 15 * bits[2]) * (1 + 3 * bits[1]) * (1 + bits[0]);
    Val top = 2 * bits[4] + bits[3];
    U32Val outVal = {
        isz(top - 0) * byte, isz(top - 1) * byte, isz(top - 2) * byte, isz(top - 3) * byte};
    out->set(outVal);
  }
  eq(get(), in);
}

Val U32Po2Impl::get() {
  Val tot = 0;
  for (size_t i = 0; i < 5; i++) {
    tot = tot + bits[i] * (1 << i);
  }
  return tot;
}

U32Val U32Po2Impl::getPo2() {
  return out->get();
}

void U32MulImpl::set(U32Val inA, U32Val inB, Val signedA, Val signedB) {
  topA->set(inA);
  topB->set(inB);
  subA->set(signedB * topB->getHighBit());
  subB->set(signedA * topA->getHighBit());
  for (int i = 0; i < 4; i++) {
    Val tot = 0;
    if (i != 0) {
      tot = tot + outRegs[(i - 1) * 3 + 2]->get();
      tot = tot + carry[i - 1]->get() * 256;
    }
    for (int p = 0; p < 2; p++) {
      Val subTotal = 0;
      int diag = 2 * i + p;
      for (int y = 0; y < 4; y++) {
        int x = diag - y;
        if (x >= 4 || x < 0) {
          continue;
        }
        subTotal = subTotal + inA.bytes[x] * inB.bytes[y];
      }
      tot = tot + subTotal * (p ? 256 : 1);
    }
    if (i == 2) {
      tot = tot + 0x20000 - (subA * (inA.bytes[0] + 256 * inA.bytes[1])) -
            (subB * (inB.bytes[0] + 256 * inB.bytes[1]));
    }
    if (i == 3) {
      tot = tot + 0x1fffe - (subA * (inA.bytes[2] + 256 * inA.bytes[3])) -
            (subB * (inB.bytes[2] + 256 * inB.bytes[3]));
    }
    tot = outRegs[i * 3 + 0]->set(tot);
    tot = outRegs[i * 3 + 1]->set(tot);
    if (i < 3) {
      tot = outRegs[i * 3 + 2]->set(tot);
    }
    carry[i]->set(tot);
  }
}

U32Val U32MulImpl::getHigh() {
  return {outRegs[6], outRegs[7], outRegs[9], outRegs[10]};
}

U32Val U32MulImpl::getLow() {
  return {outRegs[0], outRegs[1], outRegs[3], outRegs[4]};
}

void U32MulAccImpl::set(U32Val inA, U32Val inB, U32Val inC) {
  // First we compute the low two bytes
  Val low2 = inA.bytes[0] * inB.bytes[0] + inC.bytes[0] +
             256 * (inA.bytes[0] * inB.bytes[1] + inA.bytes[1] * inB.bytes[0] + inC.bytes[1]);
  // Now, set low two output registers + compute carry
  carryTwit->set(carryByte->set(outRegs[1]->set(outRegs[0]->set(low2))));
  Val carry = carryTwit * 256 + carryByte;
  // Verify that there is no trivial overflow
  eqz(inA.bytes[1] * inB.bytes[3]);
  eqz(inA.bytes[2] * inB.bytes[2]);
  eqz(inA.bytes[3] * inB.bytes[1]);
  eqz(inA.bytes[2] * inB.bytes[3]);
  eqz(inA.bytes[3] * inB.bytes[2]);
  eqz(inA.bytes[3] * inB.bytes[3]);
  // Now, compute high two bytes
  Val high2 = inA.bytes[2] * inB.bytes[0] + inA.bytes[1] * inB.bytes[1] +
              inA.bytes[0] * inB.bytes[2] + inC.bytes[2] + carry +
              256 * (inA.bytes[3] * inB.bytes[0] + inA.bytes[2] * inB.bytes[1] +
                     inA.bytes[1] * inB.bytes[2] + inA.bytes[0] * inB.bytes[3] + inC.bytes[3]);
  // Output top two bytes and make sure there is no overflow
  outRegs[3]->setExact(outRegs[2]->set(high2));
}

U32Val U32MulAccImpl::getOut() {
  return {outRegs[0], outRegs[1], outRegs[2], outRegs[3]};
}

} // namespace zirgen
