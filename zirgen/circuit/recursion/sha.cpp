// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/recursion/sha.h"

#include "zirgen/compiler/zkp/baby_bear.h"

namespace zirgen::recursion {

using BitVec = std::array<Val, 32>;
using ShortVec = std::array<Val, 2>;

static BitVec get(std::array<Bit, 32>& reg, size_t back) {
  BitVec ret;
  for (size_t i = 0; i < 32; i++) {
    ret[i] = BACK(back, reg[i]->get());
  }
  return ret;
}

static BitVec rightRotate(BitVec in, size_t n) {
  BitVec ret;
  for (size_t i = 0; i < 32; i++) {
    size_t from = (i + n) % 32;
    ret[i] = in[from];
  }
  return ret;
}

static BitVec rightShift(BitVec in, size_t n) {
  BitVec ret;
  for (size_t i = 0; i < 32; i++) {
    size_t from = i + n;
    if (from >= 32) {
      ret[i] = 0;
    } else {
      ret[i] = in[from];
    }
  }
  return ret;
}

static BitVec xor_(BitVec a, BitVec b) {
  BitVec ret;
  for (size_t i = 0; i < 32; i++) {
    ret[i] = a[i] + b[i] - 2 * a[i] * b[i];
  }
  return ret;
}

static BitVec maj(BitVec a, BitVec b, BitVec c) {
  BitVec ret;
  for (size_t i = 0; i < 32; i++) {
    ret[i] = a[i] * b[i] * (1 - c[i]) + a[i] * (1 - b[i]) * c[i] + (1 - a[i]) * b[i] * c[i] +
             a[i] * b[i] * c[i];
  }
  return ret;
}

static BitVec ch(BitVec a, BitVec b, BitVec c) {
  BitVec ret;
  for (size_t i = 0; i < 32; i++) {
    ret[i] = a[i] * b[i] + (1 - a[i]) * c[i];
  }
  return ret;
}

static ShortVec flat(BitVec a) {
  ShortVec ret;
  for (size_t i = 0; i < 2; i++) {
    ret[i] = 0;
    for (size_t j = 0; j < 16; j++) {
      ret[i] = ret[i] + (1 << j) * a[i * 16 + j];
    }
  }
  return ret;
}

static ShortVec add(ShortVec a, ShortVec b) {
  ShortVec ret;
  for (size_t i = 0; i < 2; i++) {
    ret[i] = a[i] + b[i];
  }
  return ret;
}

static ShortVec getShort(WomReg io) {
  return {io->data()[0], io->data()[1]};
}

static ShortVec getShort(std::array<Reg, 2> regs) {
  return {regs[0], regs[1]};
}

static Val toBits(std::array<Bit, 32> out, Val in, size_t offset, size_t count) {
  NONDET {
    for (size_t i = 0; i < count; i++) {
      out[i + offset]->set((in & (1 << i)) / (1 << i));
    }
  }
  Val low = 0;
  for (size_t i = 0; i < count; i++) {
    low = low + out[i + offset] * (1 << i);
  }
  Val carry = (in - low) / (1 << count);
  return carry;
}

std::array<Val, kWordSize> toBytes(std::array<Val, 32> in) {
  std::array<Val, kWordSize> bytes;
  for (size_t i = 0; i < kWordSize; i++) {
    Val byte = 0;
    for (size_t j = 0; j < 8; j++) {
      byte = byte + in[i * 8 + j] * (1 << j);
    }
    bytes[i] = byte;
  }
  return bytes;
}

static void setCarry(std::array<Bit, 32> out, ShortVec in, Twit carryLow, Twit carryHigh) {
  Val carryLow8 = toBits(out, in[0], 0, 16);
  NONDET { carryLow->set(carryLow8 & 3); }
  Val carryLow1 = (carryLow8 - carryLow) / 4;
  eqz(carryLow1 * (1 - carryLow1));
  Val carryHigh8 = toBits(out, in[1] + carryLow8, 16, 16);
  NONDET { carryHigh->set(carryHigh8 & 3); }
  Val carryHigh1 = (carryHigh8 - carryHigh) / 4;
  eqz(carryHigh1 * (1 - carryHigh1));
}

static void setBE(std::array<Bit, 32> out, ShortVec in) {
  eq(0, toBits(out, toBits(out, in[0], 24, 8), 16, 8));
  eq(0, toBits(out, toBits(out, in[1], 8, 8), 0, 8));
}

ShaCycleImpl::ShaCycleImpl(MacroOpcode major, WomHeader womHeader)
    : major(major), body(womHeader, 2, 4) {}

void ShaCycleImpl::set(MacroInst inst, Val writeAddr) {
  // XLOG("SHA_CYCLE = %u", int(major));
  // XLOG("  operands = [%u, %u, %u]", inst->operands[0], inst->operands[1], inst->operands[2]);
  switch (major) {
  case MacroOpcode::SHA_INIT:
    setInit(inst, writeAddr);
    break;
  case MacroOpcode::SHA_LOAD:
    setLoad(inst, writeAddr);
    break;
  case MacroOpcode::SHA_MIX:
    setMix(inst, writeAddr);
    break;
  case MacroOpcode::SHA_FINI:
    setFini(inst, writeAddr);
    break;
  default:
    throw std::runtime_error("Non-sha major");
  }
}

void ShaCycleImpl::setInit(MacroInst inst, Val writeAddr) {
  // XLOG("  SHA_INIT");
  io0->doRead(inst->operands[0]);
  io1->doRead(inst->operands[1]);
  setCarry(w, {0, 0}, wCarryLow, wCarryHigh);
  setCarry(a, getShort(io0), aCarryLow, aCarryHigh);
  setCarry(e, getShort(io1), eCarryLow, eCarryHigh);
  // XLOG("  INIT: a = %w", toBytes(get(a, 0)));
  // XLOG("  INIT: e = %w", toBytes(get(e, 0)));
}

void ShaCycleImpl::setLoad(MacroInst inst, Val writeAddr) {
  // XLOG("  SHA_LOAD");
  io0->doRead(inst->operands[0]);
  io1->doRead(inst->operands[1]);
  IF(1 - inst->operands[2]) {
    // Load the value montgomery encoded and reverse endian
    Val cur = kBabyBearToMontgomery * io0->data()[0];
    cur = toBits(w, cur, 24, 8);
    cur = toBits(w, cur, 16, 8);
    cur = toBits(w, cur, 8, 8);
    cur = toBits(w, cur, 0, 8);
    eq(cur, 0);
    // Verify the result is within the size of a field element
    // To do this, we note that for Baby Bear, the top most bit must be zero
    // and bits 27-31 can only be 1111 if all the rest of the lower bits are zero
    // First, we compute the sum of the 1 bits for the two cases
    Val top4 = 0;
    Val bot27 = 0;
    for (size_t i = 27; i < 31; i++) {
      size_t ii = (3 - (i / 8)) * 8 + i % 8;
      top4 = top4 + w[ii];
    }
    for (size_t i = 0; i < 27; i++) {
      size_t ii = (3 - (i / 8)) * 8 + i % 8;
      bot27 = bot27 + w[ii];
    }
    // Now, we set the value of wCarryLow (which is otherwise unused) to
    // top4 (unless it's == 4, in which case we set it to 0)
    NONDET {
      Val top4is4 = isz(top4 - 4);
      IF(top4is4) { wCarryLow->set(0); }
      IF(1 - top4is4) { wCarryLow->set(top4); }
    }
    // XLOG("cur = %u, top4 = %u, bot27 = %u, wCarryLow = %u",
    //          kBabyBearToMontgomery * io0->data()[0],
    //          top4,
    //          bot27,
    //          wCarryLow);
    // NOW: if top4 != 4, wCarryLow cancels it, thus the first term below is zero
    // Otherwise, if the second term is zero, we are OK.  If the low 31 bits are
    // >= P, the top 4 will be 4, and the lower 27 will be nonzero, so we fail
    eqz((top4 - wCarryLow) * bot27);
    // Also check the top bit
    eqz(w[7]);
  }
  IF(inst->operands[2]) {
    setBE(w, getShort(io0));
    wCarryLow->set(0);
  }
  XLOG("%u> SHA_LOAD: w = %w", writeAddr, toBytes(get(w, 0)));
  wCarryHigh->set(0);
  computeAE();
  setCarry(a, getShort(aRaw), aCarryLow, aCarryHigh);
  setCarry(e, getShort(eRaw), eCarryLow, eCarryHigh);
}

void ShaCycleImpl::setMix(MacroInst inst, Val writeAddr) {
  // XLOG("  SHA_MIX");
  io0->doNOP();
  io1->doRead(inst->operands[1]);
  computeW();
  setCarry(w, getShort(wRaw), wCarryLow, wCarryHigh);
  computeAE();
  setCarry(a, getShort(aRaw), aCarryLow, aCarryHigh);
  setCarry(e, getShort(eRaw), eCarryLow, eCarryHigh);
}

void ShaCycleImpl::setFini(MacroInst inst, Val writeAddr) {
  // XLOG("  SHA_FINI");
  setCarry(w, {0, 0}, wCarryLow, wCarryHigh);
  setCarry(a, add(flat(get(a, 4)), flat(get(a, 68))), aCarryLow, aCarryHigh);
  setCarry(e, add(flat(get(e, 4)), flat(get(e, 68))), eCarryLow, eCarryHigh);
  std::array<Val, kWordSize> outA = toBytes(get(a, 0));
  std::array<Val, kWordSize> outE = toBytes(get(e, 0));
  XLOG("%u> SHA_FINI: a = %w, e = %w", writeAddr, outA, outE);
  io0->doWrite(inst->operands[0], {outA[3] + 256 * outA[2], outA[1] + 256 * outA[0], 0, 0});
  io1->doWrite(inst->operands[1], {outE[3] + 256 * outE[2], outE[1] + 256 * outE[0], 0, 0});
}

void ShaCycleImpl::computeW() {
  auto w_2 = get(w, 2);
  auto w_7 = get(w, 7);
  auto w_15 = get(w, 15);
  auto w_16 = get(w, 16);
  auto s0 = xor_(rightRotate(w_15, 7), xor_(rightRotate(w_15, 18), rightShift(w_15, 3)));
  auto s1 = xor_(rightRotate(w_2, 17), xor_(rightRotate(w_2, 19), rightShift(w_2, 10)));
  auto w_0 = add(flat(w_16), add(flat(s0), add(flat(w_7), flat(s1))));
  for (size_t i = 0; i < 2; i++) {
    wRaw[i]->set(w_0[i]);
  }
}

void ShaCycleImpl::computeAE() {
  auto a_ = get(a, 1);
  auto b_ = get(a, 2);
  auto c_ = get(a, 3);
  auto d_ = get(a, 4);
  auto e_ = get(e, 1);
  auto f_ = get(e, 2);
  auto g_ = get(e, 3);
  auto h_ = get(e, 4);
  auto w_ = get(w, 0);
  // XLOG("    a = %w", toBytes(a_));
  // XLOG("    b = %w", toBytes(b_));
  // XLOG("    c = %w", toBytes(c_));
  // XLOG("    d = %w", toBytes(d_));
  // XLOG("    e = %w", toBytes(e_));
  // XLOG("    f = %w", toBytes(f_));
  // XLOG("    g = %w", toBytes(g_));
  // XLOG("    h = %w", toBytes(h_));
  ShortVec k_ = getShort(io1);
  auto s0 = xor_(rightRotate(a_, 2), xor_(rightRotate(a_, 13), rightRotate(a_, 22)));
  auto s1 = xor_(rightRotate(e_, 6), xor_(rightRotate(e_, 11), rightRotate(e_, 25)));
  auto stage1 = add(flat(w_), add(k_, add(flat(h_), add(flat(ch(e_, f_, g_)), flat(s1)))));
  auto aOut = add(stage1, add(flat(maj(a_, b_, c_)), flat(s0)));
  auto eOut = add(stage1, flat(d_));
  for (size_t i = 0; i < 2; i++) {
    aRaw[i]->set(aOut[i]);
    eRaw[i]->set(eOut[i]);
  }
}

} // namespace zirgen::recursion
