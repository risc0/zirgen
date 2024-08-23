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

#include "zirgen/circuit/rv32im/v1/edsl/compute.h"

#include "zirgen/circuit/rv32im/v1/edsl/top.h"

namespace zirgen::rv32im_v1 {

using BitVec = std::array<Val, 32>;
using ShortVec = std::array<Val, 2>;

static BitVec get(std::vector<Bit>& reg, size_t back) {
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

ShaCycleImpl::ShaCycleImpl(size_t major, RamHeader ramHeader) : major(major), ram(ramHeader, 2) {
  a.resize(32);
  e.resize(32);
  twits.resize(10);
  bytes.resize(22);
  for (size_t i = 0; i < 10; i++) {
    w.emplace_back(ShareBitWithRegister(), twits[i]->reg);
  }
  for (size_t i = 0; i < 22; i++) {
    w.emplace_back(ShareBitWithRegister(), bytes[i]->reg);
  }
}

static Val toBits(std::vector<Bit> out, Val in, size_t offset) {
  NONDET {
    for (size_t i = 0; i < 16; i++) {
      out[i + offset]->set((in & (1 << i)) / (1 << i));
    }
  }
  Val low16 = 0;
  for (size_t i = 0; i < 16; i++) {
    low16 = low16 + out[i + offset] * (1 << i);
  }
  Val carry = (in - low16) / (1 << 16);
  return carry;
}

static void setCarry4(std::vector<Bit> out, ShortVec in, Twit carryLow, Twit carryHigh) {
  carryLow->set(toBits(out, in[0], 0));
  carryHigh->set(toBits(out, in[1] + carryLow, 16));
}

static void setCarry8(std::vector<Bit> out, ShortVec in, Twit carryLow, Twit carryHigh) {
  Val carryLow8 = toBits(out, in[0], 0);
  NONDET { carryLow->set(carryLow8 & 3); }
  Val carryLow1 = (carryLow8 - carryLow) / 4;
  eqz(carryLow1 * (1 - carryLow1));
  Val carryHigh8 = toBits(out, in[1] + carryLow8, 16);
  NONDET { carryHigh->set(carryHigh8 & 3); }
  Val carryHigh1 = (carryHigh8 - carryHigh) / 4;
  eqz(carryHigh1 * (1 - carryHigh1));
}

static ShortVec getShort(U32Val val) {
  return {val.bytes[0] + 256 * val.bytes[1], val.bytes[2] + 256 * val.bytes[3]};
}

static ShortVec getShortBE(U32Val val) {
  return {val.bytes[3] + 256 * val.bytes[2], val.bytes[1] + 256 * val.bytes[0]};
}

static ShortVec getShort(std::array<Reg, 2> val) {
  return {val[0]->get(), val[1]->get()};
}

static U32Val toU32BE(std::vector<Bit> bits) {
  U32Val out = {0, 0, 0, 0};
  for (size_t i = 0; i < 8; i++) {
    for (size_t j = 0; j < 4; j++) {
      out.bytes[4 - j - 1] = out.bytes[4 - j - 1] + bits[j * 8 + i] * (1 << i);
    }
  }
  return out;
}

void ShaCycleImpl::set(Top top) {
  switch (major) {
  case MajorType::kShaInit:
    setInit(top);
    break;
  case MajorType::kShaLoad:
    setLoad(top);
    break;
  case MajorType::kShaMain:
    setMain(top);
    break;
  }
}

void ShaCycleImpl::setInit(Top top) {
  // Get some basic state data
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  Val curPC = BACK(1, body->pc->get());

  // Set the subtype + verify
  eqz(BACK(1, body->nextMajor->get()) - MajorType::kShaInit);
  Val isFromEcall = BACK(1, body->majorSelect->at(MajorType::kECall));
  Val isFromPageFault = BACK(1, body->majorSelect->at(MajorType::kPageFault));
  IF(isFromEcall + isFromPageFault) {
    minor->set(0);
    count->set(4);
  }
  IF(1 - isFromEcall - isFromPageFault) {
    // Handle staying in the stage
    minor->set(BACK(1, minor->get()));
    count->set(BACK(1, count->get() - 1));
  }
  countZero->set(count);
  // Set next major type if switching stages
  IF(countZero->isZero()) { body->nextMajor->set(MajorType::kShaLoad); }
  IF(1 - countZero->isZero()) { body->nextMajor->set(body->majorSelect); }
  // Keep PC the same
  body->pc->set(curPC);
  XLOG("SHA_INIT: major = %u, minor = %u, count = %u", major, minor, count);
  IF(isFromEcall) {
    ECallCycle ecall = body->majorMux->at<MajorType::kECall>();
    ECallSha ecallSha = ecall->minorMux->at<ECallType::kSha>();
    io0->doRead(cycle, RegAddr::kA2);
    io1->doRead(cycle, RegAddr::kA3);
    stateOut->set(BACK(1, ecallSha->readA0->data().flat()) / kWordSize);
    stateIn->set(BACK(1, ecallSha->readA1->data().flat()) / kWordSize);
    data0->set(io0->data().flat() / kWordSize);
    data1->set(io1->data().flat() / kWordSize);
    repeat->set(BACK(1, ecallSha->readA4->data().flat()));
    mode->set(0);
    readOp->set(MemoryOpType::kRead);
    XLOG("  FromEcall: stateOut = %10x, stateIn = %10x, data0 = %10x, data1 = %10x, repeat: %u",
         stateOut * kWordSize,
         stateIn * kWordSize,
         data0 * kWordSize,
         data1 * kWordSize,
         repeat);
  }
  IF(isFromPageFault) {
    PageFaultCycle pageFault = body->majorMux->at<MajorType::kPageFault>();
    io0->doNOP();
    io1->doNOP();
    stateOut->set(BACK(1, pageFault->stateOut->get()));
    stateIn->set(BACK(1, pageFault->stateIn->get()));
    data0->set(BACK(1, getPageAddr(pageFault->pageIndex->get())));
    data1->set(BACK(1, getPageAddr(pageFault->pageIndex->get())) + kDigestWords);
    repeat->set(BACK(1, pageFault->repeat->get()));
    mode->set(BACK(1, pageFault->isRead->get()));
    readOp->set(MemoryOpType::kPageIo);
    XLOG("  FromPageFault: stateOut = %10x, stateIn = %10x, data0 = %10x, data1 = %10x, repeat: %u",
         stateOut * kWordSize,
         stateIn * kWordSize,
         data0 * kWordSize,
         data1 * kWordSize,
         repeat);
  }
  IF(1 - isFromEcall - isFromPageFault) {
    stateOut->set(BACK(1, stateOut->get()));
    stateIn->set(BACK(1, stateIn->get()));
    data0->set(BACK(1, data0->get()));
    data1->set(BACK(1, data1->get()));
    repeat->set(BACK(1, repeat->get()));
    mode->set(BACK(1, mode->get()));
    readOp->set(BACK(1, readOp->get()));

    // First do the memory I/O (if it's a read)
    io0->doRead(cycle, stateIn + count);
    io1->doRead(cycle, stateIn + count + 4);
    // XLOG("  State in: %w, %w", io0->data(), io1->data());
  }

  finalStage->set(0);
  repeatZero->set(repeat);

  setCarry4(w, {0, 0}, wCarryLow, wCarryHigh);
  // XLOG("  w = %w", toU32(w));

  setCarry8(a, getShortBE(io0->data()), aCarryLow, aCarryHigh);
  setCarry8(e, getShortBE(io1->data()), eCarryLow, eCarryHigh);
  // XLOG("  a = %w, e = %w", toU32(a), toU32(e));
}

void ShaCycleImpl::setLoad(Top top) {
  // Get some basic state data
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  Val curPC = BACK(1, body->pc->get());

  // Set the subtype + verify
  eqz(BACK(1, body->nextMajor->get()) - MajorType::kShaLoad);
  Val isBackInit = BACK(1, body->majorSelect->at(MajorType::kShaInit));
  Val isBackMain = BACK(1, body->majorSelect->at(MajorType::kShaMain));
  IF(isBackInit + isBackMain) {
    minor->set(0);
    count->set(7);
  }
  IF(1 - isBackInit - isBackMain) {
    Val newStage = BACK(1, countZero->isZero());
    IF(newStage) {
      minor->set(1);
      count->set(7);
    }
    IF(1 - newStage) {
      // Handle staying in the stage
      minor->set(BACK(1, minor->get()));
      count->set(BACK(1, count->get() - 1));
    }
  }

  countZero->set(count);
  // Set next major type if switching stages
  IF(countZero->isZero()) {
    IF(1 - minor) { body->nextMajor->set(MajorType::kShaLoad); }
    IF(minor) { body->nextMajor->set(MajorType::kShaMain); }
  }
  IF(1 - countZero->isZero()) { body->nextMajor->set(body->majorSelect); }
  // Keep PC the same
  body->pc->set(curPC);
  stateOut->set(BACK(1, stateOut->get()));
  stateIn->set(BACK(1, stateIn->get()));
  data0->set(BACK(1, data0->get()));
  data1->set(BACK(1, data1->get()));
  repeat->set(BACK(1, repeat->get()));
  mode->set(BACK(1, mode->get()));
  readOp->set(BACK(1, readOp->get()));
  repeatZero->set(repeat);
  finalStage->set(0);
  XLOG("SHA_LOAD: major = %u, minor = %u, count = %u, data0 = %10x, data1 = %10x, state = %10x, "
       "repeat: %u",
       major,
       minor,
       count,
       data0 * kWordSize,
       data1 * kWordSize,
       stateOut * kWordSize,
       repeat);

  // First do the memory IO (if it's a read)
  IF(1 - minor) {
    io0->doRead(cycle, data0 + 7 - count, readOp->get());
    io1->doRead(cycle, kShaKOffset + 7 - count);
  }
  IF(minor) {
    io0->doRead(cycle, data1 + 7 - count, readOp->get());
    io1->doRead(cycle, kShaKOffset + 15 - count);
  }
  // XLOG("  Data in: %w, k = %w", io0->data(), io1->data());

  setCarry4(w, getShortBE(io0->data()), wCarryLow, wCarryHigh);
  // XLOG("  w = %w", toU32(w));

  // Now we compute and set a + e
  computeAE();
  // XLOG("  ae");

  setCarry8(a, getShort(aRaw), aCarryLow, aCarryHigh);
  setCarry8(e, getShort(eRaw), eCarryLow, eCarryHigh);
  // XLOG("  a = %w, e = %w", toU32(a), toU32(e));
}

void ShaCycleImpl::setMain(Top top) {
  // Get some basic state data
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  Val curPC = BACK(1, body->pc->get());

  // Set the subtype + verify
  eqz(BACK(1, body->nextMajor->get()) - MajorType::kShaMain);
  Val newStage = BACK(1, countZero->isZero());
  IF(newStage) {
    Val isBackLoad = BACK(1, body->majorSelect->at(MajorType::kShaLoad));
    IF(isBackLoad) {
      minor->set(0);
      count->set(47);
      repeat->set(BACK(1, repeat->get()));
    }
    IF(1 - isBackLoad) {
      minor->set(1);
      count->set(3);
      repeat->set(BACK(1, repeat->get() - 1));
    }
  }
  IF(1 - newStage) {
    // Handle staying in the stage
    minor->set(BACK(1, minor->get()));
    count->set(BACK(1, count->get() - 1));
    repeat->set(BACK(1, repeat->get()));
  }
  countZero->set(count);

  Val isMix = 1 - minor;
  Val isFini = minor;

  // Decrement the repeat as necessary
  IF(countZero->isZero()) {
    IF(isMix) { finalStage->set(0); }
    IF(isFini) { finalStage->set(1); }
  }
  IF(1 - countZero->isZero()) { finalStage->set(0); }

  stateIn->set(BACK(1, stateIn->get()));
  stateOut->set(BACK(1, stateOut->get()));
  mode->set(BACK(1, mode->get()));
  readOp->set(BACK(1, readOp->get()));

  repeatZero->set(repeat);

  // Keep PC the same
  body->pc->set(curPC);
  XLOG("SHA_MAIN: major = %u, minor = %u, count = %u, repeat = %u", major, minor, count, repeat);

  // First do the memory IO (if it's a read)
  IF(isMix) {
    io1->doRead(cycle, kShaKOffset + 63 - count);
    // XLOG("  k = %w", io1->data());
  }

  // Now we compute and set w...
  computeW();

  IF(isFini) { setCarry4(w, {0, 0}, wCarryLow, wCarryHigh); }
  IF(isMix) { setCarry4(w, getShort(wRaw), wCarryLow, wCarryHigh); }
  // XLOG("  w = %w", toU32(w));

  // If we are writing, we need to do it now
  IF(isFini) {
    setCarry8(a, add(flat(get(a, 4)), flat(get(a, 68))), aCarryLow, aCarryHigh);
    setCarry8(e, add(flat(get(e, 4)), flat(get(e, 68))), eCarryLow, eCarryHigh);
  }

  Val isVerify = mode;
  Val isWrite = 1 - mode;

  IF(repeatZero->isZero()) {
    IF(isVerify) {
      io0->doRead(cycle, stateOut + count);
      io1->doRead(cycle, stateOut + 4 + count);
      XLOG("  io0: [%10x] %w, a: %w", io0->addr() * kWordSize, io0->data(), toU32BE(a));
      XLOG("  io1: [%10x] %w, e: %w", io1->addr() * kWordSize, io1->data(), toU32BE(e));
      eq(io0->data().flat(), toU32BE(a).flat());
      eq(io1->data().flat(), toU32BE(e).flat());
    }
    IF(isWrite) {
      io0->doWrite(cycle, stateOut + count, toU32BE(a));
      io1->doWrite(cycle, stateOut + 4 + count, toU32BE(e));
    }
  }
  IF(1 - repeatZero->isZero()) {
    io0->doNOP();
    IF(isFini) { io1->doNOP(); }
  }

  // Now we compute and set a + e
  computeAE();
  IF(isMix) {
    setCarry8(a, getShort(aRaw), aCarryLow, aCarryHigh);
    setCarry8(e, getShort(eRaw), eCarryLow, eCarryHigh);
  }
  // XLOG("  a: %w, e: %w", toU32BE(a), toU32BE(e));

  IF(finalStage->get()) {
    IF(repeatZero->isZero()) {
      data0->set(BACK(1, data0->get()));
      data1->set(BACK(1, data1->get()));
      body->nextMajor->set(MajorType::kMuxSize);
    }

    IF(1 - repeatZero->isZero()) {
      data0->set(BACK(1, data0->get() + kBlockSize));
      data1->set(BACK(1, data1->get() + kBlockSize));
      body->nextMajor->set(MajorType::kShaLoad);
    }
  }
  IF(1 - finalStage->get()) {
    data0->set(BACK(1, data0->get()));
    data1->set(BACK(1, data1->get()));
    body->nextMajor->set(MajorType::kShaMain);
  }
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
  // XLOG("    a = %10x %10x %10x %10x", toU32(a_));
  // XLOG("    b = %10x %10x %10x %10x", toU32(b_));
  // XLOG("    c = %10x %10x %10x %10x", toU32(c_));
  // XLOG("    d = %10x %10x %10x %10x", toU32(d_));
  // XLOG("    e = %10x %10x %10x %10x", toU32(e_));
  // XLOG("    f = %10x %10x %10x %10x", toU32(f_));
  // XLOG("    g = %10x %10x %10x %10x", toU32(g_));
  // XLOG("    h = %10x %10x %10x %10x", toU32(h_));
  ShortVec k_ = getShort(io1->data());
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

} // namespace zirgen::rv32im_v1
