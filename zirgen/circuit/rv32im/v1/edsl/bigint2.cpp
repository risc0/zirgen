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

BigInt2CycleImpl::BigInt2CycleImpl(RamHeader ramHeader)
    : ram(ramHeader, 6), mix("mix"), poly("accum"), term("accum"), tot("accum"), tmp("accum") {
  this->registerCallback("compute_accum", &BigInt2CycleImpl::onAccum);
}

void BigInt2CycleImpl::setByte(Val v, size_t i) {
  if (i < 13) {
    bytes[i]->set(v);
  } else {
    twitBytes[i - 13]->set(v);
  }
}

Val BigInt2CycleImpl::getByte(size_t i) {
  if (i < 13) {
    return bytes[i]->get();
  } else {
    return twitBytes[i - 13]->get();
  }
}

void BigInt2CycleImpl::set(Top top) {
  // Get some basic state data
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  Val curPC = BACK(1, body->pc->get());

  // Verify prev major was set right
  eqz(BACK(1, body->nextMajor->get()) - MajorType::kBigInt2);

  // Get current instruction address (either from ecall or prev instruction
  Val isFirstCycle = BACK(1, body->majorSelect->at(MajorType::kECall));

  IF(isFirstCycle) {
    NONDET { doExtern("syscallBigInt2Precompute", "", 0, {}); }
    // If first cycle, do special initalization
    ECallCycle ecall = body->majorMux->at<MajorType::kECall>();
    ECallBigInt2 ecallBigInt2 = ecall->minorMux->at<ECallType::kBigInt2>();
    instWordAddr->set(BACK(1, ecallBigInt2->readVerifyAddr->data().flat()) / kWordSize - 1);
    readInst->doNOP();
    polyOp->set(0);
    memOp->set(2);
    for (size_t i = 0; i < 5; i++) {
      checkReg[i]->set(0);
    }
    for (size_t i = 0; i < 3; i++) {
      checkCoeff[i]->set(0);
    }
    offset->set(0);
  }
  IF(1 - isFirstCycle) {
    // Read & decode the instruction
    instWordAddr->set(BACK(1, instWordAddr->get()) + 1);
    readInst->doRead(cycle, instWordAddr);
    Val instType = readInst->data().bytes[3];
    Val coeffReg = readInst->data().bytes[2];
    NONDET {
      polyOp->set(instType & 0xf);
      memOp->set((instType - polyOp->get()) / 16);
      for (size_t i = 0; i < 5; i++) {
        checkReg[i]->set((coeffReg & (1 << i)) / (1 << i));
      }
      for (size_t i = 0; i < 3; i++) {
        checkCoeff[i]->set((coeffReg & (1 << (5 + i))) / (1 << (5 + i)));
      }
    }
    eq(instType, polyOp->get() + memOp->get() * 16);
    offset->set(readInst->data().bytes[1] * 256 + readInst->data().bytes[0]);
  }
  Val reg = 0;
  for (size_t i = 0; i < 5; i++) {
    reg = reg + checkReg[i] * (1 << i);
  }
  Val coeff = 0;
  for (size_t i = 0; i < 3; i++) {
    coeff = coeff + checkCoeff[i] * (1 << i);
  }
  XLOG("BigInt2: instAddr = %x, polyOp=%u, memOp=%u, reg=%u, coeff+4=%u, offset=%u",
       instWordAddr * 4,
       polyOp,
       memOp,
       reg,
       coeff,
       offset);

  // Read the register value and compute initial address
  readRegAddr->doRead(cycle, kRegisterOffset + reg);
  Val addr = readRegAddr->data().flat() / 4 + offset * 4;

  // MemoryOp 0 (read)
  IF(memOp->at(0)) {
    NONDET {
      doExtern("syscallBigInt2Witness", "", 16, {polyOp->get(), memOp->get(), reg, offset, coeff});
    }
    for (size_t i = 0; i < 4; i++) {
      io[i]->doRead(cycle, addr + i);
      for (size_t j = 0; j < 4; j++) {
        setByte(io[i]->data().bytes[j], i * 4 + j);
      }
    }
  }

  // MemoryOp (1, 2) (write / check)
  IF(memOp->at(1) + memOp->at(2)) {
    NONDET {
      std::vector<Val> ret = doExtern(
          "syscallBigInt2Witness", "", 16, {polyOp->get(), memOp->get(), reg, offset, coeff});
      for (size_t i = 0; i < 16; i++) {
        setByte(ret[i], i);
      }
    }
  }

  // Memory Op 1 (write)
  IF(memOp->at(1)) {
    for (size_t i = 0; i < 4; i++) {
      io[i]->doWrite(
          cycle,
          addr + i,
          U32Val(getByte(i * 4 + 0), getByte(i * 4 + 1), getByte(i * 4 + 2), getByte(i * 4 + 3)));
    }
  }
  IF(memOp->at(2)) {
    for (size_t i = 0; i < 4; i++) {
      io[i]->doNOP();
    }
  }

  // Check is the instruction is a pure NOP + not first
  isLast->set(polyOp->at(0) * (1 - isFirstCycle));
  // If last, back to decoding
  IF(isLast) {
    body->nextMajor->set(MajorType::kMuxSize);
    body->pc->set(curPC + 4);
  }
  // Otherwise, next is also BigInt2 major
  IF(1 - isLast) {
    body->nextMajor->set(MajorType::kBigInt2);
    body->pc->set(curPC);
  }
}

static FpExt extBack(FpExtReg in) {
  std::array<Val, kExtSize> oldVals;
  for (size_t i = 0; i < 4; i++) {
    oldVals[i] = UNCHECKED_BACK(1, in->elem(i));
  }
  return FpExt(oldVals);
}

void BigInt2CycleImpl::onAccum() {
  std::vector<FpExt> powers;
  Val coeffVal = checkCoeff[0] + checkCoeff[1] * 2 + checkCoeff[2] * 4 - 4;
  FpExt coeff(coeffVal);
  FpExt zero(Val(0));
  FpExt one(Val(1));
  FpExt c256(Val(256));
  FpExt c16k(Val(16384));
  FpExt cur = one;
  for (size_t i = 0; i < 17; i++) {
    powers.push_back(cur);
    cur = cur * mix;
  }

  FpExt oldPoly = extBack(poly);
  FpExt oldTerm = extBack(term);
  FpExt oldTot = extBack(tot);
  FpExt deltaPoly = zero;
  FpExt negPoly = zero;
  for (size_t i = 0; i < 16; i++) {
    deltaPoly = deltaPoly + powers[i] * FpExt(getByte(i));
    negPoly = negPoly + powers[i] * FpExt(Val(128));
  }
  FpExt newPoly = oldPoly + deltaPoly;

  IF(polyOp->at(0)) {
    poly->set(zero);
    term->set(one);
    tot->set(zero);
  }
  IF(polyOp->at(PolyOp::kOpShift)) {
    poly->set(newPoly * powers[16]);
    term->set(oldTerm);
    tot->set(oldTot);
  }
  IF(polyOp->at(PolyOp::kOpSetTerm)) {
    poly->set(zero);
    term->set(newPoly);
    tot->set(oldTot);
  }
  IF(polyOp->at(PolyOp::kOpAddTot)) {
    poly->set(zero);
    term->set(one);
    tmp->set(coeff * oldTerm);
    tot->set(oldTot + tmp * newPoly);
  }
  IF(polyOp->at(PolyOp::kOpCarry1)) {
    poly->set(oldPoly + (deltaPoly - negPoly) * c16k);
    term->set(oldTerm);
    tot->set(oldTot);
  }
  IF(polyOp->at(PolyOp::kOpCarry2)) {
    poly->set(oldPoly + deltaPoly * c256);
    term->set(oldTerm);
    tot->set(oldTot);
  }
  IF(polyOp->at(PolyOp::kOpEqz)) {
    FpExt carryMul = powers[1] - c256;
    FpExt goalZero = oldTot + newPoly * carryMul;
    eqz(goalZero.elem(0));
    eqz(goalZero.elem(1));
    eqz(goalZero.elem(2));
    eqz(goalZero.elem(3));
    poly->set(zero);
    term->set(one);
    tot->set(zero);
  }
}

} // namespace zirgen::rv32im_v1
