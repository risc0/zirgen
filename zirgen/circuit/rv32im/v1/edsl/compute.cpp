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

void ComputeControlImpl::set(U32Val imm,
                             CompEnum::InA aluA,
                             CompEnum::InB aluB,
                             CompEnum::AluOp aluOp,
                             CompEnum::NextMajor nextMajor) {
  using namespace CompEnum;
  Val N1 = Val(0) - 1;
  this->imm->set(imm);
  this->aluA->set(int(aluA));
  this->aluB->set(int(aluB));
  switch (aluOp) {
  case ADD:
    mA->set(1);
    mB->set(1);
    mC->set(0);
    break;
  case SUB:
    mA->set(1);
    mB->set(N1);
    mC->set(0);
    break;
  case AND:
    mA->set(0);
    mB->set(0);
    mC->set(1);
    break;
  case OR:
    mA->set(1);
    mB->set(1);
    mC->set(N1);
    break;
  case XOR:
    mA->set(1);
    mB->set(1);
    mC->set(2 * N1);
    break;
  case INB:
    mA->set(0);
    mB->set(1);
    mC->set(0);
    break;
  }
  this->nextMajor->set(int(nextMajor));
}

void ALUImpl::set(U32Val inA, U32Val inB, ComputeControl control) {
  aTop->set(inA);
  bTop->set(inB);
  regInB->set(inB);
  NONDET { andVal->set(inA & inB); }
  result->set(U32Val::underflowProtect() + control->mA * inA + control->mB * inB +
              control->mC * andVal->get());
  rTop->set(result->getNormed());
  // Get the sign bits of both inputs + the output
  Val s1 = aTop->getHighBit();
  Val s2 = bTop->getHighBit();
  Val s3 = rTop->getHighBit();
  // Compute the 'overflow' status bit
  overflow->set(s1 * (1 - s2) * (1 - s3) + (1 - s1) * s2 * s3);
  // Compute signed LT
  lt->set(overflow + s3 - 2 * overflow * s3);
  // Set the isZero goo
  isZero->set(result->getNormed());
}

U32Val ALUImpl::getInB() {
  return regInB->get();
}

U32Val ALUImpl::getAndVal() {
  return andVal->get();
}

U32Val ALUImpl::getResult() {
  return result->getNormed();
}

Val ALUImpl::getEQ() {
  return isZero->isZero();
}

Val ALUImpl::getLT() {
  return lt;
}

Val ALUImpl::getLTU() {
  return 1 - result->getCarry();
}

ComputeCycleImpl::ComputeCycleImpl(size_t major, RamHeader ramHeader)
    : major(major), ram(ramHeader, 4) {}

void ComputeCycleImpl::set(Top top) {
  using namespace CompEnum;

  // Get some basic state data
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  Val curPC = BACK(1, body->pc->get());
  Val userMode = BACK(1, body->userMode->get());

  // We should always be in 'decode'
  eqz(BACK(1, body->nextMajor->get()) - MajorType::kMuxSize);

  // Read and decode the next instruction
  U32Val inst = readInst->doRead(cycle, curPC / 4);
  decoder->set(inst);

  // Pick minor select (also non-det)
  NONDET {
    Val minor =
        doExtern("getMinor", "", 1, std::vector<Val>(inst.bytes.begin(), inst.bytes.end()))[0];
    minorSelect->set(minor);
  }

  // Nondeterministically set control data
  NONDET {
#define OPC(id, mnemonic, opc, func3, func7, immFmt, aluA, aluB, aluOp, setPC, setRD, rdEn, next)  \
  if (id / kMinorMuxSize == major) {                                                               \
    IF(minorSelect->at(id % kMinorMuxSize)) {                                                      \
      control->set(decoder->imm##immFmt(), aluA, aluB, aluOp, next);                               \
    }                                                                                              \
  }
#include "zirgen/circuit/rv32im/v1/platform/rv32im.inl"
  }

  // Load RS1 + RS2
  U32Val rs1 = readRS1->doRead(cycle, kRegisterOffset - 32 * userMode + decoder->rs1());
  U32Val rs2 = readRS2->doRead(cycle, kRegisterOffset - 32 * userMode + decoder->rs2());

  // Apply ALU
  Val aluA = control->aluA;
  U32Val inA = (1 - aluA) * rs1 + aluA * (BACK(1, body->pc->getU32()) - U32Val({4, 0, 0, 0}));
  Val aluB = control->aluB;
  U32Val inB = (1 - aluB) * rs2 + aluB * control->imm->get();
  alu->set(inA, inB, control);
  XLOG("  imm=%w, rs1=x%u -> %w, rs2=x%u -> %w, inA = %w, inB = %w, ALU output = %w, EQ:%u, LT:%u, "
       "LTU:%u",
       control->imm,
       decoder->rs1(),
       rs1,
       decoder->rs2(),
       rs2,
       inA,
       inB,
       alu->getResult(),
       alu->getEQ(),
       alu->getLT(),
       alu->getLTU());

  // Make sure not to write when rd = x0
  rdZero->set(decoder->rd());

  // Compute various PC options + RD options
  Val PC4 = curPC + 4;
  Val PCIM = curPC + control->imm->getSmallSigned();
  U32Val RES = alu->getResult();
  U32Val LT = {alu->getLT(), 0, 0, 0};
  U32Val LTU = {alu->getLTU(), 0, 0, 0};
  U32Val XPC = BACK(1, body->pc->getU32());
  Val JMP = RES.bytes[0] + (RES.bytes[1] * (1 << 8)) + (RES.bytes[2] * (1 << 16)) +
            (RES.bytes[3] * (1 << 24));
  Val BEQ = alu->getEQ() * PCIM + (1 - alu->getEQ()) * PC4;
  Val BNE = alu->getEQ() * PC4 + (1 - alu->getEQ()) * PCIM;
  Val BLT = alu->getLT() * PCIM + (1 - alu->getLT()) * PC4;
  Val BGE = alu->getLT() * PC4 + (1 - alu->getLT()) * PCIM;
  Val BLTU = alu->getLTU() * PCIM + (1 - alu->getLTU()) * PC4;
  Val BGEU = alu->getLTU() * PC4 + (1 - alu->getLTU()) * PCIM;

  // Check opcode, verify things done ND earlier, set new pc, set nextMajor, and write to RD
#define OPC(id, mnemonic, opc, f3, f7, immFmt, aluA, aluB, aluOp, setPC, setRD, rdEn, next)        \
  if (id / kMinorMuxSize == major) {                                                               \
    IF(minorSelect->at(id % kMinorMuxSize)) {                                                      \
      eq(decoder->opcode(), opc * 4 + 3);                                                          \
      if (f3 != -1) {                                                                              \
        eq(decoder->func3(), f3);                                                                  \
      }                                                                                            \
      if (f7 != -1) {                                                                              \
        eq(decoder->func7(), f7);                                                                  \
      }                                                                                            \
      control->set(decoder->imm##immFmt(), aluA, aluB, aluOp, next);                               \
      body->pc->set(setPC);                                                                        \
      body->nextMajor->set(control->nextMajor);                                                    \
      IF(rdEn*(1 - rdZero->isZero())) {                                                            \
        XLOG("  Writing to rd=x%u, val = %w", decoder->rd(), setRD);                               \
        writeRD->doWrite(cycle, kRegisterOffset - 32 * userMode + decoder->rd(), setRD);           \
      }                                                                                            \
      IF((1 - rdEn) + rdZero->isZero()) { writeRD->doNOP(); }                                      \
    }                                                                                              \
  }
#include "zirgen/circuit/rv32im/v1/platform/rv32im.inl"
}

VerifyAndCycleImpl::VerifyAndCycleImpl(RamHeader ramHeader) : ram(ramHeader) {}

void VerifyAndCycleImpl::set(Top top) {
  auto body = top->mux->at<StepType::BODY>();
  eqz(BACK(1, body->nextMajor->get()) - MajorType::kVerifyAnd);
  auto compute = body->majorMux->at<0>()->inner;
  U32Val a = BACK(1, compute->readRS1->data());
  U32Val b = BACK(1, compute->alu->getInB());
  U32Val c = BACK(1, compute->alu->getAndVal());
  NONDET {
    for (size_t byte = 0; byte < 4; byte++) {
      for (size_t bit = 0; bit < 8; bit++) {
        aBits[byte * 8 + bit]->set((a.bytes[byte] & (1 << bit)) / (1 << bit));
        bBits[byte * 8 + bit]->set((b.bytes[byte] & (1 << bit)) / (1 << bit));
      }
    }
  }
  U32Val ax = {0, 0, 0, 0};
  U32Val bx = {0, 0, 0, 0};
  U32Val cx = {0, 0, 0, 0};
  for (size_t byte = 0; byte < 4; byte++) {
    for (size_t bit = 0; bit < 8; bit++) {
      Val aBit = aBits[byte * 8 + bit];
      Val bBit = bBits[byte * 8 + bit];
      ax.bytes[byte] = ax.bytes[byte] + aBit * (1 << bit);
      bx.bytes[byte] = bx.bytes[byte] + bBit * (1 << bit);
      cx.bytes[byte] = cx.bytes[byte] + aBit * bBit * (1 << bit);
    }
  }
  XLOG("  a = %w, ax = %w", a, ax);
  XLOG("  b = %w, bx = %w", b, bx);
  XLOG("  c = %w, cx = %w", c, cx);
  eq(a, ax);
  eq(b, bx);
  eq(c, cx);
  Val curPC = BACK(1, body->pc->get());
  body->pc->set(curPC);
  body->nextMajor->set(MajorType::kMuxSize);
}

} // namespace zirgen::rv32im_v1
