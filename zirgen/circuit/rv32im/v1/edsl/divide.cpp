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

DivideCycleImpl::DivideCycleImpl(RamHeader ramHeader) : ram(ramHeader, 4) {}

void DivideCycleImpl::set(Top top) {
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

  // Set control data
  Val useImm = 0;
  Val usePo2 = 0;
  Val isSignedVal = 0;
  Val onesCompVal = 0;
  Val useRem = 0;
#define OPD(id, mnemonic, opc, f3, f7, immFmt, useImm_, usePo2_, signed_, onesComp_, useRem_)      \
  {                                                                                                \
    Val match = minorSelect->at(id % kMinorMuxSize);                                               \
    useImm = useImm + useImm_ * match;                                                             \
    usePo2 = usePo2 + usePo2_ * match;                                                             \
    isSignedVal = isSignedVal + signed_ * match;                                                   \
    onesCompVal = onesCompVal + onesComp_ * match;                                                 \
    useRem = useRem + useRem_ * match;                                                             \
  }
#include "zirgen/circuit/rv32im/v1/platform/rv32im.inl"
  isSigned->set(isSignedVal);
  onesComp->set(onesCompVal);
  XLOG("  useImm=%u, usePo2=%u, signed=%u, onesComp=%u, useRem=%u",
       useImm,
       usePo2,
       isSigned->get(),
       onesComp->get(),
       useRem);

  // Load RS1 + RS2
  U32Val rs1 = readRS1->doRead(cycle, kRegisterOffset - 32 * userMode + decoder->rs1());
  U32Val rs2 = readRS2->doRead(cycle, kRegisterOffset - 32 * userMode + decoder->rs2());
  XLOG("  rs1=x%u -> %w, rs2=x%u -> %w", decoder->rs1(), rs1, decoder->rs2(), rs2);

  U32Val imm = decoder->immI();

  // Use nondet to setup PO2 + and verify
  U32Val po2Src = useImm * imm + (1 - useImm) * rs2;
  NONDET {
    top2->set((po2Src.bytes[0] & 0xc0) / 0x40);
    bit6->set((po2Src.bytes[0] & 0x20) / 0x20);
    po2->set(po2Src.bytes[0] & 0x1f);
  }
  eq(po2Src.bytes[0], top2 * 0x40 + bit6 * 0x20 + po2->get());

  // Pick the input for the denominator and assign it
  U32Val inB = usePo2 * po2->getPo2() + (1 - usePo2) * rs2;
  for (size_t i = 0; i < 4; i++) {
    denom[i]->setExact(inB.bytes[i]);
  }
  // Now, run the nondet division computation
  NONDET {
    std::vector<Val> input(9);
    for (size_t i = 0; i < 4; i++) {
      input[i] = rs1.bytes[i];
      input[4 + i] = denom[i];
    }
    input[8] = isSigned + onesComp;
    auto output = doExtern("divide", "", 8, input);
    for (size_t i = 0; i < 4; i++) {
      quot[i]->setExact(output[i]);
      rem[i]->setExact(output[4 + i]);
    }
  }
  XLOG("  numer=%w, denom=%w, quot=%w, rem=%w", rs1, denom, quot, rem);

  // Set output
  rdZero->set(decoder->rd());
  IF(useRem * (1 - rdZero->isZero())) {
    writeRd->doWrite(cycle,
                     kRegisterOffset - 32 * userMode + decoder->rd(),
                     U32Val({rem[0], rem[1], rem[2], rem[3]}));
  }
  IF((1 - useRem) * (1 - rdZero->isZero())) {
    writeRd->doWrite(cycle,
                     kRegisterOffset - 32 * userMode + decoder->rd(),
                     U32Val({quot[0], quot[1], quot[2], quot[3]}));
  }
  IF(rdZero->isZero()) { writeRd->doNOP(); }

  // Prepare next cycle
  body->pc->set(curPC + 4);
  body->nextMajor->set(MajorType::kVerifyDivide);

  // Verify decoding
#define OPD(id, mnemonic, opc, f3, f7, immFmt, useImm_, usePo2_, signedA_, signedB_, useHigh_)     \
  IF(minorSelect->at(id % kMinorMuxSize)) {                                                        \
    eq(decoder->opcode(), opc * 4 + 3);                                                            \
    if (f3 != -1) {                                                                                \
      eq(decoder->func3(), f3);                                                                    \
    }                                                                                              \
    if (f7 != -1) {                                                                                \
      eq(decoder->func7(), f7);                                                                    \
    }                                                                                              \
  }
#include "zirgen/circuit/rv32im/v1/platform/rv32im.inl"
}

VerifyDivideCycleImpl::VerifyDivideCycleImpl(RamHeader ramHeader) : ram(ramHeader) {}

void VerifyDivideCycleImpl::set(Top top) {
  BodyStep body = top->mux->at<StepType::BODY>();
  eqz(BACK(1, body->nextMajor->get()) - MajorType::kVerifyDivide);
  Val curPC = BACK(1, body->pc->get());
  auto fromRegs = [](std::array<ByteReg, 4> bytes) {
    return U32Val({bytes[0], bytes[1], bytes[2], bytes[3]});
  };
  auto divCycle = body->majorMux->at<MajorType::kDivide>();
  U32Val numer = BACK(1, divCycle->readRS1->data());
  U32Val denom = BACK(1, fromRegs(divCycle->denom));
  U32Val quot = BACK(1, fromRegs(divCycle->quot));
  U32Val rem = BACK(1, fromRegs(divCycle->rem));
  Val isSigned = BACK(1, divCycle->isSigned->get());
  Val onesComp = BACK(1, divCycle->onesComp->get());
  numerTop->set(numer);
  denomTop->set(denom);
  negNumer->set(isSigned * numerTop->getHighBit());
  negDenom->set(isSigned * (1 - onesComp) * denomTop->getHighBit());
  U32Val one = {1, 0, 0, 0};
  numerAbs->set(U32Val::underflowProtect() + (1 - negNumer) * numer - negNumer * numer -
                negNumer * onesComp * one);
  XLOG("  numer = %w, numerAbs = %w", numer, numerAbs->getNormed());
  denomAbs->set(U32Val::underflowProtect() + (1 - negDenom) * denom - negDenom * denom -
                negDenom * onesComp * one);
  XLOG("  demom = %w, denomAbs = %w", denom, denomAbs->getNormed());
  denomZero->set(denomAbs->getNormed());
  negQuot->set(negNumer + negDenom - 2 * negNumer * negDenom - denomZero->isZero() * negNumer);
  Val negRem = negNumer;
  quotAbs->set(U32Val::underflowProtect() + (1 - negQuot) * quot - negQuot * quot -
               negQuot * onesComp * one);
  XLOG("  quot = %w, quotAbs = %w", quot, quotAbs->getNormed());
  remAbs->set(U32Val::underflowProtect() + (1 - negRem) * rem - negRem * rem -
              negRem * onesComp * one);
  XLOG("  rem = %w, remAbs = %w", rem, remAbs->getNormed());
  denomRemCheck->set(U32Val::underflowProtect() + denomAbs->getNormed() - one -
                     remAbs->getNormed());
  mul->set(quotAbs->getNormed(), denomAbs->getNormed(), remAbs->getNormed());
  XLOG("  mul->getOut() = %w, denomRemCheck->carry = %u", mul->getOut(), denomRemCheck->getCarry());
  eq(mul->getOut(), numerAbs->getNormed());
  IF(1 - denomZero->isZero()) { eq(denomRemCheck->getCarry(), 1); }
  IF(denomZero->isZero()) {
    eq(rem, numer);
    eq(quot, U32Val(0xff, 0xff, 0xff, 0xff));
  }
  body->pc->set(curPC);
  body->nextMajor->set(MajorType::kMuxSize);
}

} // namespace zirgen::rv32im_v1
