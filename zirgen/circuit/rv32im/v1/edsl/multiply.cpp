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

MultiplyCycleImpl::MultiplyCycleImpl(RamHeader ramHeader) : ram(ramHeader, 4) {}

void MultiplyCycleImpl::set(Top top) {
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
  Val signedA = 0;
  Val signedB = 0;
  Val useHigh = 0;
#define OPM(id, mnemonic, opc, f3, f7, immFmt, useImm_, usePo2_, signedA_, signedB_, useHigh_)     \
  {                                                                                                \
    Val match = minorSelect->at(id % kMinorMuxSize);                                               \
    useImm = useImm + useImm_ * match;                                                             \
    usePo2 = usePo2 + usePo2_ * match;                                                             \
    signedA = signedA + signedA_ * match;                                                          \
    signedB = signedB + signedB_ * match;                                                          \
    useHigh = useHigh + useHigh_ * match;                                                          \
  }
  XLOG("  useImm=%u, usePo2=%u, signedA=%u, signedB=%u, useHigh=%u",
       useImm,
       usePo2,
       signedA,
       signedB,
       useHigh);
#include "zirgen/circuit/rv32im/v1/platform/rv32im.inl"

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

  // Pick the input
  U32Val inB = usePo2 * po2->getPo2() + (1 - usePo2) * rs2;

  // Run the multipler
  mul->set(rs1, inB, signedA, signedB);

  // Make sure not to write when rd = x0
  rdZero->set(decoder->rd());

  body->pc->set(curPC + 4);
  body->nextMajor->set(MajorType::kMuxSize);
  IF(useHigh * (1 - rdZero->isZero())) {
    writeRd->doWrite(cycle, kRegisterOffset - 32 * userMode + decoder->rd(), mul->getHigh());
  }
  IF((1 - useHigh) * (1 - rdZero->isZero())) {
    writeRd->doWrite(cycle, kRegisterOffset - 32 * userMode + decoder->rd(), mul->getLow());
  }
  IF(rdZero->isZero()) { writeRd->doNOP(); }

  // Verify decoding
#define OPM(id, mnemonic, opc, f3, f7, immFmt, useImm_, usePo2_, signedA_, signedB_, useHigh_)     \
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

} // namespace zirgen::rv32im_v1
