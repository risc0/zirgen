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

#include "zirgen/circuit/rv32im/v1/edsl/body.h"

#include "zirgen/circuit/rv32im/v1/edsl/top.h"

#include <deque>
#include <type_traits>

namespace zirgen::rv32im_v1 {

PCRegImpl::PCRegImpl() {
  for (size_t i = 0; i < 3; i++) {
    bytes.emplace_back(Label("bytes", i));
  }
  for (size_t i = 0; i < 2; i++) {
    twits.emplace_back(Label("twits", i));
  }
}

void PCRegImpl::set(Val val, size_t offset) {
  val = bytes[0]->set(val + offset);
  val = bytes[1]->set(val);
  val = bytes[2]->set(val);
  NONDET { twits[0]->set(val & 3); }
  twits[1]->set((val - twits[0]->get()) / 4);
  Val top2 = twits[1];
  // Prevent PC from ever entering high 1/4 of RAM (system RAM)
  // Need to buffer since this is called in deg=2 situtations
  buffer->set(top2 * (1 - top2));
  eqz(buffer * (2 - top2));
}

Val PCRegImpl::get() {
  return bytes[0] * (1 << 0) +  //
         bytes[1] * (1 << 8) +  //
         bytes[2] * (1 << 16) + //
         twits[0] * (1 << 24) + //
         twits[1] * (1 << 26) - 4;
}

U32Val PCRegImpl::getU32() {
  return {bytes[0], bytes[1], bytes[2], twits[0] + twits[1] * 4};
}

void PCRegImpl::checkValid(Val userMode) {
  Val top2 = twits[1];
  eqz(userMode * top2 * (1 - top2));
}

ResetStepImpl::ResetStepImpl(BytesHeader bytesHeader)
    : bytes(bytesHeader, kNumBodyBytes)
    , ramHeader(Label("header"), global->sysExitCode) // If sysExit == 0, don't check dirty
    , ram(ramHeader, kDigestWords / 2) {}

void ResetStepImpl::set(Top top) {
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  PageTableInfo info;
  Val first = top->code->stepInfo->at<StepType::RESET>()->isFirst;
  Val isInit = top->code->stepInfo->at<StepType::RESET>()->isInit;
  Val isOutput = top->code->stepInfo->at<StepType::RESET>()->isOutput;
  Val isFini = top->code->stepInfo->at<StepType::RESET>()->isFini;

  IF(isInit) {
    haltType->set(0);
    sysExitCode->set(0);

    IF(first) {
      for (size_t i = 0; i < kDigestWords / 2; i++) {
        imageIdWrites[i]->doPreLoad(
            cycle, info.rootAddr + i, global->pre->imageId->words[i]->get());
      }
    }
    IF(1 - first) {
      for (size_t i = 0; i < kDigestWords / 2; i++) {
        imageIdWrites[i]->doPreLoad(cycle,
                                    info.rootAddr + kDigestWords / 2 + i,
                                    global->pre->imageId->words[kDigestWords / 2 + i]->get());
      }
    }
    NONDET { userMode->set(global->pre->pc->get().bytes[0] & 1); }
    pc->set(global->pre->pc->get().flat() - userMode);
    // If usermode is set correctly PC low 2 bits should be 00
    verifyPC->set(pc->getU32().bytes[0] / 4);
  }
  IF(isOutput) {
    userMode->set(BACK(1, userMode->get()));
    haltType->set(0);
    verifyPC->set(0);

    pc->set(BACK(1, pc->get()));
    IF(first) {
      // Make sure we are in 'halted' state
      eq(BACK(1, body->majorSelect->get()), MajorType::kHalt);
      auto haltCycle = body->majorMux->at<MajorType::kHalt>();
      sysExitCode->set(BACK(1, haltCycle->sysExitCode->get()));

      Val addr = BACK(1, haltCycle->writeAddr->get());

      for (size_t i = 0; i < kDigestWords / 2; i++) {
        U32Val word = imageIdWrites[i]->doRead(cycle, addr + i, MemoryOpType::kPageIo);
        global->output->words[i]->set(word);
      }
    }
    IF(1 - first) {
      auto haltCycle = body->majorMux->at<MajorType::kHalt>();
      Val addr = BACK(2, haltCycle->writeAddr->get());
      sysExitCode->set(BACK(1, sysExitCode->get()));

      for (size_t i = 0; i < kDigestWords / 2; i++) {
        U32Val word =
            imageIdWrites[i]->doRead(cycle, addr + kDigestWords / 2 + i, MemoryOpType::kPageIo);
        global->output->words[kDigestWords / 2 + i]->set(word);
      }
    }
  }
  IF(isFini) {
    userMode->set(BACK(1, userMode->get()));
    sysExitCode->set(BACK(1, sysExitCode->get()));
    haltType->set(sysExitCode->get());
    verifyPC->set(0);

    // This toggle is used to set the post state imageId to 0 in the case of a halt/terminate.
    Val enablePostState = 1 - haltType->at(HaltType::kTerminate);

    IF(first) {
      // Set offset to 0 here to prevent adding default +4 offset to final value written to globals.
      pc->set(BACK(1, pc->get()), /*offset=*/0);

      for (size_t i = 0; i < kDigestWords / 2; i++) {
        U32Val word = imageIdWrites[i]->doRead(cycle, info.rootAddr + i, MemoryOpType::kPageIo);
        global->post->imageId->words[i]->setWithFactor(word, enablePostState);
      }
      U32Val pcAndUserMode = pc->getU32() + U32Val(BACK(1, userMode->get()), 0, 0, 0);
      global->post->pc->setWithFactor(pcAndUserMode, enablePostState);
    }
    IF(1 - first) {
      pc->set(BACK(1, pc->get()));

      for (size_t i = 0; i < kDigestWords / 2; i++) {
        U32Val word = imageIdWrites[i]->doRead(
            cycle, info.rootAddr + kDigestWords / 2 + i, MemoryOpType::kPageIo);
        global->post->imageId->words[kDigestWords / 2 + i]->setWithFactor(word, enablePostState);
      }
    }
  }
  nextMajor->set(MajorType::kMuxSize);
  XLOG("%u: Reset: PC = %10x", cycle, pc);
}

HaltCycleImpl::HaltCycleImpl(RamHeader ramHeader) : ram(ramHeader) {}

void HaltCycleImpl::set(Top top) {
  BodyStep body = top->mux->at<StepType::BODY>();
  Val curPC = BACK(1, body->pc->get());

  eqz(BACK(1, body->nextMajor->get()) - MajorType::kHalt);
  Val isHalt = BACK(1, body->majorSelect->at(MajorType::kHalt));
  IF(isHalt) {
    sysExitCode->set(BACK(1, sysExitCode->get()));
    userExitCode->set(BACK(1, userExitCode->get()));
    writeAddr->set(BACK(1, writeAddr->get()));
  }

  Val isFromEcall = BACK(1, body->majorSelect->at(MajorType::kECall));
  IF(isFromEcall) {
    ECallCycle ecall = body->majorMux->at<MajorType::kECall>();
    Val isFromEcallHalt = BACK(1, ecall->minorSelect->at(ECallType::kHalt));
    eq(1, isFromEcallHalt);
    auto ecallPrev = ecall->minorMux->at<ECallType::kHalt>();
    Val sysCode = BACK(1, ecallPrev->readA0->data().bytes[0]);
    Val userCode = BACK(1, ecallPrev->readA0->data().bytes[1]);
    writeAddr->set(BACK(1, ecallPrev->writeAddr->get()));
    sysExitCode->set(sysCode);
    userExitCode->set(userCode);
    XLOG("isFromEcall, set sysExitCode: %u", sysCode);
    body->global->sysExitCode->set(sysCode);
    body->global->userExitCode->set(userCode);

    // Notify host of halt
    NONDET { doExtern("halt", "", 0, {sysCode, curPC}); }
  }

  Val isFromPageFault = BACK(1, body->majorSelect->at(MajorType::kPageFault));
  IF(isFromPageFault) {
    sysExitCode->set(HaltType::kSystemSplit);
    userExitCode->set(0);
    writeAddr->set(kZerosOffset);
    XLOG("isFromPageFault, set sysExitCode: %u", HaltType::kSystemSplit);
    body->global->sysExitCode->set(HaltType::kSystemSplit);
    body->global->userExitCode->set(0);

    // Notify host of halt
    NONDET { doExtern("halt", "", 0, {HaltType::kSystemSplit, curPC}); }
  }

  body->pc->set(curPC);
  body->nextMajor->set(MajorType::kHalt);
}

BodyStepImpl::BodyStepImpl(BytesHeader bytesHeader)
    : global(Label("global"))
    , bytes(bytesHeader, kNumBodyBytes)
    , ramHeader(Label("header"), global->sysExitCode) // If sysExit == 0, don't check dirty
    , majorMux(Label("major_mux"),
               Labels({"compute0",
                       "compute1",
                       "compute2",
                       "mem_io",
                       "multiply",
                       "divide",
                       "verify_and",
                       "verify_divide",
                       "ecall",
                       "sha_init",
                       "sha_load",
                       "sha_main",
                       "page_fault",
                       "ecall_copy_in",
                       "big_int",
                       "big_int2",
                       "halt"}),
               majorSelect,
               ramHeader) {}

void BodyStepImpl::set(Top top) {
  Val cycle = top->code->cycle;
  Val curPC = BACK(1, pc->get());
  // Pick out major cycle type using non-det
  NONDET {
    Val prevMajor = BACK(1, nextMajor->get());
    Val isDecode = isz(prevMajor - MajorType::kMuxSize);
    IF(isDecode) {
      XLOG("%u: BODY pc: %10x", cycle, curPC);
      doExtern("trace", "", 0, {curPC});
      Val major = doExtern("getMajor", "", 1, {cycle, curPC})[0];
      majorSelect->set(major);
    }
    IF(1 - isDecode) {
      IF(1 - isz(prevMajor - MajorType::kHalt)) {
        XLOG("%u: BODY pc: %10x, major = %u", cycle, curPC, prevMajor);
      }
      majorSelect->set(prevMajor);
    }
  }
  majorMux->doMux([&](auto inner) {
    if (!std::is_same<decltype(inner), ECallCycle>::value) {
      userMode->set(BACK(1, userMode->get()));
    }
    inner->set(top);
    pc->checkValid(userMode);
  });
}

} // namespace zirgen::rv32im_v1
