// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/rv32im/v1/edsl/top.h"

namespace zirgen::rv32im_v1 {

BytesInitStepImpl::BytesInitStepImpl(BytesHeader bytesHeader) : bytes(bytesHeader) {}

void BytesInitStepImpl::set(Top top) {}

BytesSetupStepImpl::BytesSetupStepImpl(BytesHeader bytesHeader)
    : bytes(bytesHeader, kSetupStepRegs) {}

void BytesSetupStepImpl::set(Top top) {
  Val isFirst = 1 - BACK(1, top->code->stepType->at(StepType::BYTES_SETUP));
  Val isLast = top->code->stepInfo->at<StepType::BYTES_SETUP>()->isLastSetup;
  bytes->set(isFirst, isLast);
}

RamInitStepImpl::RamInitStepImpl(BytesHeader bytesHeader)
    : bytes(bytesHeader, kNumBodyBytes)
    , ramHeader(Label("header"), global->sysExitCode)
    , ram(ramHeader) {}

void RamInitStepImpl::set(Top top) {}

RamLoadStepImpl::RamLoadStepImpl(BytesHeader bytesHeader)
    : bytes(bytesHeader, kNumBodyBytes)
    , ramHeader(Label("header"), global->sysExitCode)
    , ram(ramHeader, 4) {}

void RamLoadStepImpl::set(Top top) {
  RamLoadInfo info = top->code->stepInfo->at<StepType::RAM_LOAD>();
  for (size_t i = 0; i < kRamLoadStepIOCount; i++) {
    // Decode shorts into bytes, this technically breaks the U32 abstraction
    decode[i * 4 + 1]->setExact(decode[i * 4 + 0]->set(info->data[i * 2]));
    decode[i * 4 + 3]->setExact(decode[i * 4 + 2]->set(info->data[i * 2 + 1]));
    // Write data into memory
    U32Val val(decode[i * 4 + 0], //
               decode[i * 4 + 1],
               decode[i * 4 + 2],
               decode[i * 4 + 3]);
    writes[i]->doPreLoad(top->code->cycle, info->startAddr + i, val);
  }
}

RamFiniStepImpl::RamFiniStepImpl(BytesHeader bytesHeader)
    : bytes(bytesHeader, kNumBodyBytes)
    , ramHeader(Label("header"), global->sysExitCode)
    , ram(ramHeader) {}

void RamFiniStepImpl::set(Top top) {
  XLOG("%u: RamFini", top->code->cycle);
}

BytesFiniStepImpl::BytesFiniStepImpl(BytesHeader bytesHeader) : bytes(bytesHeader) {}

void BytesFiniStepImpl::set(Top top) {
  XLOG("%u: BytesFini", top->code->cycle);
}

TopImpl::TopImpl()
    : halted(Label("halted"))
    , mux(Label("mux"),
          Labels({"bytes_init",
                  "bytes_setup",
                  "ram_init",
                  "ram_load",
                  "reset",
                  "body",
                  "ram_fini",
                  "bytes_fini"}),
          code->stepType,
          bytesHeader) {}

Val TopImpl::set() {
  // Do the actual execution
  mux->doMux([&](auto inner) { inner->set(asComp()); });

  Val isActive = 0;
  for (size_t i = 0; i < StepType::COUNT; i++) {
    isActive = isActive + code->stepType->at(i);
  }
  // Compute halt state.  Note if the mux values aren't the right
  // types that 'casts' are likely to return nonsense, but it's no harm
  // since the output will be multiplied by zero.
  Val isBody = code->stepType->at(StepType::BODY);
  IF(isBody) {
    BodyStep body = mux->at<StepType::BODY>();
    Val isHalt = body->majorSelect->at(MajorType::kHalt);
    halted->set(isHalt);
  }
  IF(isActive - isBody) { halted->set(0); }
  return 1 - halted;
}

} // namespace zirgen::rv32im_v1
