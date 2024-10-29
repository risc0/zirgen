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

#include "zirgen/circuit/rv32im/v1/edsl/top.h"

namespace zirgen::rv32im_v1 {

void ECallHaltImpl::set(Top top) {
  // Get common objects
  Val cycle = top->code->cycle;
  auto body = top->mux->at<StepType::BODY>();
  auto ecall = body->majorMux->at<MajorType::kECall>();
  Val curPC = BACK(1, body->pc->get());

  // Read the address to read form, verify both the low and high word
  // are in the user accessible region of ram
  U32Val ramAddr = readA1->doRead(cycle, RegAddr::kA1);
  writeAddr->set(ramAddr.flat() / 4);

  // Read the exit code data (halt/pause, user code)
  U32Val exitCode = readA0->doRead(cycle, RegAddr::kA0);
  XLOG("ECallHalt> exitCode = %w, ramAddr = %w", exitCode, ramAddr);

  body->pc->set(curPC + 4);
  body->userMode->set(BACK(1, body->userMode->get()));
  body->nextMajor->set(MajorType::kHalt);
}

void ECallInputImpl::set(Top top) {
  // Get common objects
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  ECallCycle ecall = body->majorMux->at<MajorType::kECall>();
  Val curPC = BACK(1, body->pc->get());

  // Read input selector from a0 = x10
  U32Val value = readA0->doRead(cycle, RegAddr::kA0);
  selector->set(value.bytes[0]);

  for (size_t i = 0; i < kDigestWords; i++) {
    IF(selector->at(i)) {
      word->set(body->global->input->words[i]->get());
      XLOG("  Read from %u: %w", selector, word->get());
    }
  }

  // Write word to a0 = x10
  writeA0->doWrite(cycle, RegAddr::kA0, word->get());

  // Prep for next cycle
  body->pc->set(curPC + 4);
  body->userMode->set(BACK(1, body->userMode->get()));
  body->nextMajor->set(MajorType::kMuxSize);
}

// A software ecall potentially receives data from the host, so common to all SOFTWARE ecalls, here
// are the assigned registers:
//
// t0: ecall:::SOFTWARE (2)
// a0: Output address of system call; the data returned from the host will start writing here.
//     Must be word aligned.
// a1: Number of 4-word chunks (16 bytes) to read
//
// The syscall args start at a2, the system call id.  However, we don't pass those to the host;
// instead, the host reads those from the guest memory state.
//
// The host may optionally supply return values in a0 and a1.
void ECallSoftwareImpl::set(Top top) {
  // Get common objects
  Val cycle = top->code->cycle;
  auto body = top->mux->at<StepType::BODY>();
  auto ecall = body->majorMux->at<MajorType::kECall>();
  Val curPC = BACK(1, body->pc->get());

  readOutputAddr->doRead(cycle, RegAddr::kA0);
  readOutputWords->doRead(cycle, RegAddr::kA1);

  NONDET {
    Val totWords = readOutputWords->data().flat();
    Val strayWords = totWords & (kIoChunkWords - 1);
    IF(strayWords) {
      outputChunks->set((totWords - strayWords + kIoChunkWords) / kIoChunkWords);
      outputFirstCycleWordsMinusOne->set(strayWords - 1);
    }
    IF(isz(strayWords)) {
      outputChunks->set(totWords / kIoChunkWords);
      outputFirstCycleWordsMinusOne->set(kIoChunkWords - 1);
    }
  }

  XLOG("Calculated orig = %u, output chunks = %u, first cycle-1 = %u",
       readOutputWords->data().flat(),
       outputChunks,
       outputFirstCycleWordsMinusOne);
  eq(readOutputWords->data().flat(),
     (outputChunks - 1) * kIoChunkWords + (outputFirstCycleWordsMinusOne + 1));

  // If these are not word-aligned, dividing by 4 will be too large to fit in a byte.
  eqz(requireAlignedAddr->set(readOutputAddr->data().bytes[0] / 4));
  eqz(requireAlignedBytes->set(readOutputAddr->data().bytes[0] / 4));

  NONDET { doExtern("syscallInit", "", 0, {readOutputWords->data().flat()}); }

  body->pc->set(curPC);
  body->userMode->set(BACK(1, body->userMode->get()));
  body->nextMajor->set(MajorType::kECallCopyIn);
  XLOG("Set next major to be ecall copy in");
}

void ECallShaImpl::set(Top top) {
  // Get common objects
  Val cycle = top->code->cycle;
  auto body = top->mux->at<StepType::BODY>();
  auto ecall = body->majorMux->at<MajorType::kECall>();
  Val curPC = BACK(1, body->pc->get());
  // Read data
  readA0->doRead(cycle, RegAddr::kA0);
  readA1->doRead(cycle, RegAddr::kA1);
  readA4->doRead(cycle, RegAddr::kA4);
  // Prep for next cycle
  body->pc->set(curPC + 4);
  body->userMode->set(BACK(1, body->userMode->get()));
  body->nextMajor->set(MajorType::kShaInit);
}

void ECallBigIntImpl::set(Top top) {
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  Val curPC = BACK(1, body->pc->get());

  readA1->doRead(cycle, RegAddr::kA1);

  // Keep the current PC and set the next major cycle to BigInt.
  body->pc->set(curPC);
  body->userMode->set(BACK(1, body->userMode->get()));
  body->nextMajor->set(MajorType::kBigInt);
}

void ECallBigInt2Impl::set(Top top) {
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  Val curPC = BACK(1, body->pc->get());

  // TODO: Verify it's a valid address
  readVerifyAddr->doRead(cycle, RegAddr::kT2);

  // Keep the current PC and set the next major cycle to BigInt.
  body->pc->set(curPC);
  body->userMode->set(BACK(1, body->userMode->get()));
  body->nextMajor->set(MajorType::kBigInt2);
}

void ECallUserImpl::set(Top top) {
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  eqz(BACK(1, body->userMode->get()));
  // Load the user PC + jump to
  U32Val newPC = readPC->doRead(cycle, kUserPC / 4);
  body->pc->set(newPC.flat());
  body->userMode->set(1);
  NONDET { doExtern("setUserMode", "", 0, {1}); }
  body->nextMajor->set(MajorType::kMuxSize);
}

void ECallMachineImpl::set(Top top) {
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  auto ecall = body->majorMux->at<MajorType::kECall>();
  eqz(1 - BACK(1, body->userMode->get()));
  // Store the current PC
  writePC->doWrite(cycle, kUserPC / 4, BACK(1, body->pc->getU32()));
  // Pick which entry to load (normal ecall or trap)
  Val newEntry = ecall->isTrap * (kTrapEntry / 4) + (1 - ecall->isTrap) * (kECallEntry / 4);
  // Load entry point and jump to it
  U32Val newPC = readEntry->doRead(cycle, newEntry);
  body->pc->set(newPC.flat());
  body->userMode->set(0);
  NONDET { doExtern("setUserMode", "", 0, {0}); }
  body->nextMajor->set(MajorType::kMuxSize);
}

ECallCycleImpl::ECallCycleImpl(RamHeader ramHeader) : ram(ramHeader, 5), minorMux(minorSelect) {}

void ECallCycleImpl::set(Top top) {
  // Load some things
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  Val curPC = BACK(1, body->pc->get());
  Val userMode = BACK(1, body->userMode->get());
  // Load isTrap nondeterministically
  NONDET { isTrap->set(doExtern("isTrap", "", 1, {})[0]); }
  // isTrap can only be set when in user mode
  eqz((1 - userMode) * isTrap);
  // We should always be in 'decode'
  eqz(BACK(1, body->nextMajor->get()) - MajorType::kMuxSize);
  IF(isTrap) {
    // In a trap, selector is machine mode
    minorSelect->set(ECallType::kMuxSize);
  }
  IF(1 - isTrap) {
    // Read instruction
    U32Val inst = readInst->doRead(cycle, curPC / 4);
    // Verify it's an ecall
    eq(inst.bytes[0], 0b1110011);
    eqz(inst.bytes[1]);
    eqz(inst.bytes[2]);
    eqz(inst.bytes[3]);
    // Read the selector from t0 = x5
    U32Val selector = readSelector->doRead(cycle, RegAddr::kT0);
    // Use top byte for minor select, except if in user mode, we always go to machine mode
    Val select = (1 - userMode) * selector.bytes[0] + userMode * ECallType::kMuxSize;
    minorSelect->set(select);
  }
  // Print if it's not in a halt loop
  NONDET {
    IF(1 - isz(minorSelect - ECallType::kHalt)) { XLOG("  ecall, selector = %u", minorSelect); }
  }
  // Call into selected code
  minorMux->doMux([&](auto inner) { inner->set(top); });
}

TwitByteRegImpl::TwitByteRegImpl() {}

void TwitByteRegImpl::set(Val val) {
  Val check = 0;
  for (size_t i = 0; i < 4; i++) {
    uint32_t po2 = 1 << (2 * i);
    uint32_t mask = 3 * po2;
    NONDET { twits[i]->set((val & mask) / po2); }
    check = check + twits[i]->get() * po2;
  }
  eq(check, val);
}

Val TwitByteRegImpl::get() {
  Val ret = 0;
  for (size_t i = 0; i < 4; i++) {
    uint32_t po2 = 1 << (2 * i);
    ret = ret + twits[i]->get() * po2;
  }
  return ret;
}

ECallCopyInCycleImpl::ECallCopyInCycleImpl(RamHeader ramHeader) : ram(ramHeader, 4) {}

void ECallCopyInCycleImpl::set(Top top) {
  BodyStep body = top->mux->at<StepType::BODY>();
  Val curPC = BACK(1, body->pc->get());
  Val cycle = top->code->cycle;

  eqz(BACK(1, body->nextMajor->get()) - MajorType::kECallCopyIn);
  Val isFirstCycle = BACK(1, body->majorSelect->at(MajorType::kECall));
  IF(isFirstCycle) {
    ECallCycle ecall = body->majorMux->at<MajorType::kECall>();
    ECallSoftware software = ecall->minorMux->at<ECallType::kSoftware>();

    // If we're only filling in two words [c,d], outputAddr points to
    // [a, b, c, d] so it's easy to calculate the next outputAddr in
    // the non-first-cycle case.
    outputAddr->set(BACK(1,
                         software->readOutputAddr->data().flat() / 4 +
                             (software->outputFirstCycleWordsMinusOne->get() + 1) - kIoChunkWords));
    chunksRemaining->set(BACK(1, software->outputChunks->get()));
  }

  IF(1 - isFirstCycle) {
    chunksRemaining->set(BACK(1, chunksRemaining->get()) - 1);
    outputAddr->set(BACK(1, outputAddr->get()) + 4);
  }

  chunksRemainingZ->set(chunksRemaining);

  IF(isFirstCycle) {
    ECallCycle ecall = body->majorMux->at<MajorType::kECall>();
    ECallSoftware software = ecall->minorMux->at<ECallType::kSoftware>();
    outputWords->set(BACK(1, software->outputFirstCycleWordsMinusOne->get() + 1) *
                     (1 - chunksRemainingZ->isZero()));
    XLOG("  COPYIN INIT: dest=%x, remaining=%u first chunk words=%u",
         outputAddr * 4,
         chunksRemaining,
         outputWords);
  }

  IF(1 - isFirstCycle) { outputWords->set(kIoChunkWords * (1 - chunksRemainingZ->isZero())); }

  IF(outputWords->at(0)) {
    // All done!

    // Fill in a0 and a1 from system call return
    NONDET {
      std::vector<Val> ret = doExtern("syscallFini", "", 8, {});
      io[0]->doWrite(cycle, RegAddr::kA0, U32Val(ret[0], ret[1], ret[2], ret[3]));
      io[1]->doWrite(cycle, RegAddr::kA1, U32Val(ret[4], ret[5], ret[6], ret[7]));
    }
    eq(io[0]->cycle(), cycle);
    eq(io[1]->cycle(), cycle);

    eq(io[0]->addr(), RegAddr::kA0);
    eq(io[1]->addr(), RegAddr::kA1);

    XLOG("  COPYIN FINI: a0=%w a1=%w", io[0]->data(), io[1]->data());

    io[2]->doNOP();
    io[3]->doNOP();

    body->pc->set(curPC + 4);
    body->nextMajor->set(MajorType::kMuxSize);
  }
  IF(1 - outputWords->at(0)) {
    body->pc->set(curPC);
    body->nextMajor->set(MajorType::kECallCopyIn);
  }
  body->userMode->set(BACK(1, body->userMode->get()));

  for (size_t ioReg = 0; ioReg < kIoChunkWords; ++ioReg) {
    Val useThisReg = 0;
    Val nopThisReg = 0;
    for (size_t nwords = 1; nwords <= kIoChunkWords; ++nwords) {
      if (nwords > (kIoChunkWords - 1 - ioReg)) {
        useThisReg = useThisReg + outputWords->at(nwords);
      } else {
        nopThisReg = nopThisReg + outputWords->at(nwords);
      }
    }

    IF(useThisReg) {
      Val writeAddr = outputAddr + ioReg;
      NONDET {
        std::vector<Val> ret = doExtern("syscallBody", "", 4, {});
        io[ioReg]->doWrite(cycle, writeAddr, U32Val(ret[0], ret[1], ret[2], ret[3]));
        XLOG("  COPYIN BODY: %x <- %w", writeAddr * 4, io[ioReg]->data());
      }

      eq(io[ioReg]->cycle(), cycle);
      eq(io[ioReg]->addr(), writeAddr);
    }
    IF(nopThisReg) { io[ioReg]->doNOP(); }
    // Verify input was bytes
    for (size_t i = 0; i < 4; i++) {
      size_t byteId = ioReg * 4 + i;
      Val byte = io[ioReg]->data().bytes[i];
      if (byteId < 14) {
        checkBytes[byteId]->set(byte);
      } else {
        checkBytesTwits[byteId - 14]->set(byte);
      }
    }
  }
}

} // namespace zirgen::rv32im_v1
