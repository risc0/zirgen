// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/rv32im/v1/edsl/page_fault.h"

#include "zirgen/circuit/rv32im/v1/edsl/top.h"
#include "zirgen/compiler/zkp/util.h"
#include "llvm/Support/Format.h"

namespace zirgen::rv32im_v1 {

Val getPageAddr(Val pageIndex) {
  return pageIndex * kPageSize / kWordSize;
}

Val getPageIndex(Val addr) {
  return addr * kWordSize / kPageSize;
}

void PageIndexDiffReg::set(Val diff) {
  high->set(mid->set(low->set(diff)));
}

void PageIndexDiffReg::setZero() {
  low->setExact(0);
  mid->setExact(0);
  high->set(0);
}

PageFaultCycleImpl::PageFaultCycleImpl(RamHeader ramHeader) : ram(ramHeader) {}

void PageFaultCycleImpl::set(Top top) {
  BodyStep body = top->mux->at<StepType::BODY>();
  Val curPC = BACK(1, body->pc->get());
  body->pc->set(curPC);

  XLOG("  PageFault: PC = %10x", curPC);

  NONDET {
    std::vector<Val> ret = doExtern("pageInfo", "", 3, {curPC});
    isRead->set(ret[0]);
    pageIndex->set(ret[1]);
    isDone->set(ret[2]);
  }

  PageTableInfo info;

  isRootIndex->set(pageIndex - info.rootIndex);
  IF(isRootIndex->isZero()) {
    repeat->set(info.numRootEntries / 2);
    indexOffset->set((info.rootAddr - info.lastAddr) / kDigestWords);
  }
  IF(1 - isRootIndex->isZero()) {
    repeat->set(kPageSize / (kBlockSize * kWordSize));
    indexOffset->set(0);
  }

  Val pageTableIndex = pageIndex + indexOffset;
  Val entryAddr = kPageTableAddr + pageTableIndex * kDigestWords;
  stateOut->set(entryAddr);
  stateIn->set(kShaInitOffset);

  IF(1 - isDone) {
    // Disallow the first 'null' page
    Val lowerBound = 1;
    Val upperBound = info.rootIndex + 1;

    // for any range `[start, end)`
    //   compute `x - start` and show it can be decomposed into a low bit value
    //   compute `end - x - 1` and show it can be decomposed into a low bit value
    //   If x >= end, upperBound will underflow and you'll get a big number
    //   if x < start, lowerBound will underflow and you'll get a big number
    ltBound.set(pageIndex - lowerBound);
    gtBound.set(upperBound - 1 - pageIndex);

    body->nextMajor->set(MajorType::kShaInit);
  }

  IF(isDone) {
    ltBound.set(0);
    gtBound.set(0);
    body->nextMajor->set(MajorType::kHalt);
  }
}

} // namespace zirgen::rv32im_v1
