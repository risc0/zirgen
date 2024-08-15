// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/components/bytes.h"

namespace zirgen {

namespace impl {

std::vector<Val> BytesPlonkElementImpl::toVals() {
  return {high, low};
}

void BytesPlonkElementImpl::setFromVals(std::vector<Val> vals) {
  high->set(vals[0]);
  low->set(vals[1]);
}

void BytesPlonkElementImpl::setInit() {
  high->set(0);
  low->set(0);
}

void BytesPlonkElementImpl::setFini() {
  high->set(255);
  low->set(255);
}

void BytesPlonkElementImpl::setNext(std::vector<Val> vals) {
  Val oldHigh = vals[0];
  Val oldLow = vals[1];
  NONDET {
    Val wrap = isz(oldLow - 254);
    IF(1 - wrap) {
      high->set(oldHigh);
      low->set(oldLow + 2);
    }
    IF(wrap) {
      high->set(oldHigh + 1);
      low->set(0);
    }
  }
  Val diffHigh = high - oldHigh;
  Val diffLow = low - oldLow;
  // The following constraint enforces that diffHigh is either 0 or 1
  // diffHigh will be 0 if we don't wrap and 1 if we do.
  eqz(diffHigh * (diffHigh - 1));
  // The following constraint enforces that either diffHigh = 0 or diffLow = -254
  // diffHigh will be 0 if we don't wrap; diffLow + 254 will be 0 if we do.
  eqz(diffHigh * (diffLow + 254));
  // The following constraint enforces that either diffHigh = 1 or diffLow = 2
  // diffLow will be 2 if we don't wrap; diffHigh will be 1 if we do.
  eqz((diffHigh - 1) * (diffLow - 2));
}

void BytesPlonkVerifierImpl::verify(BytesPlonkElement a,
                                    BytesPlonkElement b,
                                    Comp<BytesPlonkVerifierImpl> prevVerifier,
                                    size_t back,
                                    Val checkDirty) {
  Val oldHigh = BACK(back, a->high->get());
  Val oldLow = BACK(back, a->low->get());
  Val newHigh = b->high;
  Val newLow = b->low;
  Val diffHigh = newHigh - oldHigh;
  Val diffLow = newLow - oldLow;
  // High byte can only move 0 or 1 forward
  eqz(diffHigh * (diffHigh - 1));
  // If high byte moves
  IF(diffHigh) {
    // New value is 0
    eqz(newLow);
    // Old value is 254 or 255
    eqz((oldLow - 255) * (oldLow - 254));
  }
  IF(1 - diffHigh) {
    // Low moves forward 0, 1, or 2
    eqz(diffLow * (diffLow - 1) * (diffLow - 2));
  }
}

} // namespace impl

BytesHeaderImpl::BytesHeaderImpl()
    : PlonkHeaderBase("bytes", "_bytes_finalize", "bytes_verify", "compute_accum", "verify_accum") {
}

ByteRegImpl::ByteRegImpl() : reg(CompContext::allocateFromPool<impl::ByteAlloc>("bytes")->reg) {}

Val ByteRegImpl::get() {
  return reg->get();
}

Val ByteRegImpl::set(Val in) {
  NONDET { reg->set(in & 0xff); }
  return (in - reg->get()) / 256;
}

void ByteRegImpl::setExact(Val in) {
  reg->set(in);
}

size_t BytesSetupImpl::setupCount(size_t useRegs) {
  size_t pairCount = useRegs / 4;
  return ceilDiv(32768, pairCount);
}

BytesSetupImpl::BytesSetupImpl(BytesHeader header, size_t useRegs)
    : pairCount(useRegs / 4), body(header, pairCount, 4) {
  assert(pairCount > 0);
}

void BytesSetupImpl::set(Val isFirst, Val isLast) {
  size_t rem = 32768 % pairCount;
  IF(isFirst) { body->at(0)->setInit(); }
  IF(1 - isFirst) {
    auto oldVals = BACK(1, body->at(pairCount - 1)->toVals());
    body->at(0)->setNext(oldVals);
  }
  for (size_t i = 1; i < rem; i++) {
    body->at(i)->setNext(body->at(i - 1)->toVals());
  }
  IF(isLast) {
    for (size_t i = rem; i < pairCount; i++) {
      body->at(i)->high->set(0);
      body->at(i)->low->set(0);
    }
  }
  IF(1 - isLast) {
    for (size_t i = rem; i < pairCount; i++) {
      body->at(i)->setNext(body->at(i - 1)->toVals());
    }
  }
}

BytesBodyImpl::BytesBodyImpl(BytesHeader header, size_t count)
    : pairCount(ceilDiv(count, 2)), body(header, pairCount, 4) {
  assert(pairCount > 0);
  for (size_t i = 0; i < pairCount; i++) {
    CompContext::addToPool("bytes", std::make_shared<impl::ByteAlloc>(body->at(i)->high, 2 * i));
    CompContext::addToPool("bytes", std::make_shared<impl::ByteAlloc>(body->at(i)->low, 2 * i + 1));
  }
}

} // namespace zirgen
