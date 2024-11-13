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

#include "zirgen/components/ram.h"

#include "zirgen/components/bits.h"
#include "llvm/Support/Format.h"

namespace zirgen {

namespace impl {

void RamPlonkElementImpl::setInit() {
  addr->set(0);
  cycle->set(0);
  memOp->set(MemoryOpType::kPageIo);
  data->setZero();
}

void RamPlonkElementImpl::setFini() {
  addr->set((1 << 26) - 1);
  cycle->set((1 << 25) - 1);
  memOp->set(MemoryOpType::kPageIo);
  data->setZero();
}

std::vector<Val> RamPlonkElementImpl::toVals() {
  std::vector<Val> out = {addr, cycle, memOp};
  std::vector<Val> dataRaw = data->toVals();
  out.insert(out.end(), dataRaw.begin(), dataRaw.end());
  return out;
}

void RamPlonkElementImpl::setFromVals(llvm::ArrayRef<Val> vals) {
  addr->set(vals[0]);
  cycle->set(vals[1]);
  memOp->set(vals[2]);
  data->setFromVals(vals.drop_front(3));
}

void RamPlonkElementImpl::setNOP() {
  addr->set(0);
  cycle->set(0);
  memOp->set(MemoryOpType::kRead);
  data->setZero();
}

void RamPlonkVerifierImpl::verify(RamPlonkElement a,
                                  RamPlonkElement b,
                                  RamPlonkVerifier prevVerifier,
                                  size_t back,
                                  Val checkDirty) {
  // Extract values from element a
  Val aAddr = BACK(back, a->addr->get());
  Val aCycle = BACK(back, a->cycle->get());
  Val aMemOp = BACK(back, a->memOp->get());
  U32Val aData = BACK(back, a->data->get());
  U32Val bData = b->data->get();
  Val prevDirty = BACK(back, prevVerifier->dirty->get());

  // XLOG("RAM_VERIFY a: [%10x, %5u, %u, %10x], b: [%10x, %5u, %u, %10x], prevDirty: %u"
  //      "%u",
  //      aAddr * 4,
  //      aCycle,
  //      aMemOp,
  //      aData.flat(),
  //      b->addr * 4,
  //      b->cycle,
  //      b->memOp,
  //      bData.flat(),
  //      prevDirty);

  // Decide non-det if this is a new address
  NONDET { isNewAddr->set(1 - isz(aAddr - b->addr)); }
  // Utility to check a given value is less that 2^26
  auto isValidDiff = [&](Val val) {
    for (size_t i = 0; i < 3; i++) {
      val = diff[i]->set(val);
    }
    extra->set(val);
  };

  // Addresses differ, addr must go up (and by less than 2^26)
  IF(isNewAddr) {
    // Only allow PageIo when the address differs
    eqz(MemoryOpType::kPageIo - b->memOp);
    // Ensure that the address advanced by one word
    isValidDiff(b->addr - aAddr - 1);
    // All memory ops must end non-dirty
    eqz(checkDirty * prevDirty);
  }

  // Addresses are the same
  IF(1 - isNewAddr) {
    // They match
    eqz(aAddr - b->addr);
    // Cycle goes up by less than 2^25 and reads happen before writes on the same cycle
    isValidDiff(b->cycle * 3 + b->memOp - aCycle * 3 + aMemOp);
    // If 'b' is a read, it must have the same data as whatever 'a' had
    IF(MemoryOpType::kWrite - b->memOp) { eq(aData, bData); }
  }

  Val isWrite = (MemoryOpType::kRead - b->memOp) * (MemoryOpType::kPageIo - b->memOp);
  Val isRead = (MemoryOpType::kPageIo - b->memOp) * (MemoryOpType::kWrite - b->memOp);
  Val isPageIo = (MemoryOpType::kRead - b->memOp) * (MemoryOpType::kWrite - b->memOp);

  // Compute the dirty bit
  IF(isPageIo) { dirty->set(0); }
  IF(isWrite) { dirty->set(1); }
  IF(isRead) { dirty->set(prevDirty); }
}

void RamPlonkVerifierImpl::setInit() {
  isNewAddr->set(0);
  dirty->set(0);
  diff[0]->set(0);
  diff[1]->set(0);
  diff[2]->set(0);
  extra->set(0);
}

std::vector<Val> RamPlonkVerifierImpl::toVals() {
  return {isNewAddr, dirty, diff[0], diff[1], diff[2], extra};
}

void RamPlonkVerifierImpl::setFromVals(llvm::ArrayRef<Val> vals) {
  isNewAddr->set(vals[0]);
  dirty->set(vals[1]);
  diff[0]->set(vals[2]);
  diff[1]->set(vals[3]);
  diff[2]->set(vals[4]);
  extra->set(vals[5]);
}

RamHeaderImpl::RamHeaderImpl(Reg checkDirty)
    : PlonkHeaderBase("ram", "_ram_finalize", "ram_verify", "compute_accum", "verify_accum")
    , checkDirty(checkDirty) {}

} // namespace impl

RamRegImpl::RamRegImpl() : elem(CompContext::allocateFromPool<impl::RamAlloc>("ram")->elem) {}

U32Val RamRegImpl::doRead(Val cycle, Val addr, Val op) {
  NONDET { elem->data->setFromVals(doExtern("ramRead", "", 4, {addr, op})); }
  set(addr, cycle, op, elem->data->get());
  return elem->data->get();
}

void RamRegImpl::doWrite(Val cycle, Val addr, U32Val data, Val op) {
  elem->data->set(data);
  NONDET {
    std::vector<Val> args = {addr, data.bytes[0], data.bytes[1], data.bytes[2], data.bytes[3], op};
    doExtern("ramWrite", "", 0, args);
  }
  set(addr, cycle, op, elem->data->get());
}

void RamRegImpl::doPreLoad(Val cycle, Val addr, U32Val data) {
  doWrite(cycle, addr, data, MemoryOpType::kPageIo);
}

void RamRegImpl::doNOP() {
  elem->setNOP();
}

void RamRegImpl::set(Val addr, Val cycle, Val memOp, U32Val data) {
  elem->addr->set(addr);
  elem->cycle->set(cycle);
  elem->memOp->set(memOp);
  elem->data->set(data);
}

Val RamRegImpl::addr() {
  return elem->addr->get();
}

Val RamRegImpl::cycle() {
  return elem->cycle->get();
}

Val RamRegImpl::memOp() {
  return elem->memOp->get();
}

U32Val RamRegImpl::data() {
  return elem->data->get();
}

U32Val ramPeek(Val addr) {
  auto vec = doExtern("ramRead", "", 4, {addr, MemoryOpType::kRead});
  return U32Val(vec[0], vec[1], vec[2], vec[3]);
}

RamBodyImpl::RamBodyImpl(RamHeader header, size_t count) : body(header, count, /*maxDeg=*/3) {
  for (size_t i = 0; i < count; i++) {
    CompContext::addToPool("ram", std::make_shared<impl::RamAlloc>(body->at(i)));
  }
}

RamExternHandler::RamExternHandler() : image(256 * 1024 * 1024 / 4) {}

std::optional<std::vector<uint64_t>>
RamExternHandler::doExtern(llvm::StringRef name,
                           llvm::StringRef extra,
                           llvm::ArrayRef<const Zll::InterpVal*> args,
                           size_t outCount) {
  if (name == "ramRead") {
    uint32_t addr = args[0]->getBaseFieldVal();
    uint32_t word = image[addr];
    return std::vector<uint64_t>{
        (word >> 0) & 0xff,
        (word >> 8) & 0xff,
        (word >> 16) & 0xff,
        (word >> 24) & 0xff,
    };
  }
  if (name == "ramWrite") {
    auto fpArgs = asFpArray(args);
    uint32_t addr = fpArgs[0];
    assert(fpArgs[1] < 0x100);
    assert(fpArgs[2] < 0x100);
    assert(fpArgs[3] < 0x100);
    assert(fpArgs[4] < 0x100);
    uint32_t word = (fpArgs[1] << 0) |  //
                    (fpArgs[2] << 8) |  //
                    (fpArgs[3] << 16) | //
                    (fpArgs[4] << 24);
    image[addr] = word;
    return std::vector<uint64_t>{};
  }
  return PlonkExternHandler::doExtern(name, extra, args, outCount);
}

uint32_t RamExternHandler::loadU32(uint32_t addr) {
  return image[addr];
}

void RamExternHandler::storeU32(uint32_t addr, uint32_t value) {
  image[addr] = value;
}

} // namespace zirgen
