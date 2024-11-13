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

#include "zirgen/circuit/recursion/wom.h"

namespace zirgen::recursion {

namespace impl {

void WomPlonkElementImpl::setInit() {
  addr->set(0);
  setData(Val(0));
}

void WomPlonkElementImpl::setFini(Val lastAddr) {
  addr->set(lastAddr);
  setData(Val(0));
}

std::vector<Val> WomPlonkElementImpl::toVals() {
  std::vector<Val> out = {addr->get()};
  auto more = dataVals();
  out.insert(out.end(), more.begin(), more.end());
  assert(out.size() == rawSize());
  return out;
}

void WomPlonkElementImpl::setFromVals(std::vector<Val> vals) {
  assert(vals.size() == rawSize());
  addr->set(vals[0]);
  for (size_t i = 0; i < kExtSize; i++) {
    data[i]->set(vals[i + 1]);
  }
}

void WomPlonkElementImpl::setNOP() {
  setInit();
}

std::array<Val, kExtSize> WomPlonkElementImpl::dataVals() {
  std::array<Val, kExtSize> out;
  for (size_t i = 0; i < out.size(); i++) {
    out[i] = data[i]->get();
  }
  return out;
}

void WomPlonkElementImpl::setData(llvm::ArrayRef<Val> d) {
  for (size_t i = 0; i < kExtSize; i++) {
    data[i]->set(d[std::min(i, d.size() - 1)]);
  }
}

void WomPlonkVerifierImpl::verify(WomPlonkElement a,
                                  WomPlonkElement b,
                                  Comp<WomPlonkVerifierImpl> prevVerifier,
                                  size_t back,
                                  Val checkDirty) {
  // Get the address difference
  Val aAddr = BACK(back, a->addr->get());
  Val addrDiff = (b->addr - aAddr);
  // Verify it is zero or one
  eqz(addrDiff * (1 - addrDiff));
  // If it's one, we don't verify anything, if it's 0 data must match
  IF(1 - addrDiff) {
    for (size_t i = 0; i < kExtSize; i++) {
      eq(BACK(back, a->data[i]->get()), b->data[i]->get());
    }
  }
}

} // namespace impl

WomHeaderImpl::WomHeaderImpl()
    : PlonkHeaderBase("wom", "_wom_finalize", "wom_verify", "compute_accum", "verify_accum") {}

WomRegImpl::WomRegImpl() : elem(CompContext::allocateFromPool<impl::WomAlloc>("wom")->elem) {}

std::array<Val, kExtSize> WomRegImpl::doRead(Val addr) {
  NONDET { elem->setData(doExtern("womRead", "", kExtSize, {addr})); }
  elem->addr->set(addr);
  return elem->dataVals();
}

void WomRegImpl::doWrite(Val addr, std::array<Val, kExtSize> data) {
  elem->addr->set(addr);
  elem->setData(data);
  NONDET { doExtern("womWrite", "", 0, elem->toVals()); }
}

void WomRegImpl::doNOP() {
  elem->setNOP();
}

Val WomRegImpl::addr() {
  return elem->addr->get();
}
std::array<Val, kExtSize> WomRegImpl::data() {
  return elem->dataVals();
}

WomBodyImpl::WomBodyImpl(WomHeader header, size_t count, size_t maxDeg)
    : body(Label("plonk_body"), header, count, maxDeg) {
  for (size_t i = 0; i < count; i++) {
    CompContext::addToPool("wom", std::make_shared<impl::WomAlloc>(body->at(i)));
  }
}

WomExternHandler::WomExternHandler() {
  state[0] = {0, 0, 0, 0};
}

std::optional<std::vector<uint64_t>>
WomExternHandler::doExtern(llvm::StringRef name,
                           llvm::StringRef extra,
                           llvm::ArrayRef<const Zll::InterpVal*> args,
                           size_t outCount) {
  if (name == "womWrite") {
    uint64_t addr = args[0]->getBaseFieldVal();
    if (state.count(addr) != 0) {
      llvm::errs() << "Invalid WOM write at address " << addr << "\n";
      throw std::runtime_error("INVALID WOM WRITE");
    }
    std::array<uint64_t, kExtSize> data;
    for (size_t i = 0; i < data.size(); i++) {
      data[i] = args[1 + i]->getBaseFieldVal();
    }
    state[addr] = data;
    return std::vector<uint64_t>{};
  }
  if (name == "womRead") {
    uint32_t addr = args[0]->getBaseFieldVal();
    if (state.count(addr) != 1) {
      llvm::errs() << "Invalid WOM read at address " << addr << "\n";
      throw std::runtime_error("INVALID WOM READ");
    }
    auto data = state[addr];
    return std::vector<uint64_t>{data[0], data[1], data[2], data[3]};
  }
  return PlonkExternHandler::doExtern(name, extra, args, outCount);
}

} // namespace zirgen::recursion
