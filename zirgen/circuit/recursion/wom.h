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

#pragma once

#include "zirgen/compiler/edsl/component.h"
#include "zirgen/components/plonk.h"

#include <array>

namespace zirgen::recursion {

namespace impl {

struct WomPlonkElementImpl : public CompImpl<WomPlonkElementImpl> {
  static constexpr size_t rawSize() { return kExtSize + 1; }
  void setInit();
  void setFini() {}
  void setFini(Val lastAddr);
  std::vector<Val> toVals();
  void setFromVals(std::vector<Val> vals);
  void setNOP();
  std::array<Val, kExtSize> dataVals();
  void setData(llvm::ArrayRef<Val> d);

  // In verification order
  Reg addr = Label("addr");
  std::array<Reg, kExtSize> data = Label("data");
};
using WomPlonkElement = Comp<WomPlonkElementImpl>;

struct WomPlonkVerifierImpl : public CompImpl<WomPlonkVerifierImpl> {
  void verify(WomPlonkElement a,
              WomPlonkElement b,
              Comp<WomPlonkVerifierImpl> prevVerifier,
              size_t back,
              Val checkDirty);
  void setInit() {}
  bool hasState() { return false; }
  void setFromVals(std::vector<Val> vals) {}
  std::vector<Val> toVals() { return {}; }
};
using WomPlonkVerifier = Comp<WomPlonkVerifierImpl>;

struct WomAlloc : public AllocatableBase {
  WomAlloc(WomPlonkElement elem, size_t id = 0) : AllocatableBase(id), elem(elem) {}
  void finalize() override { elem->setNOP(); }
  void saveLabel(llvm::StringRef label) override { elem->saveLabel(label); }
  WomPlonkElement elem;
};

} // namespace impl

struct WomHeaderImpl : public PlonkHeaderBase<impl::WomPlonkElement, impl::WomPlonkVerifier>,
                       public CompImpl<WomHeaderImpl> {
  WomHeaderImpl();
  Val getCheckDirty() { return 0; }
};

using WomHeader = Comp<WomHeaderImpl>;
using WomInit = PlonkInit<WomHeader>;
using WomPass = PlonkPass<WomHeader>;
using WomFini = PlonkFini<impl::WomPlonkElement, WomHeader>;

class WomRegImpl : public CompImpl<WomRegImpl> {
public:
  WomRegImpl();

  std::array<Val, kExtSize> doRead(Val addr);
  void doWrite(Val addr, std::array<Val, kExtSize> data);
  void doNOP();

  Val addr();
  std::array<Val, kExtSize> data();

private:
  impl::WomPlonkElement elem;
};
using WomReg = Comp<WomRegImpl>;

class WomBodyImpl : public CompImpl<WomBodyImpl> {
public:
  // Make a table capable of holding 'count' memory IOs
  WomBodyImpl(WomHeader header, size_t count, size_t maxDeg);

private:
  PlonkBody<impl::WomPlonkElement, impl::WomPlonkVerifier, WomHeader> body;
};
using WomBody = Comp<WomBodyImpl>;

class WomExternHandler : public PlonkExternHandler {
public:
  WomExternHandler();
  std::optional<std::vector<uint64_t>> doExtern(llvm::StringRef name,
                                                llvm::StringRef extra,
                                                llvm::ArrayRef<const Zll::InterpVal*> args,
                                                size_t outCount) override;

  std::map<size_t, std::array<uint64_t, kExtSize>> state;
};

} // namespace zirgen::recursion
