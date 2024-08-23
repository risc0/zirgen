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

#include "zirgen/circuit/recursion/bits.h"
#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/recursion/sha.h"

namespace zirgen::recursion {

struct NopImpl : public CompImpl<NopImpl> {
  NopImpl(WomHeader header) {}
  void set(MacroInst inst, Val writeAddr) {}
};
using Nop = Comp<NopImpl>;

struct WomInitWrapperImpl : public CompImpl<WomInitWrapperImpl> {
  WomInitWrapperImpl(WomHeader header);
  void set(MacroInst inst, Val writeAddr);

  WomInit init;
};
using WomInitWrapper = Comp<WomInitWrapperImpl>;

struct WomFiniWrapperImpl : public CompImpl<WomFiniWrapperImpl> {
  WomFiniWrapperImpl(WomHeader header);
  void set(MacroInst inst, Val writeAddr);

  WomFini fini;
};
using WomFiniWrapper = Comp<WomFiniWrapperImpl>;

struct SetGlobalImpl : public CompImpl<SetGlobalImpl> {
  SetGlobalImpl(WomHeader header);
  void set(MacroInst inst, Val writeAddr);

  OneHot<2 * kOutDigests> select = Label("select");
  WomBody body;
  std::array<WomReg, 4> regs = Label("regs");
  std::vector<Reg> outRegs;
};
using SetGlobal = Comp<SetGlobalImpl>;

struct MacroOpImpl : public CompImpl<MacroOpImpl> {
  MacroOpImpl(Code code, WomHeader header);
  void set(Code inst, Val writeAddr);

  TwitPrepare<6> twitPrep;

  Mux<Nop,
      WomInitWrapper,
      WomFiniWrapper,
      BitAndElem,
      BitOpShorts,
      ShaWrap<MacroOpcode::SHA_INIT>,
      ShaWrap<MacroOpcode::SHA_FINI>,
      ShaWrap<MacroOpcode::SHA_LOAD>,
      ShaWrap<MacroOpcode::SHA_MIX>,
      SetGlobal>
      mux;
};
using MacroOp = Comp<MacroOpImpl>;

} // namespace zirgen::recursion
