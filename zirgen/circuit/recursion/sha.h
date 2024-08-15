// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/circuit/recursion/code.h"
#include "zirgen/circuit/recursion/wom.h"

namespace zirgen::recursion {

class ShaCycleImpl : public CompImpl<ShaCycleImpl> {
public:
  ShaCycleImpl(MacroOpcode major, WomHeader womHeader);
  void set(MacroInst inst, Val writeAddr);

private:
  MacroOpcode major;

  void setInit(MacroInst inst, Val writeAddr);
  void setLoad(MacroInst inst, Val writeAddr);
  void setMix(MacroInst inst, Val writeAddr);
  void setFini(MacroInst inst, Val writeAddr);
  void computeW();
  void computeAE();
  void onVerify();

  WomBody body;
  WomReg io0 = Label("io0");
  WomReg io1 = Label("io1");

  std::array<Bit, 32> a = Label("a");
  std::array<Reg, 2> aRaw = Label("a_raw");
  Twit aCarryLow = Label("a_carry_low");
  Twit aCarryHigh = Label("a_carry_hi");

  std::array<Bit, 32> e = Label("e");
  std::array<Reg, 2> eRaw = Label("e_raw");
  Twit eCarryLow = Label("e_carry_low");
  Twit eCarryHigh = Label("e_carry_high");

  std::array<Bit, 32> w = Label("w");
  std::array<Reg, 2> wRaw = Label("w_raw");
  Twit wCarryLow = Label("w_carry_low");
  Twit wCarryHigh = Label("w_carry_high");
};

using ShaCycle = Comp<ShaCycleImpl>;

template <MacroOpcode major> struct ShaWrapImpl : public CompImpl<ShaWrapImpl<major>> {
  ShaWrapImpl(WomHeader womHeader) : inner(Label("sha_cycle"), major, womHeader) {}
  void set(MacroInst inst, Val writeAddr) { inner->set(inst, writeAddr); }
  ShaCycle inner;
};
template <MacroOpcode major> using ShaWrap = Comp<ShaWrapImpl<major>>;

} // namespace zirgen::recursion
