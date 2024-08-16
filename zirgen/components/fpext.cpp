// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "fpext.h"

using namespace risc0;

namespace zirgen {

FpExtRegImpl::FpExtRegImpl(llvm::StringRef source) {
  for (size_t i = 0; i < kExtSize; i++) {
    elems.emplace_back(Label("elem", i), source);
  }
}

FpExt FpExtRegImpl::get(SourceLoc loc) {
  OverrideLocation local(loc);
  std::array<Val, kExtSize> arr;
  for (size_t i = 0; i < kExtSize; i++) {
    arr[i] = elems[i];
  }
  return FpExt(arr, loc);
}

void FpExtRegImpl::set(CaptureFpExt rhs) {
  for (size_t i = 0; i < kExtSize; i++) {
    elems[i]->set(rhs.ext.elem(i));
  }
}

FpExt::FpExt(Val x, SourceLoc loc) {
  OverrideLocation local(loc);
  elems[0] = x;
  for (size_t i = 1; i < kExtSize; i++) {
    elems[i] = 0;
  }
}

FpExt::FpExt(std::array<Val, kExtSize> elems, risc0::SourceLoc loc) {
  OverrideLocation local(loc);
  for (size_t i = 0; i < kExtSize; i++) {
    this->elems[i] = elems[i];
  }
}

FpExt::FpExt(FpExtReg reg, risc0::SourceLoc loc) {
  OverrideLocation local(loc);
  for (size_t i = 0; i < kExtSize; i++) {
    elems[i] = reg->elem(i);
  }
}

FpExt FpExt::fromVals(llvm::ArrayRef<Val> vals, risc0::SourceLoc loc) {
  assert(vals.size() == kExtSize);
  std::array<Val, kExtSize> elems;
  std::copy(vals.begin(), vals.end(), elems.begin());
  return FpExt(elems, loc);
}

FpExt operator+(CaptureFpExt a, CaptureFpExt b) {
  OverrideLocation local(a.loc);
  std::array<Val, kExtSize> out;
  for (size_t i = 0; i < kExtSize; i++) {
    out[i] = a.ext.elem(i) + b.ext.elem(i);
  }
  return FpExt(out, a.loc);
}

FpExt operator-(CaptureFpExt a, CaptureFpExt b) {
  OverrideLocation local(a.loc);
  std::array<Val, kExtSize> out;
  for (size_t i = 0; i < kExtSize; i++) {
    out[i] = a.ext.elem(i) - b.ext.elem(i);
  }
  return FpExt(out, a.loc);
}

FpExt operator*(CaptureFpExt a, CaptureFpExt b) {
  OverrideLocation local(a.loc);
  std::array<Val, kExtSize> out;
  Val NBETA = -Val(11);
  // Rename the element arrays to something small for readability
#define a(i) a.ext.elem(i)
#define b(i) b.ext.elem(i)
#if GOLDILOCKS
  out[0] = a(0) * b(0) + NBETA * (a(1) * b(1));
  out[1] = a(0) * b(1) + a(1) * b(0);
#else
  out[0] = a(0) * b(0) + NBETA * (a(1) * b(3) + a(2) * b(2) + a(3) * b(1));
  out[1] = a(0) * b(1) + a(1) * b(0) + NBETA * (a(2) * b(3) + a(3) * b(2));
  out[2] = a(0) * b(2) + a(1) * b(1) + a(2) * b(0) + NBETA * (a(3) * b(3));
  out[3] = a(0) * b(3) + a(1) * b(2) + a(2) * b(1) + a(3) * b(0);
#endif
#undef a
#undef b
  return FpExt(out, a.loc);
}

void eq(CaptureFpExt a, CaptureFpExt b) {
  OverrideLocation local(a.loc);
  for (size_t i = 0; i < kExtSize; i++) {
    eq(a.ext.elem(i), b.ext.elem(i));
  }
}

FpExt inv(CaptureFpExt a) {
  OverrideLocation local(a.loc);
  Val BETA = 11;
#define a(i) a.ext.elem(i)
#if GOLDILOCKS
  Val det = a(0) * a(0) + BETA * a(1) * a(1);
  Val idet = inv(det);
  return FpExt({a(0) * idet, -a(1) * idet});
#else
  Val b0 = a(0) * a(0) + BETA * (a(1) * (a(3) + a(3)) - a(2) * a(2));
  Val b2 = a(0) * (a(2) + a(2)) - a(1) * a(1) + BETA * (a(3) * a(3));
  Val c = b0 * b0 + BETA * b2 * b2;
  Val ic = inv(c);
  b0 = b0 * ic;
  b2 = b2 * ic;
  return FpExt(std::array<Val, kExtSize>({a(0) * b0 + BETA * a(2) * b2,
                                          -a(1) * b0 - BETA * a(3) * b2,
                                          -a(0) * b2 + a(2) * b0,
                                          a(1) * b2 - a(3) * b0}));
#endif
#undef a
}

} // namespace zirgen
