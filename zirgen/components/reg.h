// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/compiler/edsl/component.h"

namespace zirgen {

// RegImpl is an implementation of a single register.
// A register takes on values in the field.
// Registers are often organized into three groups for Merklization (control, data, and accum).
class RegImpl : public CompImpl<RegImpl> {
public:
  RegImpl(llvm::StringRef source = "data", llvm::StringRef label = {})
      : buf(CompContext::allocateFromPool<RegAlloc>(source)->buf)
      , constructPath(CompContext::getCurConstructPath()) {
    CompContext::saveLabel(buf, label);
  }
  RegImpl(Buffer buf, llvm::StringRef label = {})
      : buf(buf), constructPath(CompContext::getCurConstructPath()) {
    CompContext::saveLabel(buf, label);
  }
  // The "source" parameter indicates the grouping of a RegImpl.
  Val get(SourceLoc loc = current()) {
    OverrideLocation guard(loc);
    return buf.getRegister(0, constructPath, loc);
  }
  void set(Val in, SourceLoc loc = current()) {
    OverrideLocation guard(loc);
    buf.getRegister(0, constructPath, loc) = in;
  }
  Buffer raw() { return buf; }

private:
  Buffer buf;
  std::string constructPath;
};

using Reg = Comp<RegImpl>;

} // namespace zirgen
