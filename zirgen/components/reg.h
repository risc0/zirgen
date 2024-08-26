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
