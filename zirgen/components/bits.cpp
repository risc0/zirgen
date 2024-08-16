// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/components/bits.h"

namespace zirgen {

// Checks that a val is a bit.
void isBit(Val val, SourceLoc loc) {
  OverrideLocation local(loc);
  // The following constraint enforces that either val = 0 or val = 1
  eqz(val * (1 - val));
}

void isBits(Buffer buf, SourceLoc loc) {
  OverrideLocation local(loc);
  for (size_t i = 0; i < buf.size(); i++) {
    isBit(buf[i]);
  }
}

} // namespace zirgen
