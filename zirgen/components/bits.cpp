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
