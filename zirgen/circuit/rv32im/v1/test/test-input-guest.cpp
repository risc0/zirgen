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

#include <cstdint>

#include "guest.h"

// result of `shaHash("Hello")`
constexpr uint32_t expected[8] = {0xb38d5f18, //
                                  0x25fe7122,
                                  0xfca661f5,
                                  0x262e8b93,
                                  0x30ec0643,
                                  0x8051da4e,
                                  0x4876d107,
                                  0x69193826};

extern "C" void start() {
  for (int i = 0; i < 8; i++) {
    if (sys_input(i) != expected[i]) {
      fail();
    }
  }
  sys_halt();
}
