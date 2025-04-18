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

extern "C" void start() {
  uint32_t input[2];
  uint32_t len = sys_io(input, 2, 0, 0, 0);
  if (len != 8 || input[0] != 0x01020304 || input[1] != 0x05060708) {
    fail();
  }
  sys_halt();
}
