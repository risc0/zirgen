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

#include "zirgen/circuit/rv32im/v1/platform/constants.h"

#include <cstdint>
#include <string>

namespace zirgen::rv32im_v1 {

struct OpcodeInfo {
  OpcodeInfo() : major(MajorType::kMuxSize), minor(0) {}
  OpcodeInfo(const char* mnemonic, uint32_t major, uint32_t minor)
      : mnemonic(mnemonic), major(major), minor(minor) {}
  std::string mnemonic;
  uint32_t major;
  uint32_t minor;
};

OpcodeInfo getOpcodeInfo(uint32_t inst);

}; // namespace zirgen::rv32im_v1
