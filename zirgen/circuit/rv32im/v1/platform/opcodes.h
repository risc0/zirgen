// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
