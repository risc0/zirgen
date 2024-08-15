// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace risc0 {

// Loads an ELF file and collects the memory that would be written on loading
// in memOut All writes are 32 bit wide and aligned.  The memory map is from
// 'word_num' (i.e. addr / 4) to word.  Throws std::runtime_error on any errors
// (file, type, misalignment, addr/4 < minWord, addr/4 >= maxMem, etc).
// Returns the entry point address.

uint32_t loadElf(const std::vector<uint8_t>& elfBytes,
                 std::map<uint32_t, uint32_t>& memOut,
                 uint32_t minWord = 0,
                 uint32_t maxWord = 0x40000000);

} // namespace risc0
