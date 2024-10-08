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
