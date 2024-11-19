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

#include "fp.h"

#include <iostream>
#include <map>

namespace risc0 {

struct MemTxnKey {
  uint32_t addr;
  uint32_t cycle;
  uint32_t data;

  bool operator<(const MemTxnKey& rhs) const {
    if (addr != rhs.addr) {
      return addr < rhs.addr;
    }
    if (cycle != rhs.cycle) {
      return cycle < rhs.cycle;
    }
    return data < rhs.data;
  }
};

struct LookupTables {
  std::map<uint32_t, Fp> tableU8;
  std::map<uint32_t, Fp> tableU16;
  std::map<MemTxnKey, Fp> tableMem;
  std::map<Fp, Fp> tableCycle;

  void lookupDelta(Fp table, Fp index, Fp count) {
    uint32_t tableU32 = table.asUInt32();
    if (tableU32 == 0) {
      tableCycle[index] += count;
      return;
    }
    if (tableU32 != 8 && tableU32 != 16) {
      throw std::runtime_error("Invalid lookup table");
    }
    if (index.asUInt32() >= (1 << tableU32)) {
      std::cerr << "LOOKUP ERROR: table = " << table.asUInt32() << ", index = " << index.asUInt32()
                << "\n";
      throw std::runtime_error("u8/16 table error");
    }
    if (tableU32 == 8) {
      tableU8[index.asUInt32()] += count;
    } else {
      tableU16[index.asUInt32()] += count;
    }
  }

  Fp lookupCurrent(Fp table, Fp index) {
    uint32_t tableU32 = table.asUInt32();
    if (tableU32 != 8 && tableU32 != 16) {
      throw std::runtime_error("Invalid lookup table");
    }
    if (tableU32 == 8) {
      return tableU8[index.asUInt32()];
    } else {
      return tableU16[index.asUInt32()];
    }
  }

  void memoryDelta(uint32_t addr, uint32_t cycle, uint32_t data, Fp count) {
    tableMem[{addr, cycle, data}] += count;
  }

  void check() {
    for (const auto& kvp : tableU8) {
      if (kvp.second != 0) {
        std::cerr << "U8 entry " << kvp.first << ": " << kvp.second.asUInt32() << "\n";
        throw std::runtime_error("Table not zero");
      }
    }
    for (const auto& kvp : tableU16) {
      if (kvp.second != 0) {
        std::cerr << "U16 entry " << kvp.first << ": " << kvp.second.asUInt32() << "\n";
        throw std::runtime_error("Table not zero");
      }
    }
    for (const auto& kvp : tableMem) {
      if (kvp.second != 0) {
        std::cerr << "Nonzero memory entry: (" << kvp.first.addr << ", " << kvp.first.cycle << ", "
                  << kvp.first.data << ") = " << kvp.second.asUInt32() << "\n";
        throw std::runtime_error("Table not zero");
      }
    }
    for (const auto& kvp : tableCycle) {
      if (kvp.second != 0) {
        std::cerr << "Cycle entry " << kvp.first.asUInt32() << ": " << kvp.second.asUInt32()
                  << "\n";
        throw std::runtime_error("Table not zero");
      }
    }
  }
};

} // namespace risc0
