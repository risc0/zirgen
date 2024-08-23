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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace zirgen::rv32im_v1 {

struct PageTableInfo {
  std::vector<size_t> layers;
  size_t maxMem;
  size_t numPages;
  uint32_t lastAddr;
  uint32_t rootAddr;
  uint32_t rootIndex;
  uint32_t rootPageAddr;
  size_t numRootEntries;

  PageTableInfo();
};

uint32_t getPageIndex(uint32_t addr);
uint32_t getPageAddr(uint32_t pageIndex);

} // namespace zirgen::rv32im_v1
