// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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
