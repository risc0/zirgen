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

#include "zirgen/circuit/rv32im/v2/emu/paging.h"

#include <algorithm>
#include <iostream>

namespace zirgen::rv32im_v2 {

size_t CYCLE_COST_PAGE = 1 +                          // POSEIDON_PAGING
                         10 * (PAGE_SIZE_WORDS / 8) + // POSEIDON_LOAD_IN + 8 Ext + 1 Int
                         1;                           // // PoseidonDoOut

size_t CYCLE_COST_MERKLE = 1 + // POSEIDON_PAGING
                           2 + // POSEIDON_LOAD_IN
                           9 + // (8 Ext + 1 Int)
                           1;  // PoseidonDoOut

size_t CYCLE_COST_EXTRA = 1 + // LOAD_ROOT
                          1 + // POSEIDON_ENTRY
                          // Page reads
                          1 + // POSEIDON_PAGING
                          2 + // RESUME
                          // Normal execution
                          2 + // SUSPEND
                          1 + // POSEIDON_ENTRY
                          // Page writes
                          1 +          // POSEIDON_PAGING
                          1 +          // STORE_ROOT
                          256 / 16 +   // U8 table
                          65536 / 16 + // U16 table
                          0;

PagedMemory::PagedMemory(MemoryImage& image) : image(image), pagingCycles(CYCLE_COST_EXTRA) {}

uint32_t PagedMemory::load(uint32_t word) {
  if (word >= 0x40000000) {
    std::cerr << "Load of invalid word: " << word << "\n";
    throw std::runtime_error("Load of invalid word");
  }
  uint32_t page = word / PAGE_SIZE_WORDS;
  uint32_t idx = MEMORY_SIZE_PAGES + page;
  PageState& state = stateTable[idx];
  if (state == PageState::UNLOADED) {
    loadPage(page);
    state = PageState::LOADED;
  }
  return pageCache[page][word % PAGE_SIZE_WORDS];
}

uint32_t PagedMemory::peek(uint32_t word) {
  if (word >= 0x40000000) {
    std::cerr << "Peek of invalid word: " << word << "\n";
    throw std::runtime_error("Peek of invalid word");
  }
  uint32_t page = word / PAGE_SIZE_WORDS;
  uint32_t idx = MEMORY_SIZE_PAGES + page;
  auto it = stateTable.find(idx);
  if (it == stateTable.end()) {
    // Unloaded, peek into image
    return (*image.getPage(page))[word % PAGE_SIZE_WORDS];
  } else {
    // Loaded, get from cache
    return pageCache[page][word % PAGE_SIZE_WORDS];
  }
}

void PagedMemory::store(uint32_t word, uint32_t val) {
  if (word >= 0x40000000) {
    std::cerr << "Store of invalid word: " << word << "\n";
    throw std::runtime_error("Store of invalid word");
  }
  uint32_t page = word / PAGE_SIZE_WORDS;
  uint32_t idx = MEMORY_SIZE_PAGES + page;
  PageState& state = stateTable[idx];
  if (state == PageState::UNLOADED) {
    loadPage(page);
    state = PageState::LOADED;
  }
  if (state == PageState::LOADED) {
    // TODO: Handle page out
    pagingCycles += CYCLE_COST_PAGE;
    fixupCosts(idx, PageState::DIRTY);
    state = PageState::DIRTY;
  }
  pageCache[page][word % PAGE_SIZE_WORDS] = val;
}

size_t PagedMemory::getPagingCycles() {
  return pagingCycles;
}

MemoryImage PagedMemory::commit() {
  MemoryImage ret;
  // Gather the original pages
  for (auto& kvp : pageCache) {
    ret.setPage(kvp.first, image.getPage(kvp.first));
  }
  std::vector<size_t> orderedIdx;
  for (auto& kvp : stateTable) {
    orderedIdx.push_back(kvp.first);
  }
  std::sort(orderedIdx.begin(), orderedIdx.end());
  // Add minimal needed 'uncles'
  for (size_t idx : orderedIdx) {
    // If this is a leaf, break
    if (idx >= MEMORY_SIZE_PAGES) {
      break;
    }
    // Otherwise, add whichever child digest (if any) is not loaded
    if (!stateTable.count(idx * 2)) {
      ret.setDigest(idx * 2, image.getDigest(idx * 2));
    }
    if (!stateTable.count(idx * 2 + 1)) {
      ret.setDigest(idx * 2 + 1, image.getDigest(idx * 2 + 1));
    }
  }
  // Update data in image
  for (auto& kvp : pageCache) {
    size_t page = kvp.first;
    uint32_t idx = MEMORY_SIZE_PAGES + page;
    if (stateTable[idx] == PageState::DIRTY) {
      image.setPage(page, std::make_shared<Page>(kvp.second));
    }
  }
  return ret;
}

void PagedMemory::clear() {
  pageCache.clear();
  stateTable.clear();
  pagingCycles = CYCLE_COST_EXTRA;
}

PagingInfo PagedMemory::readPaging() {
  PagingInfo ret;
  ret.pages = image.getKnownPages();
  computePaging(ret);
  return ret;
}

PagingInfo PagedMemory::writePaging() {
  PagingInfo ret;
  for (const auto& kvp : pageCache) {
    size_t page = kvp.first;
    uint32_t idx = MEMORY_SIZE_PAGES + page;
    if (stateTable[idx] == PageState::DIRTY) {
      ret.pages[page] = image.getPage(page);
    }
  }
  computePaging(ret);
  return ret;
}

void PagedMemory::loadPage(uint32_t page) {
  uint32_t idx = MEMORY_SIZE_PAGES + page;
  pageCache[page] = *image.getPage(page);
  pagingCycles += CYCLE_COST_PAGE;
  fixupCosts(idx, PageState::LOADED);
}

void PagedMemory::fixupCosts(uint32_t idx, PageState goalState) {
  while (idx != 0) {
    PageState& state = stateTable[idx];
    if (goalState > state) {
      if (idx < MEMORY_SIZE_PAGES) {
        if (state == PageState::UNLOADED) {
          pagingCycles += CYCLE_COST_MERKLE;
        }
        if (goalState == PageState::DIRTY) {
          pagingCycles += CYCLE_COST_MERKLE;
        }
      }
      state = goalState;
    }
    idx /= 2;
  }
}

void PagedMemory::computePaging(PagingInfo& info) {
  for (const auto& kvp : info.pages) {
    size_t idx = MEMORY_SIZE_PAGES + kvp.first;
    info.nodes[idx / 2];
  }
  for (auto it = info.nodes.rbegin(); it != info.nodes.rend(); ++it) {
    it->second.left = image.getDigest(2 * it->first);
    it->second.right = image.getDigest(2 * it->first + 1);
    if (it->first != 1) {
      info.nodes[it->first / 2];
    }
  }
}

} // namespace zirgen::rv32im_v2
