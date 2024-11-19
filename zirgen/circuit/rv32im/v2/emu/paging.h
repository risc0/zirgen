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

#include <array>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "zirgen/circuit/rv32im/v2/emu/image.h"

namespace zirgen::rv32im_v2 {

struct MerkleNode {
  MerkleNode() = default;
  MerkleNode(const Digest& left, const Digest& right) : left(left), right(right) {}
  Digest left;
  Digest right;
};

// Info required to emit paging cycles
struct PagingInfo {
  // List of pages (by page #) read/written
  std::map<uint32_t, PagePtr> pages;
  // The list (by index) of all Merkle nodes above pages
  std::map<uint32_t, MerkleNode> nodes;
};

class PagedMemory {
public:
  // Make a paging structure based a memory image
  PagedMemory(MemoryImage& image);

  // Read from memory, load page if needed and copy to 'active' page set.
  uint32_t load(uint32_t word);
  // Peek from memory, no load required
  uint32_t peek(uint32_t word);
  // Write to memory, and also sets dirty flag on page.
  void store(uint32_t word, uint32_t val);

  // Get the total cost of page loads / stores in cycles
  size_t getPagingCycles();
  // Commit to image and return 'initial' memory substate
  MemoryImage commit();
  // Clear paging state
  void clear();
  // Get the info to do page reads, done before execution
  PagingInfo readPaging();
  // Get the info to do page writes, done after execution + commit
  PagingInfo writePaging();

private:
  // Page state
  enum class PageState {
    UNLOADED = 0,
    LOADED = 1,
    DIRTY = 2,
  };

  void loadPage(uint32_t page);
  void fixupCosts(uint32_t idx, PageState goalState);
  void computePaging(PagingInfo& info);

  MemoryImage& image;
  // The cache of pages.
  std::unordered_map<uint32_t, Page> pageCache;
  // The 'state' of each page + merkle node
  std::unordered_map<uint32_t, PageState> stateTable;
  // Current 'costs' for all page operations
  size_t pagingCycles;
};

} // namespace zirgen::rv32im_v2
