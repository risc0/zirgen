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

#include "zirgen/circuit/rv32im/v2/emu/image.h"

#include "zirgen/circuit/rv32im/v2/emu/r0vm.h"
#include "zirgen/compiler/zkp/poseidon2.h"

#include <iostream>

namespace zirgen::rv32im_v2 {

Digest hashPage(const uint32_t* data) {
  std::array<uint32_t, 24> cells;
  cells.fill(0);
  // Load data into felts as 16 bit values
  for (size_t i = 0; i < PAGE_SIZE_WORDS / 8; i++) {
    for (size_t j = 0; j < 8; j++) {
      cells[2 * j] = data[i * 8 + j] & 0xffff;
      cells[2 * j + 1] = data[i * 8 + j] >> 16;
    }
    poseidonSponge(cells);
  }
  Digest out;
  for (size_t i = 0; i < 8; i++) {
    out.words[i] = cells[i];
  }
  return out;
}

Digest hashPair(const Digest& lhs, const Digest& rhs) {
  std::array<uint32_t, 24> cells;
  cells.fill(0);
  for (size_t i = 0; i < 8; i++) {
    cells[i] = rhs.words[i];
    cells[8 + i] = lhs.words[i];
  }
  poseidonSponge(cells);
  Digest out;
  for (size_t i = 0; i < 8; i++) {
    out.words[i] = cells[i];
  }
  return out;
}

MemoryImage::MemoryImage() {
  initZeros();
}

MemoryImage MemoryImage::zeros() {
  MemoryImage ret;
  ret.digests[1] = ret.zeroDigests[0];
  return ret;
}

MemoryImage MemoryImage::fromWords(const std::map<uint32_t, uint32_t>& words) {
  MemoryImage ret = MemoryImage::zeros();
  uint32_t curPageID = 0xffffffff;
  std::shared_ptr<Page> curPage;
  for (const auto& kvp : words) {
    uint32_t pageID = kvp.first / PAGE_SIZE_WORDS;
    if (pageID != curPageID) {
      if (curPage) {
        ret.setPage(curPageID, curPage);
      }
      curPage = std::make_shared<Page>();
      curPageID = pageID;
    }
    // printf("store(0x%08x, 0x%08x)\n", kvp.first, kvp.second);
    (*curPage)[kvp.first % PAGE_SIZE_WORDS] = kvp.second;
  }
  if (curPage) {
    ret.setPage(curPageID, curPage);
  }
  return ret;
}

MemoryImage MemoryImage::fromElfs(const std::string& kernel, const std::string& user) {
  std::map<uint32_t, uint32_t> words;
  loadWithKernel(words, kernel, user);
  return MemoryImage::fromWords(words);
}

MemoryImage MemoryImage::fromRawElf(const std::string& elf) {
  std::map<uint32_t, uint32_t> words;
  loadRaw(words, elf);
  return MemoryImage::fromWords(words);
}

PagePtr MemoryImage::getPage(size_t page) {
  // If page exists, return it
  auto it = pages.find(page);
  if (it != pages.end()) {
    return it->second;
  }
  // Otherwise try an expand
  if (expandIfZero(MEMORY_SIZE_PAGES + page)) {
    pages[page] = zeroPage;
    return zeroPage;
  }
  // Otherwise fail
  std::cerr << "Unavailable page: " << page << "\n";
  throw std::runtime_error("Attempting to read unavailable page");
}

// Sets the data for a page
void MemoryImage::setPage(size_t page, PagePtr data) {
  // printf("setPage(0x%08zx)\n", page);
  // If page is zero, reify it so I have proper uncles
  expandIfZero(MEMORY_SIZE_PAGES + page);
  // Set page
  pages[page] = data;
  // Set the diest value
  digests[MEMORY_SIZE_PAGES + page] = hashPage(data->data());
  // Fixup digest values
  fixupDigests(MEMORY_SIZE_PAGES + page);
}

const Digest& MemoryImage::getDigest(size_t idx) const {
  // Expand if needed: make this appear const
  const_cast<MemoryImage*>(this)->expandIfZero(idx);
  // Return digest if available
  auto it = digests.find(idx);
  if (it != digests.end()) {
    return it->second;
  }
  // Otherwise fail
  throw std::runtime_error("Attempting to read unavailable digest");
}

void MemoryImage::setDigest(size_t idx, const Digest& digest) {
  // If digest is in a zero region, reify for proper uncles
  expandIfZero(idx);
  // Set the diest value
  digests[idx] = digest;
  // Fixup digest values
  fixupDigests(idx);
}

std::map<uint32_t, PagePtr> MemoryImage::getKnownPages() const {
  return std::map<uint32_t, PagePtr>(pages.begin(), pages.end());
}

std::map<uint32_t, Digest> MemoryImage::getKnownDigests() const {
  return std::map<uint32_t, Digest>(digests.begin(), digests.end());
}

void MemoryImage::initZeros() {
  auto writableZeroPage = std::make_shared<Page>();
  writableZeroPage->fill(0);
  zeroPage = writableZeroPage;
  Digest curDigest = hashPage(zeroPage->data());
  zeroDigests.resize(MERKLE_TREE_DEPTH + 1);
  for (size_t depth = MERKLE_TREE_DEPTH + 1; depth-- > 0;) {
    zeroDigests[depth] = curDigest;
    curDigest = hashPair(curDigest, curDigest);
  }
}

void MemoryImage::fixupDigests(size_t idx) {
  while (idx != 1) {
    size_t up = idx / 2;
    if (!digests.count(2 * up) || !digests.count(2 * up + 1)) {
      return;
    }
    Digest left = digests.at(2 * up);
    Digest right = digests.at(2 * up + 1);
    digests[up] = hashPair(left, right);
    idx = up;
  }
}

bool MemoryImage::isZero(size_t idx) {
  // Compute the depth in the tree of this node
  size_t depth = log2Floor(idx);
  // Go up until we hit a valid node or get past the root
  while (digests.count(idx) == 0 && idx > 0) {
    idx /= 2;
    depth--;
  }
  if (idx == 0) {
    return false;
  } // Failed to find a root at all
  return digests.at(idx) == zeroDigests[depth];
}

void MemoryImage::expandZero(size_t idx) {
  // Compute the depth in the tree of this node
  size_t depth = log2Floor(idx);
  // Go up until we hit the valid zero node
  while (digests.count(idx) == 0) {
    size_t newIdx = idx / 2;
    digests[2 * newIdx] = zeroDigests[depth];
    digests[2 * newIdx + 1] = zeroDigests[depth];
    idx = newIdx;
    depth--;
  }
}

bool MemoryImage::expandIfZero(size_t idx) {
  // printf("expandIfZero(0x%08zx)\n", idx);
  if (isZero(idx)) {
    expandZero(idx);
    return true;
  }
  return false;
}

} // namespace zirgen::rv32im_v2
