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

#include "zirgen/compiler/zkp/digest.h"

#include <array>
#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

namespace zirgen::rv32im_v2 {

// Defining constants from which others derive
constexpr size_t PAGE_SIZE_BYTES = 1024;
constexpr size_t MEMORY_SIZE_BYTES = size_t(1) << 32;

// Compute the largest PO2 such that (1 << PO2) <= in
inline size_t constexpr log2Floor(size_t in) {
  size_t po2 = 0;
  while ((size_t(1) << (po2 + 1)) <= in) {
    po2++;
  }
  return po2;
}

// Derived constants
constexpr size_t PAGE_SIZE_WORDS = PAGE_SIZE_BYTES / 4;
constexpr size_t MEMORY_SIZE_PAGES = MEMORY_SIZE_BYTES / PAGE_SIZE_BYTES;
constexpr size_t MERKLE_TREE_DEPTH = log2Floor(MEMORY_SIZE_PAGES);

using Page = std::array<uint32_t, PAGE_SIZE_WORDS>;
using PagePtr = std::shared_ptr<const Page>;

// A class to hold 'memory images'.  A memory image may not know all page data
// (for example partial transfer of image for proving).  Internally, the memory image
// is an actual tree of pages, with null pointer to 'unknown' pages.
class MemoryImage {
public:
  // Construct an 'all-unknown' memory image
  MemoryImage();
  // Construct an 'all-zero' memory image
  static MemoryImage zeros();
  // Construct an image for a 'word map'
  static MemoryImage fromWords(const std::map<uint32_t, uint32_t>& words);
  // Construct and image from elf files
  static MemoryImage fromElfs(const std::string& kernel, const std::string& user);
  // Construct and image from a single MM elf file
  static MemoryImage fromRawElf(const std::string& elf);

  // Returns a pointer to the page data, fails if unavailable
  PagePtr getPage(size_t page);
  // Sets the data for a page
  void setPage(size_t page, PagePtr data);
  // Gets a digest, fails if unavailable
  const Digest& getDigest(size_t idx) const;
  // Set a digest
  void setDigest(size_t idx, const Digest& digest);
  // Return a map of all 'known' pages
  std::map<uint32_t, PagePtr> getKnownPages() const;
  // Return a map of all 'known' digests
  std::map<uint32_t, Digest> getKnownDigests() const;

private:
  PagePtr zeroPage;
  std::vector<Digest> zeroDigests;

  // An nonexistant Page/Digest means the data is unavailable
  std::unordered_map<uint32_t, Digest> digests;
  std::unordered_map<uint32_t, PagePtr> pages;

  // Initialized the zeroPage + zeroDigests
  void initZeros();
  // Fixup digests after a change
  void fixupDigests(size_t idx);
  // Check is given MT node is a zero
  bool isZero(size_t idx);
  // Expand zero MT node (presume isZero(idx) returned true)
  void expandZero(size_t idx);
  // Do expansion if idx is a zero, return if expanded
  bool expandIfZero(size_t idx);
};

} // namespace zirgen::rv32im_v2
