// Copyright 2025 RISC Zero, Inc.
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

#include "zirgen/compiler/r1cs/wtnsfile.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <string>

namespace zirgen::wtnsfile {

IOException::IOException(const char* file, const char* func, int line, const char* msg)
    : std::runtime_error::runtime_error(std::string(file) + ":" + std::to_string(line) +
                                        ", check failed in " + std::string(func) + ": " +
                                        std::string(msg)) {}

#define check(cond)                                                                                \
  if (cond)                                                                                        \
  throw IOException(__FILE__, __FUNCTION__, __LINE__, #cond)

namespace {

enum class SectionType : uint32_t {
  Header = 0x01,
  Witness = 0x02,
};

struct Section {
  SectionType type;
  uint64_t size;
  std::fpos_t pos;
};

using Index = std::vector<Section>;

class Reader {
  FILE* stream;
  Witness& wit;
  Index contents;

  // IO helpers:
  // Definitely read the requested item, with little-endian byte order.
  // On EOF, or error, throw an IOException.
  // For readers with a section size parameter, verify that the remaining
  // capacity allows for the requested read, then decrement.
  uint32_t fs() const { return wit.header.fieldSize; }
  uint32_t readU32();
  uint64_t readU64();
  BigintLE readBigint();
  uint32_t readU32(uint64_t& cap);
  uint64_t readU64(uint64_t& cap);
  BigintLE readBigint(uint64_t& cap);

  uint64_t seekTo(SectionType type);
  bool isPresent(SectionType type);

  // Section readers
  void readHeader();
  void readWitness();

public:
  Reader(FILE* stream, Witness& out);
};

uint32_t Reader::readU32() {
  check(feof(stream));
  std::array<uint8_t, 4> buf;
  size_t got = fread(buf.data(), buf.size(), 1, stream);
  check(ferror(stream) || !got);
  return (static_cast<uint32_t>(buf[0]) << 0x00) | (static_cast<uint32_t>(buf[1]) << 0x08) |
         (static_cast<uint32_t>(buf[2]) << 0x10) | (static_cast<uint32_t>(buf[3]) << 0x18);
}

uint32_t Reader::readU32(uint64_t& capacity) {
  check(capacity < sizeof(uint32_t));
  capacity -= sizeof(uint32_t);
  return readU32();
}

uint64_t Reader::readU64() {
  check(feof(stream));
  std::array<uint8_t, 8> buf;
  size_t got = fread(buf.data(), buf.size(), 1, stream);
  check(ferror(stream) || !got);
  return (static_cast<uint64_t>(buf[0]) << 0x00) | (static_cast<uint64_t>(buf[1]) << 0x08) |
         (static_cast<uint64_t>(buf[2]) << 0x10) | (static_cast<uint64_t>(buf[3]) << 0x18) |
         (static_cast<uint64_t>(buf[4]) << 0x20) | (static_cast<uint64_t>(buf[5]) << 0x28) |
         (static_cast<uint64_t>(buf[6]) << 0x30) | (static_cast<uint64_t>(buf[7]) << 0x38);
}

[[maybe_unused]] uint64_t Reader::readU64(uint64_t& capacity) {
  check(capacity < sizeof(uint64_t));
  capacity -= sizeof(uint64_t);
  return readU64();
}

BigintLE Reader::readBigint() {
  BigintLE out;
  out.resize(fs());
  check(feof(stream));
  size_t got = fread(out.data(), out.size(), 1, stream);
  check(ferror(stream) || !got);
  return out;
}

BigintLE Reader::readBigint(uint64_t& capacity) {
  check(capacity < fs());
  capacity -= fs();
  return readBigint();
}

uint64_t Reader::seekTo(SectionType type) {
  for (auto& section : contents) {
    if (section.type == type) {
      int err = fsetpos(stream, &section.pos);
      check(ferror(stream) || err != 0);
      return section.size;
    }
  }
  check(true);
}

bool Reader::isPresent(SectionType type) {
  for (auto& section : contents) {
    if (section.type == type) {
      return true;
    }
  }
  return false;
}

void Reader::readHeader() {
  uint64_t capacity = seekTo(SectionType::Header);
  Header& h = wit.header;
  h.fieldSize = readU32(capacity);
  // Field size must be a non-zero multiple of 8.
  check((0 == h.fieldSize) || (0 != (h.fieldSize & 0x7)));
  h.prime = readBigint(capacity);
  h.nValues = readU32(capacity);
  check(capacity != 0);
}

void Reader::readWitness() {
  uint64_t capacity = seekTo(SectionType::Witness);
  wit.values.reserve(wit.header.nValues);
  for (uint32_t i = 0; i < wit.header.nValues; ++i) {
    wit.values.push_back(readBigint(capacity));
  }
  check(capacity != 0);
}

Reader::Reader(FILE* stream, Witness& wit) : stream(stream), wit(wit) {
  assert(stream);
  // Validate the file type and version.
  uint32_t magic = readU32();
  check(magic != 0x736E7477);
  uint32_t version = readU32();
  check(version != 2);
  uint32_t nSections = readU32();
  check(0 == nSections);
  // Read the type, size, and location of each section.
  for (uint32_t i = 0; i < nSections; ++i) {
    Section s;
    s.type = static_cast<SectionType>(readU32());
    s.size = readU64();
    int err = fgetpos(stream, &s.pos);
    check(ferror(stream) || err != 0);
    contents.push_back(s);
    // Skip the contents of the section.
    // There is no platform-independent way to seek past 2GB, so we will
    // repeatedly fseek in chunks, until we have moved the whole distance.
    uint64_t skip = s.size;
    while (skip > 0) {
      const uint64_t k2GB = 0x7FFFFFFFUL;
      uint64_t step = std::min(k2GB, skip);
      skip -= step;
      int err = fseek(stream, step, SEEK_CUR);
      check(ferror(stream) || err != 0);
    }
  }
  // Read the header section. This must be present, since it defines the
  // field size, without which we could not parse the other sections.
  readHeader();
  // If there is a witness values section, read it.
  if (isPresent(SectionType::Witness)) {
    readWitness();
  }
  check(wit.values.size() != wit.header.nValues);
}

} // namespace

std::unique_ptr<Witness> read(FILE* stream) {
  auto out = std::make_unique<Witness>();
  Reader(stream, *out);
  return out;
}

} // namespace zirgen::wtnsfile
