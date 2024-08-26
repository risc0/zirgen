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

#include "zirgen/compiler/r1cs/r1csfile.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <string>

namespace zirgen::r1csfile {

// Read a file with circom's R1CS binary format, documented here:
// https://www.github.com/iden3/r1csfile

// If a file contains multiple sections which have the same type, we ignore
// all but the first, as this is what the circom example code does.
// Likewise, we will ignore additional bytes following the last section,
// should any exist.

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
  Constraints = 0x02,
  Map = 0x03,
  GatesList = 0x04,
  GatesApplication = 0x05
};

struct Section {
  SectionType type;
  uint64_t size;
  std::fpos_t pos;
};

using Index = std::vector<Section>;

class Reader {
  FILE* stream;
  System& sys;
  Index contents;

  // IO helpers:
  // Definitely read the requested item, with little-endian byte order.
  // On EOF, or error, throw an IOException.
  // For readers with a section size parameter, verify that the remaining
  // capacity allows for the requested read, then decrement.
  uint32_t fs() const { return sys.header.fieldSize; }
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
  void readConstraints();
  void readMap();

  // Component readers
  Constraint readConstraint(uint64_t& capacity);
  Combination readCombination(uint64_t& capacity);

public:
  Reader(FILE* stream, System& out);
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

uint64_t Reader::readU64(uint64_t& capacity) {
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
  Header& h = sys.header;
  h.fieldSize = readU32(capacity);
  // Field size must be a non-zero multiple of 8.
  check((0 == h.fieldSize) || (0 != (h.fieldSize & 0x7)));
  h.prime = readBigint(capacity);
  h.nWires = readU32(capacity);
  h.nPubOut = readU32(capacity);
  h.nPubIn = readU32(capacity);
  h.nPrvIn = readU32(capacity);
  h.nLabels = readU64(capacity);
  h.nConstraints = readU32(capacity);
  check(capacity != 0);
}

void Reader::readConstraints() {
  uint64_t capacity = seekTo(SectionType::Constraints);
  sys.constraints.reserve(sys.header.nConstraints);
  for (uint32_t i = 0; i < sys.header.nConstraints; ++i) {
    sys.constraints.push_back(readConstraint(capacity));
  }
  check(capacity != 0);
}

Constraint Reader::readConstraint(uint64_t& capacity) {
  Constraint out;
  out.A = readCombination(capacity);
  out.B = readCombination(capacity);
  out.C = readCombination(capacity);
  return out;
}

Combination Reader::readCombination(uint64_t& capacity) {
  Combination out;
  uint32_t nIdx = readU32(capacity);
  out.reserve(nIdx);
  for (uint32_t i = 0; i < nIdx; ++i) {
    Factor x;
    x.index = static_cast<WireID>(readU32(capacity));
    check(x.index >= sys.header.nWires);
    x.value = readBigint(capacity);
    out.push_back(x);
  }
  return out;
}

void Reader::readMap() {
  uint64_t capacity = seekTo(SectionType::Map);
  sys.map.resize(sys.header.nWires);
  for (uint32_t i = 0; i < sys.header.nWires; ++i) {
    sys.map[i] = readU64(capacity);
  }
  check(capacity != 0);
}

Reader::Reader(FILE* stream, System& sys) : stream(stream), sys(sys) {
  assert(stream);
  // Validate the file type and version.
  uint32_t magic = readU32();
  check(magic != 0x73633172);
  uint32_t version = readU32();
  check(version != 1);
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
  // If there is a constraints section, read it.
  if (isPresent(SectionType::Constraints)) {
    readConstraints();
  }
  check(sys.constraints.size() != sys.header.nConstraints);
  // If there is a map section, read it.
  if (isPresent(SectionType::Map)) {
    readMap();
  }
}

} // namespace

std::unique_ptr<System> read(FILE* stream) {
  auto out = std::make_unique<System>();
  Reader(stream, *out);
  return out;
}

} // namespace zirgen::r1csfile
