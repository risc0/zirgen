// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace zirgen::r1csfile {

using WireID = uint32_t;
using LabelID = uint64_t;

using BigintLE = std::vector<uint8_t>;

struct Factor {
  WireID index;
  BigintLE value;
};

using Combination = std::vector<Factor>;

struct Constraint {
  Combination A;
  Combination B;
  Combination C;
};

struct Header {
  uint32_t fieldSize;
  BigintLE prime;
  uint32_t nWires;
  uint32_t nPubOut;
  uint32_t nPubIn;
  uint32_t nPrvIn;
  uint64_t nLabels;
  uint32_t nConstraints;
};

struct System {
  Header header;
  std::vector<Constraint> constraints;
  std::vector<LabelID> map;
};

std::unique_ptr<System> read(FILE*);

struct IOException : public std::runtime_error {
  IOException(const char*, const char*, int, const char*);
};

} // namespace zirgen::r1csfile
