// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace zirgen::wtnsfile {

using BigintLE = std::vector<uint8_t>;

struct Header {
  uint32_t fieldSize;
  BigintLE prime;
  uint32_t nValues;
};

struct Witness {
  Header header;
  std::vector<BigintLE> values;
};

std::unique_ptr<Witness> read(FILE*);

struct IOException : public std::runtime_error {
  IOException(const char*, const char*, int, const char*);
};

} // namespace zirgen::wtnsfile
