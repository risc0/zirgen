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
