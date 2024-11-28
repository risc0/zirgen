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
#include <vector>

#include "risc0/fp/fp.h"
#include "risc0/fp/fpext.h"

namespace zirgen::keccak2 {

using Fp = risc0::Fp;
using FpExt = risc0::FpExt;
using KeccakState = std::array<uint64_t, 25>;

class TraceGroup {
public:
  TraceGroup(size_t rows, size_t cols);
  size_t getRows() { return rows; }
  size_t getCols() { return cols; }

  void set(size_t row, size_t col, Fp val);
  Fp get(size_t row, size_t col);
  void setUnset();

  void setUnsafe(bool val = true);

private:
  size_t rows;
  size_t cols;
  std::vector<Fp> vec;
  bool unsafeReads;
};

class GlobalTraceGroup {
public:
  GlobalTraceGroup(size_t cols);
  size_t getCols() { return cols; }

  void set(size_t col, Fp val);
  Fp get(size_t col);

  void setUnsafe(bool val = true);

private:
  size_t cols;
  std::vector<Fp> vec;
  bool unsafeReads;
};

struct CircuitParams {
  size_t dataCols;
  size_t globalCols;
};

struct ExecutionTrace {
  ExecutionTrace(size_t rows, const CircuitParams& params);
  TraceGroup data;
  GlobalTraceGroup global;
};

} // namespace zirgen::keccak2
