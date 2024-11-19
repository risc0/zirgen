// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include <vector>

#include "risc0/fp/fp.h"
#include "risc0/fp/fpext.h"

namespace zirgen::rv32im_v2 {

using Fp = risc0::Fp;
using FpExt = risc0::FpExt;

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
  size_t accumCols;
  size_t mixCols;
};

struct ExecutionTrace {
  ExecutionTrace(size_t rows, const CircuitParams& params);
  TraceGroup data;
  GlobalTraceGroup global;
  TraceGroup accum;
  GlobalTraceGroup mix;
};

} // namespace zirgen::rv32im_v2
