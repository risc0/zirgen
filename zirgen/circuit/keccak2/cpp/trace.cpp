// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/keccak2/cpp/trace.h"
#include <iostream>

namespace zirgen::keccak2 {

TraceGroup::TraceGroup(size_t rows, size_t cols)
    : rows(rows), cols(cols), vec(rows * cols, Fp::invalid()), unsafeReads(false) {}

void TraceGroup::set(size_t row, size_t col, Fp val) {
  Fp& elem = vec[row * cols + col];
  if (elem != Fp::invalid() && elem != val) {
    if (col == 5 && val == 0) {
      return;
    }
    std::cerr << "Invalid trace set: row = " << row << ", col = " << col << "\n";
    std::cerr << "Current = " << elem.asUInt32() << ", new = " << val.asUInt32() << "\n";
    throw std::runtime_error("Inconsistant set");
  }
  elem = val;
}

Fp TraceGroup::get(size_t row, size_t col) {
  Fp ret = vec[row * cols + col];
  if (ret == Fp::invalid() && !unsafeReads) {
    std::cerr << "Invalid trace get: row = " << row << ", col = " << col << "\n";
    throw std::runtime_error("Read of unset value");
  }
  return ret;
}

void TraceGroup::setUnset() {
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i] == Fp::invalid()) {
      vec[i] = 0;
    }
  }
}

void TraceGroup::setUnsafe(bool val) {
  unsafeReads = true;
}

GlobalTraceGroup::GlobalTraceGroup(size_t cols)
    : cols(cols), vec(cols, Fp::invalid()), unsafeReads(false) {}

void GlobalTraceGroup::set(size_t col, Fp val) {
  Fp& elem = vec[col];
  if (elem != Fp::invalid() && elem != val) {
    std::cerr << "Invalid global trace set: col = " << col << "\n";
    std::cerr << "Current = " << elem.asUInt32() << ", new = " << val.asUInt32() << "\n";
    throw std::runtime_error("Inconsistant set");
  }
  elem = val;
}

Fp GlobalTraceGroup::get(size_t col) {
  Fp ret = vec[col];
  if (ret == Fp::invalid() && !unsafeReads) {
    std::cerr << "Invalid global trace get: col = " << col << "\n";
    throw std::runtime_error("Read of unset value");
  }
  return ret;
}

void GlobalTraceGroup::setUnsafe(bool val) {
  unsafeReads = true;
}

ExecutionTrace::ExecutionTrace(size_t rows, const CircuitParams& params)
    : data(rows, params.dataCols)
    , global(params.globalCols)
    , accum(rows, params.accumCols)
    , mix(params.mixCols) {}

} // namespace zirgen::keccak2
