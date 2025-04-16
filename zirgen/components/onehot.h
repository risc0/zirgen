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

#pragma once

#include "zirgen/components/bits.h"

namespace zirgen {

template <size_t size> class OneHotImpl : public CompImpl<OneHotImpl<size>> {
public:
  OneHotImpl(llvm::StringRef source = "data", bool check = true) : OneHotImpl({}, source, check) {}
  OneHotImpl(std::vector<const char*> labels, llvm::StringRef source = "data", bool check = true) {
    assert(labels.empty() || labels.size() == size);
    for (size_t i = 0; i < size; i++) {
      bits.emplace_back(labels.empty() ? Label("hot", i) : Label(labels[i]), source);
    }
    if (check) {
      this->registerCallback("_builtin_verify", &OneHotImpl::onVerify);
    }
  }

  void onVerify() {
    // Make sure exactly one bit is set + all elements are bits
    Val tot = 0;
    for (size_t i = 0; i < size; i++) {
      eqz(bits[i] * (1 - bits[i]));
      tot = tot + bits[i];
    }
    eq(tot, 1);
  }

  void set(Val val) {
    NONDET {
      for (size_t i = 0; i < size; i++) {
        bits[i]->set(isz(val - i));
      }
    }
    eq(get(), val);
  }

  Val get() {
    Val tot = 0;
    for (size_t i = 0; i < size; i++) {
      tot = tot + i * bits[i];
    }
    return tot;
  }

  Val at(size_t idx, mlir::Location loc = currentLoc()) {
    ScopedLocation guard(loc);
    return bits[idx];
  }

  // Special access for mux
  Buffer atRaw(size_t idx) { return bits[idx]->raw(); }

private:
  std::vector<Reg> bits;
};

template <size_t size> using OneHot = Comp<OneHotImpl<size>>;

} // namespace zirgen
