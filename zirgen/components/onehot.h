// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

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

  Val at(size_t idx, SourceLoc loc = current()) {
    OverrideLocation guard(loc);
    return bits[idx];
  }

  // Special access for mux
  Buffer atRaw(size_t idx) { return bits[idx]->raw(); }

private:
  std::vector<Reg> bits;
};

template <size_t size> using OneHot = Comp<OneHotImpl<size>>;

} // namespace zirgen
