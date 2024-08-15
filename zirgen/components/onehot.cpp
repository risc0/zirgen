// Copyright (c) 2023 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/components/onehot.h"

#include "zirgen/components/bits.h"

namespace zirgen {

OneHot::OneHot(Buffer buf, const std::vector<uint64_t>& options) : buf(buf), options(options) {
  assert(buf.size() == options.size());
}

void OneHot::set(Val val) {
  NONDET {
    for (size_t i = 0; i < buf.size(); i++) {
      buf[i] = isz(val - options[i]);
    }
  }
  isBits(buf);
  eq(get(), val);
  Val tot = 0;
  for (size_t i = 0; i < buf.size(); i++) {
    tot = tot + buf[i];
  }
  eq(tot, 1);
}

Val OneHot::get() {
  Val tot = 0;
  for (size_t i = 0; i < buf.size(); i++) {
    tot = tot + options[i] * buf[i];
  }
  return tot;
}

Val OneHot::is(uint64_t val) {
  for (size_t i = 0; i < buf.size(); i++) {
    if (options[i] == val) {
      return buf[i];
    }
  }
  throw std::runtime_error("Invalid 'is' in onehot");
}

Val OneHot::isIdx(size_t idx) {
  return buf[idx];
}

} // namespace zirgen
