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
