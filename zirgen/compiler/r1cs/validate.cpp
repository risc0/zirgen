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

#include <iostream>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Endian.h"

#include "zirgen/compiler/r1cs/validate.h"

namespace zirgen::R1CS {

namespace {

// T should be akin to r1csfile::BigintLE or wtnsfile::BigintLE
template <typename T> llvm::APInt cvtint(T& in) {
  assert(0 == (in.size() & 0x07));
  unsigned words = in.size() >> 3;
  unsigned bits = in.size() << 3;
  std::vector<uint64_t> bigVal(words);
  for (unsigned i = 0; i < words; ++i) {
    // compute the position from the input array
    size_t offset = i << 3;
    // shift & merge bytes into 64-bit value
    bigVal[i] = llvm::support::endian::read64le(&in[offset]);
  }
  llvm::APInt value(bits, words, bigVal.data());
  return value;
}

struct Impl {
  Impl(const wtnsfile::Witness& wtns);
  llvm::APInt sum(llvm::APInt lhs, llvm::APInt rhs);
  llvm::APInt prod(llvm::APInt lhs, llvm::APInt rhs);
  llvm::APInt mod(llvm::APInt lhs, llvm::APInt rhs);
  llvm::APInt diff(llvm::APInt lhs, llvm::APInt rhs);
  llvm::APInt eval(const r1csfile::Factor& factor);
  llvm::APInt eval(const r1csfile::Combination& combination);
  void check(const r1csfile::Constraint& constraint);

private:
  llvm::APInt prime;
  std::vector<llvm::APInt> witvals;
};

Impl::Impl(const wtnsfile::Witness& wtns) : prime(cvtint(wtns.header.prime)) {
  witvals.resize(wtns.values.size());
  for (size_t i = 0; i < wtns.values.size(); ++i) {
    witvals[i] = cvtint(wtns.values[i]);
  }
}

llvm::APInt Impl::sum(llvm::APInt lhs, llvm::APInt rhs) {
  lhs = mod(lhs, prime);
  rhs = mod(rhs, prime);
  unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  return mod(lhs.zext(width) + rhs.zext(width), prime);
}

llvm::APInt Impl::prod(llvm::APInt lhs, llvm::APInt rhs) {
  lhs = mod(lhs, prime);
  rhs = mod(rhs, prime);
  unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  return mod(lhs.zext(width) * rhs.zext(width), prime);
}

llvm::APInt Impl::mod(llvm::APInt lhs, llvm::APInt rhs) {
  unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  lhs = lhs.zext(width);
  rhs = rhs.zext(width);
  // x % y = x - (y * floor(x/y))
  return lhs - (rhs * lhs.sdiv(rhs));
}

llvm::APInt Impl::diff(llvm::APInt lhs, llvm::APInt rhs) {
  rhs.negate();
  return sum(lhs, rhs);
}

llvm::APInt Impl::eval(const r1csfile::Factor& factor) {
  assert(factor.index < witvals.size());
  llvm::APInt lhs = witvals[factor.index];
  llvm::APInt rhs = cvtint(factor.value);
  return prod(lhs, rhs);
}

llvm::APInt Impl::eval(const r1csfile::Combination& combination) {
  llvm::APInt out;
  for (auto& factor : combination) {
    out = sum(out, eval(factor));
  }
  return out;
}

void Impl::check(const r1csfile::Constraint& constraint) {
  // Verify that A*B-C=0.
  llvm::APInt a = eval(constraint.A);
  llvm::APInt b = eval(constraint.B);
  llvm::APInt c = eval(constraint.C);
  llvm::APInt result = diff(prod(a, b), c);
  assert(result.isZero());
}

} // namespace

void validate(const r1csfile::System& sys, const wtnsfile::Witness& wtns) {
  assert(sys.header.fieldSize == wtns.header.fieldSize);
  assert(sys.header.prime.size() == wtns.header.prime.size());
  for (size_t i = 0; i < sys.header.prime.size(); ++i) {
    assert(sys.header.prime[i] == wtns.header.prime[i]);
  }
  assert(sys.header.nWires == wtns.header.nValues);
  Impl validator(wtns);
  // Verify each constraint
  for (auto& constraint : sys.constraints) {
    validator.check(constraint);
  }
}

} // namespace zirgen::R1CS
