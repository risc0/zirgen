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

#include "zirgen/circuit/bigint/test/rsa_helper.h"
#include "zirgen/Dialect/BigInt/IR/BigInt.h"

using namespace zirgen;
using namespace mlir;

namespace zirgen::BigInt::test {
// Used for testing, this runs a full RSA using `Def` instead of `Load`/`Store`
void testMakeRSAChecker(OpBuilder builder, Location loc, size_t bits) {
  // Check if (S^e = M (mod N)), where e = 65537
  auto N = builder.create<BigInt::DefOp>(loc, bits, 0, true, bits - 1);
  auto S = builder.create<BigInt::DefOp>(loc, bits, 1, true);
  auto M = builder.create<BigInt::DefOp>(loc, bits, 2, true);
  // We square S 16 times to get S^65536
  Value x = S;
  for (size_t i = 0; i < 16; i++) {
    auto xm = builder.create<BigInt::MulOp>(loc, x, x);
    x = builder.create<BigInt::ReduceOp>(loc, xm, N);
  }
  // Multiply in one more copy of S + reduce
  auto xm = builder.create<BigInt::MulOp>(loc, x, S);
  x = builder.create<BigInt::ReduceOp>(loc, xm, N);
  // Subtract M and see if it's zero
  auto diff = builder.create<BigInt::SubOp>(loc, x, M);
  builder.create<BigInt::EqualZeroOp>(loc, diff);
}

// Used for testing, to compute expected outputs.
// I verified this by comparing against:
// pow(S, 65537, N) in python
APInt testRSA(APInt N, APInt S) {
  size_t width = S.getBitWidth();
  N = N.zext(2 * width);
  S = S.zext(2 * width);
  APInt cur = S;
  for (size_t i = 0; i < 16; i++) {
    cur = (cur * cur).urem(N);
  }
  cur = (cur * S).urem(N);
  return cur.trunc(width);
}
} // namespace zirgen::BigInt::test
