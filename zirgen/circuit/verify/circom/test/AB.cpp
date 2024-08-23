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

#include "zirgen/circuit/verify/circom/test/AB.h"
#include "zirgen/compiler/zkp/poseidon_254.h"

#include <gtest/gtest.h>

namespace zirgen::snark {

TEST(SNARK, p254) {
  P254 x = 1234567;
  P254 pow20 = 1;
  for (size_t i = 0; i < 20; i++) {
    pow20 = pow20 * x;
  }
  assert(pow20 ==
         P254("9704507806522716394566620077678775344142006794597453436923221800576174385187"));
  assert(P254(pow20.toDigest()) == pow20);
}

TEST(SNARK, Poseidon) {
  std::vector<uint32_t> data = {0, 1, 2, 3, 4};
  assert(hexDigest(poseidon254Hash(data.data(), data.size())) ==
         "8611cede0c468a53fc64c20d5aae2b6ddc0dd621713594d17febfca89a968618");
}

TEST(SNARK, ArithAB) {
  std::vector<uint32_t> iop;
  push_fp(iop, 1024);
  push_fp(iop, 12356);
  doAB(1, iop, [&](Buffer out, ReadIopVal iop) {
    Val a = iop.readBaseVals(1)[0];
    Val b = iop.readBaseVals(1)[0];
    Val c = 17 * a + b;
    for (size_t i = 0; i < 20; i++) {
      c = c * a + b - inv(c * c + b);
      c = c & (17 * b);
    }
    out[0] = 17 * c - 23 * b;
  });
}

TEST(SNARK, Hash) {
  std::vector<uint32_t> iop;
  for (size_t i = 1; i <= 5; i++) {
    push_fp(iop, i);
  }
  doAB(5, iop, [&](Buffer out, ReadIopVal iop) {
    auto vals = iop.readBaseVals(5);
    auto digest1 = hash(vals);
    auto digest2 = fold(digest1, digest1);
    auto digest3 = fold(digest1, digest2);
    iop.commit(digest3);
    out[0] = iop.rngBits(7);
    out[1] = iop.rngBaseVal();
    for (size_t i = 0; i < 23; i++) {
      vals.push_back(iop.rngBaseVal());
    }
    iop.commit(hash(vals));
    out[2] = iop.rngBaseVal();
    Val a = iop.rngBaseVal();
    Val b = iop.rngBaseVal();
    Val c = iop.rngBaseVal();
    Val d = iop.rngBaseVal();
    out[3] = select(iop.rngBits(2), {a, b, c, d});
    iop.commit(select(a & 1, {hash({b}), hash({c})}));
    out[4] = iop.rngBaseVal();
  });
}

TEST(SNARK, HashEncode) {
  std::vector<uint32_t> huh = {5};
  std::vector<uint32_t> iop;
  push_fp(iop, 5);
  Digest d = poseidon254Hash(huh.data(), 1);
  push_digest(iop, d);
  doAB(1, iop, [&](Buffer out, ReadIopVal iop) {
    Val in = iop.readBaseVals(1)[0];
    DigestVal digest1 = iop.readDigests(1)[0];
    DigestVal digest2 = hash({in});
    assert_eq(digest1, digest2);
    auto digest3 = fold(digest1, digest2);
    iop.commit(digest3);
    out[0] = in + iop.rngBaseVal();
  });
}

} // namespace zirgen::snark
