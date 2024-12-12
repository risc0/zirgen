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

#include "zirgen/circuit/keccak2/cpp/wrap_dsl.h"
#include "zirgen/compiler/zkp/sha256.h"

/* The below implementation (up till the end of keccackf) is taken from:
   https://github.com/brainhub/SHA3IUF/blob/master/sha3.c
*/

#define SHA3_CONST(x) x##L

#define SHA3_ROTL64(x, y) (((x) << (y)) | ((x) >> ((sizeof(uint64_t) * 8) - (y))))

static const uint64_t keccakf_rndc[24] = {
    SHA3_CONST(0x0000000000000001UL), SHA3_CONST(0x0000000000008082UL),
    SHA3_CONST(0x800000000000808aUL), SHA3_CONST(0x8000000080008000UL),
    SHA3_CONST(0x000000000000808bUL), SHA3_CONST(0x0000000080000001UL),
    SHA3_CONST(0x8000000080008081UL), SHA3_CONST(0x8000000000008009UL),
    SHA3_CONST(0x000000000000008aUL), SHA3_CONST(0x0000000000000088UL),
    SHA3_CONST(0x0000000080008009UL), SHA3_CONST(0x000000008000000aUL),
    SHA3_CONST(0x000000008000808bUL), SHA3_CONST(0x800000000000008bUL),
    SHA3_CONST(0x8000000000008089UL), SHA3_CONST(0x8000000000008003UL),
    SHA3_CONST(0x8000000000008002UL), SHA3_CONST(0x8000000000000080UL),
    SHA3_CONST(0x000000000000800aUL), SHA3_CONST(0x800000008000000aUL),
    SHA3_CONST(0x8000000080008081UL), SHA3_CONST(0x8000000000008080UL),
    SHA3_CONST(0x0000000080000001UL), SHA3_CONST(0x8000000080008008UL)};

static const unsigned keccakf_rotc[24] = {1,  3,  6,  10, 15, 21, 28, 36, 45, 55, 2,  14,
                                          27, 41, 56, 8,  25, 43, 62, 18, 39, 61, 20, 44};

static const unsigned keccakf_piln[24] = {10, 7,  11, 17, 18, 3, 5,  16, 8,  21, 24, 4,
                                          15, 23, 19, 13, 12, 2, 20, 14, 22, 9,  6,  1};

static void keccakf(uint64_t s[25]) {
  int i, j, round;
  uint64_t t, bc[5];
#define KECCAK_ROUNDS 24

  for (round = 0; round < KECCAK_ROUNDS; round++) {
    std::cout << "Round: " << round << ", s[0] = " << std::hex << s[0] << std::dec << "\n";

    /* Theta */
    for (i = 0; i < 5; i++)
      bc[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];

    for (i = 0; i < 5; i++) {
      t = bc[(i + 4) % 5] ^ SHA3_ROTL64(bc[(i + 1) % 5], 1);
      for (j = 0; j < 25; j += 5)
        s[j + i] ^= t;
    }

    /* Rho Pi */
    t = s[1];
    for (i = 0; i < 24; i++) {
      j = keccakf_piln[i];
      bc[0] = s[j];
      s[j] = SHA3_ROTL64(t, keccakf_rotc[i]);
      t = bc[0];
    }

    /* Chi */
    for (j = 0; j < 25; j += 5) {
      for (i = 0; i < 5; i++)
        bc[i] = s[j + i];
      for (i = 0; i < 5; i++)
        s[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
    }

    /* Iota */
    s[0] ^= keccakf_rndc[round];
  }
  std::cout << "Final: " << round << ", s[0] = " << std::hex << s[0] << std::dec << "\n";
}

void ShaSingleKeccak(zirgen::Digest& digest, zirgen::keccak2::KeccakState state) {
  std::vector<uint32_t> toHash(64);
  uint32_t* viewState = (uint32_t*)&state;
  for (size_t i = 0; i < 50; i++) {
    toHash[i] = htonl(viewState[i]);
  }
  for (size_t i = 0; i < 4; i++) {
    zirgen::impl::compress(digest, toHash.data() + i * 16);
  }
}

void DoTransaction(zirgen::Digest& digest, zirgen::keccak2::KeccakState state) {
  ShaSingleKeccak(digest, state);
  std::cout << "After compressing input: " << digest << "\n";
  keccakf(state.data());
  ShaSingleKeccak(digest, state);
  std::cout << "After compressing output: " << digest << "\n";
}

int main() {
  // Make an example
  using namespace zirgen::keccak2;
  KeccakState state;
  uint64_t pows = 987654321;
  for (size_t i = 0; i < state.size(); i++) {
    printf("[%zu]: 0x%lx\n", i, pows);
    state[i] = pows;
    pows *= 123456789;
  }

  // Compute what the circuit should say
  zirgen::Digest digest = zirgen::impl::initState();
  DoTransaction(digest, state);

  // Now run the circuit
  size_t cycles = 200;
  std::vector<KeccakState> inputs;
  inputs.push_back(state);
  // Do the preflight
  auto preflight = preflightSegment(inputs, cycles);
  // Make the execution trace
  auto trace = ExecutionTrace(cycles, getDslParams());
  // Apply the preflight (i.e. scatter)
  applyPreflight(trace, preflight);
  // Run backwords
  std::cout << "out.ctypeOneHot = " << getLayoutInfo().ctypeOneHot << "\n";
  for (size_t i = cycles; i-- > 0;) {
    StepHandler ctx(preflight, i);
    std::cout << "Running cycle " << i << "\n";
    DslStep(ctx, trace, i);
  }

  // Make sure the results match
  zirgen::Digest compare;
  for (size_t i = 0; i < 8; i++) {
    uint32_t elem =
        trace.global.get(2 * i).asUInt32() | (trace.global.get(2 * i + 1).asUInt32() << 16);
    compare.words[i] = elem;
  }
  std::cout << "From circuit: " << compare << "\n";
  if (compare != digest) {
    throw std::runtime_error("Mismatch!\n");
  }
}
