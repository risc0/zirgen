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

#include "zirgen/compiler/edsl/edsl.h"

namespace zirgen::predicates {

constexpr size_t kDigestHalfs = 16;

struct U32Reg {
  constexpr static size_t size = 4;
  // Default constructor
  U32Reg() = default;
  // Construct via reading from a stream
  U32Reg(llvm::ArrayRef<Val>& stream);
  // Write to an output
  void write(std::vector<Val>& stream);

  Val flat();

  static U32Reg zero();

  std::array<Val, 4> val;
};

struct SystemState {
  constexpr static size_t size = kDigestHalfs + U32Reg::size;
  // Default constructor
  SystemState() = default;
  // Construct via reading from a stream
  SystemState(llvm::ArrayRef<Val>& stream, bool longDigest = false);
  // Write to an output
  void write(std::vector<Val>& stream);
  // Digest into a single value
  DigestVal digest();

  U32Reg pc;
  DigestVal memory;
};

struct ReceiptClaim {
  constexpr static size_t size = 2 * kDigestHalfs + 2 * SystemState::size + 2;
  // Default constructor
  ReceiptClaim() = default;
  // Construct via reading from a stream
  ReceiptClaim(llvm::ArrayRef<Val>& stream, bool longDigest = false);
  // Write to an output
  void write(std::vector<Val>& stream);
  // Digest into a single value
  DigestVal digest();

  static ReceiptClaim fromRv32imV2(llvm::ArrayRef<Val>& stream, size_t po2);

  DigestVal input;
  SystemState pre;
  SystemState post;
  Val sysExit;
  Val userExit;
  DigestVal output;
};

struct Output {
  // Default constructor
  Output() = default;
  // Digest into a single value
  DigestVal digest();

  /// Digest of the journal committed to by the guest execution.
  DigestVal journal;

  /// Digest of an ordered list of Assumption digests corresponding to the
  /// calls to `env::verify` and `env::verify_integrity`.
  DigestVal assumptions;
};

struct Assumption {
  constexpr static size_t size = 2 * kDigestHalfs;
  // Default constructor
  Assumption() = default;
  // Construct via reading from a stream
  Assumption(llvm::ArrayRef<Val>& stream, bool longDigest = false);
  // Digest into a single value
  DigestVal digest();

  DigestVal claim;
  DigestVal controlRoot;
};

struct UnionClaim {
  constexpr static size_t size = 2 * kDigestHalfs;
  // Default constructor
  UnionClaim() = default;
  // Write to an output
  void write(std::vector<Val>& stream);
  // Digest into a single value
  DigestVal digest();

  DigestVal left;
  DigestVal right;
};

// ReciptClaim lift(size_t po2, ReadIopVal seal);

ReceiptClaim join(ReceiptClaim in1, ReceiptClaim in2);
ReceiptClaim identity(ReceiptClaim in);
ReceiptClaim resolve(ReceiptClaim cond, Assumption assum, DigestVal tail, DigestVal journal);

DigestVal readSha(llvm::ArrayRef<Val>& stream, bool longDigest = false);
void writeSha(DigestVal val, std::vector<Val>& stream);

// Cannot be called "union" as that is a keyword.
UnionClaim unionFunc(Assumption left, Assumption right);

} // namespace zirgen::predicates
