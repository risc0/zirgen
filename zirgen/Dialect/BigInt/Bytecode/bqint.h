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

#pragma once

// BQInt replicates a subset of the llvm::APInt type for use in the bytecode
// evaluator, which cannot link against LLVM. Do not include any LLVM or
// MLIR headers here or in the implementation.
#include <cstdint>
#include <vector>

namespace zirgen::BigInt::Bytecode {

class BQInt {
  union {
    uint64_t word;
    uint64_t *ptr;
  } U;
  unsigned BitWidth = 1;
public:
  BQInt(unsigned numBits, uint64_t val, bool isSigned = false);
  BQInt(unsigned numBits, unsigned numWords, const uint64_t val[]);
  ~BQInt();
  bool isSingleWord() const;
  uint64_t getLimitedValue(uint64_t Limit = UINT64_MAX) const;
  BQInt &operator-=(uint64_t RHS);
  BQInt &operator+=(const BQInt &RHS);
  BQInt &operator*=(uint64_t RHS);
  BQInt &operator<<=(unsigned ShiftAmt);
  BQInt operator*(const BQInt &RHS);
  bool operator!=(uint64_t Val) const;
  BQInt udiv(const BQInt &RHS) const;
  BQInt urem(const BQInt &RHS) const;
  BQInt smul_sat(const BQInt &RHS) const;
  bool intersects(const BQInt &RHS) const;
  BQInt trunc(unsigned width) const;
  BQInt zext(unsigned width) const;
  BQInt extractBits(unsigned numBits, unsigned bitPosition) const;
  unsigned getBitWidth() const { return BitWidth; }
protected:
  // used internally but never by the eval function
  unsigned getNumWords() const;
  static unsigned getNumWords(unsigned);
  BQInt &clearUnusedBits();
  void initSlowCase(uint64_t val, bool isSigned);
  bool ugt(uint64_t RHS) const;
  bool ult(const BQInt &RHS) const;
  uint64_t getZExtValue() const;
  unsigned getActiveBits() const { return BitWidth - countl_zero(); }
  unsigned countl_zero() const;
  unsigned countLeadingZerosSlowCase() const;
  bool operator==(uint64_t Val) const;
  bool operator==(const BQInt &RHS) const;
};

inline BQInt operator-(BQInt a, uint64_t RHS) {
  a -= RHS;
  return a;
}

inline BQInt operator*(BQInt a, uint64_t RHS) {
  a *= RHS;
  return a;
}

} // namespace zirgen::BigInt::Bytecode
