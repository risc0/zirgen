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

#include <cstring>
#include <stdexcept>
#include "zirgen/Dialect/BigInt/Bytecode/bqint.h"

// Do not include any LLVM or MLIR headers!
// The BQInt mimics LLVM's APInt type, for use in the bytecode evaluator,
// which must not have any dependency on LLVM or MLIR.

namespace zirgen::BigInt::Bytecode {

namespace {

constexpr unsigned WORD_SIZE = sizeof(uint64_t);
constexpr unsigned BITS_PER_WORD = 64;

uint64_t *getClearedMemory(unsigned numWords) {
  return new uint64_t[numWords]();
}

uint64_t *getMemory(unsigned numWords) {
  return new uint64_t[numWords];
}

unsigned count_zeros(uint64_t val) {
  if (!val) {
    return BITS_PER_WORD;
  }
  unsigned zeroBits = 0;
  for (uint64_t shift = BITS_PER_WORD >> 1; shift; shift >>= 1) {
    uint64_t tmp = val >> shift;
    if (tmp) {
      val = tmp;
    } else {
      zeroBits |= shift;
    }
  }
  return zeroBits;
}

uint64_t tcAdd(uint64_t *dst, const uint64_t *rhs, unsigned parts) {
  uint64_t carry = 0;
  for (unsigned i = 0; i < parts; i++) {
    uint64_t l = dst[i];
    if (carry) {
      dst[i] += rhs[i] + 1;
      carry = (dst[i] <= l);
    } else {
      dst[i] += rhs[i];
      carry = (dst[i] < l);
    }
  }
  return carry;
}

uint64_t tcSubtractPart(uint64_t *dst, uint64_t src, unsigned parts) {
  for (unsigned i = 0; i < parts; ++i) {
    uint64_t Dst = dst[i];
    dst[i] -= src;
    if (src <= Dst) {
      return 0;
    }
    src = 1;
  }
  return 1;
}

} // namespace

BQInt::BQInt(unsigned numBits, uint64_t val, bool isSigned):
  BitWidth(numBits) {
  if (isSingleWord()) {
    U.word = val;
    clearUnusedBits();
  } else {
    if (isSigned && int64_t(val) < 0) {
      U.ptr = getMemory(getNumWords());
      U.ptr[0] = val;
      std::memset(&U.ptr[1], 0xFF, sizeof(uint64_t) * (getNumWords() - 1));
      clearUnusedBits();
    } else {
      U.ptr = getClearedMemory(getNumWords());
      U.ptr[0] = val;
    }
  }
}

BQInt::~BQInt() {
  if (isSingleWord()) {
    delete[] U.ptr;
  }
}

bool BQInt::isSingleWord() const {
  return BitWidth <= BITS_PER_WORD;
}

uint64_t BQInt::getLimitedValue(uint64_t Limit) const {
  return ugt(Limit) ? Limit : getZExtValue();
}

BQInt &BQInt::operator-=(uint64_t RHS) {
  if (isSingleWord()) {
    U.word -= RHS;
  } else {
    tcSubtractPart(U.ptr, RHS, getNumWords());
  }
  return clearUnusedBits();
}

BQInt &BQInt::operator+=(const BQInt &RHS) {
  if (BitWidth != RHS.BitWidth) {
    throw std::runtime_error("Bit widths must be the same");
  }
  if (isSingleWord()) {
    U.word += RHS.U.word;
  } else {
    tcAdd(U.ptr, RHS.U.ptr, getNumWords());
  }
  return clearUnusedBits();
}

BQInt &BQInt::operator*=(uint64_t RHS) {
  if (isSingleWord()) {
    U.word *= RHS;
  } else {
    unsigned NumWords = getNumWords();
    tcMultiplyPart(U.ptr, U.ptr, RHS, 0, NumWords, NumWords, false);
  }
  return clearUnusedBits();
}

BQInt &BQInt::operator<<=(unsigned ShiftAmt) {
  if (ShiftAmt > BitWidth) {
    throw std::runtime_error("Invalid shift amount");
  }
  if (isSingleWord()) {
    if (ShiftAmt == BitWidth) {
      U.word = 0;
    } else {
      U.word <<= ShiftAmt;
    }
  } else {
    tcShiftLeft(U.ptr, getNumWords(), ShiftAmt);
  }
  return clearUnusedBits();
}

BQInt BQInt::operator*(const BQInt &RHS) {
  if (BitWidth != RHS.BitWidth) {
    throw std::runtime_error("Bit widths must be the same");
  }
  if (isSingleWord()) {
    return BQInt(BitWidth, U.word * RHS.U.word);
  }
  BQInt Result(BitWidth, 0);
  tcMultiply(Result.U.ptr, U.ptr, RHS.U.ptr, getNumWords());
  Result.clearUnusedBits();
  return Result;
}

bool BQInt::operator!=(uint64_t Val) const {
  return !((*this) == Val);
}

BQInt BQInt::udiv(const BQInt &RHS) const {
  if (BitWidth != RHS.BitWidth) {
    throw std::runtime_error("Bit widths must be the same");
  }
  // First, deal with the easy case
  if (isSingleWord()) {
    if (RHS.U.word == 0) {
      throw std::runtime_error("Divide by zero?");
    }
    return BQInt(BitWidth, U.word / RHS.U.word);
  }
  // Get some facts about the LHS and RHS number of bits and words
  unsigned lhsWords = getNumWords(getActiveBits());
  unsigned rhsBits  = RHS.getActiveBits();
  unsigned rhsWords = getNumWords(rhsBits);
  if (!rhsWords) {
    throw std::runtime_error("Divided by zero???");
  }
  // Deal with some degenerate cases
  if (!lhsWords) {
    // 0 / X ===> 0
    return BQInt(BitWidth, 0);
  }
  if (rhsBits == 1) {
    // X / 1 ===> X
    return *this;
  }
  if (lhsWords < rhsWords || this->ult(RHS)) {
    // X / Y ===> 0, iff X < Y
    return BQInt(BitWidth, 0);
  }
  if (*this == RHS) {
    // X / X ===> 1
    return BQInt(BitWidth, 1);
  }
  if (lhsWords == 1) { // rhsWords is 1 if lhsWords is 1.
    // All high words are zero, just use native divide
    return BQInt(BitWidth, this->U.ptr[0] / RHS.U.ptr[0]);
  }
  // We have to compute it the hard way. Invoke the Knuth divide algorithm.
  BQInt Quotient(BitWidth, 0); // to hold result.
  divide(U.ptr, lhsWords, RHS.U.ptr, rhsWords, Quotient.U.ptr, nullptr);
  return Quotient;
}

BQInt BQInt::urem(const BQInt &RHS) const {
  if (BitWidth != RHS.BitWidth) {
    throw std::runtime_error("Bit widths must be the same");
  }
  if (isSingleWord()) {
    if (RHS.U.word == 0) {
      throw std::runtime_error("Remainder by zero?");
    }
    return BQInt(BitWidth, U.word % RHS.U.word);
  }
  // Get some facts about the LHS
  unsigned lhsWords = getNumWords(getActiveBits());
  // Get some facts about the RHS
  unsigned rhsBits = RHS.getActiveBits();
  unsigned rhsWords = getNumWords(rhsBits);
  if (!rhsWords) {
    throw std::runtime_error("Performing remainder operation by zero ???");
  }
  // Check the degenerate cases
  if (lhsWords == 0) {
    // 0 % Y ===> 0
    return BQInt(BitWidth, 0);
  }
  if (rhsBits == 1) {
    // X % 1 ===> 0
    return BQInt(BitWidth, 0);
  }
  if (lhsWords < rhsWords || this->ult(RHS)) {
    // X % Y ===> X, iff X < Y
    return *this;
  }
  if (*this == RHS) {
    // X % X == 0;
    return BQInt(BitWidth, 0);
  }
  if (lhsWords == 1) {
    // All high words are zero, just use native remainder
    return BQInt(BitWidth, U.ptr[0] % RHS.U.ptr[0]);
  }
  // We have to compute it the hard way. Invoke the Knuth divide algorithm.
  BQInt Remainder(BitWidth, 0);
  divide(U.ptr, lhsWords, RHS.U.ptr, rhsWords, nullptr, Remainder.U.ptr);
  return Remainder;
}

BQInt BQInt::smul_sat(const BQInt &RHS) const {
  bool overflow = false;
  BQInt res = smul_ov(RHS, overflow);
  if (!overflow) {
    return Res;
  }
  // The result is negative if one and only one of inputs is negative.
  if (isNegative() ^ RHS.isNegative()) {
    return BQInt::getSignedMinValue(BitWidth);
  } else {
    return BQInt::getSignedMaxValue(BitWidth);
  }
}

bool BQInt::intersects(const BQInt &RHS) const {
  if (BitWidth != RHS.BitWidth) {
    throw std::runtime_error("Bit widths must be the same");
  }
  if (isSingleWord()) {
    return (U.word & RHS.U.word) != 0;
  }
  for (unsigned i = 0, e = getNumWords(); i != e; ++i) {
    if ((U.ptr[i] & RHS.U.ptr[i]) != 0) {
      return true;
    }
  }
  return false;
}

BQInt BQInt::trunc(unsigned width) const {
  if (width > BitWidth) {
    throw std::runtime_error("cannot truncate to longer bit width");
  }
  if (width <= BITS_PER_WORD) {
    return BQInt(width, isSingleWord()? U.word: U.ptr[0]);
  }
  if (width == BitWidth) {
    return *this;
  }
  BQInt result(width, 0);
  // Copy full words.
  unsigned i;
  for (i = 0; i != width / BITS_PER_WORD; i++) {
    result.U.ptr[i] = U.ptr[i];
  }
  // Truncate and copy any partial word.
  unsigned bits = (0 - width) % BITS_PER_WORD;
  if (bits != 0) {
    result.U.ptr[i] = U.ptr[i] << bits >> bits;
  }
  return result;
}

BQInt BQInt::zext(unsigned width) const {
  if (width < BitWidth) {
    throw std::runtime_error("cannot extend to shorter bit width");
  }
  if (width <= BITS_PER_WORD) {
    return BQInt(width, U.word);
  }
  if (width == BitWidth) {
    return *this;
  }
  BQInt result(width, 0);
  // Copy words.
  if (isSingleWord()) {
    *result.U.ptr = U.word;
  } else {
    std::memcpy(result.U.ptr, U.ptr, getNumWords() * sizeof(uint64_t));
  }
  // Zero remaining words.
  size_t count = (result.getNumWords() - getNumWords()) * sizeof(uint64_t);
  std::memset(result.U.ptr + getNumWords(), 0, count);
  return result;
}

BQInt BQInt::extractBits(unsigned numBits, unsigned bitPosition) const {
  if (bitPosition > BitWidth || (numBits + bitPosition) > BitWidth) {
    throw std::runtime_error("Out-of-bounds bit extraction");
  }
  if (isSingleWord()) {
    return BQInt(numBits, U.word >> bitPosition);
  }
  unsigned loBit = bitPosition % BITS_PER_WORD;
  unsigned loWord = bitPosition / BITS_PER_WORD;
  unsigned hiWord = (bitPosition + numBits - 1) / BITS_PER_WORD;
  // Single word result extracting bits from a single word source.
  if (loWord == hiWord) {
    return BQInt(numBits, U.ptr[loWord] >> loBit);
  }
  // Extracting bits that start on a source word boundary can be done
  // as a fast memory copy.
  if (loBit == 0) {
    return BQInt(numBits, 1 + hiWord - loWord, U.ptr + loWord);
  }
  // General case - shift + copy source words directly into place.
  BQInt result(numBits, 0);
  unsigned NumSrcWords = getNumWords();
  unsigned NumDstWords = result.getNumWords();
  uint64_t *DestPtr = result.isSingleWord() ? &result.U.word : result.U.ptr;
  for (unsigned word = 0; word < NumDstWords; ++word) {
    uint64_t w0 = U.ptr[loWord + word];
    uint64_t comp = NumSrcWords ? U.ptr[loWord + word + 1] :0;
    uint64_t w1 = (loWord + word + 1) < comp;
    DestPtr[word] = (w0 >> loBit) | (w1 << (BITS_PER_WORD - loBit));
  }
  return result.clearUnusedBits();
}

unsigned BQInt::getNumWords() const {
  return getNumWords(BitWidth);
}

unsigned BQInt::getNumWords(unsigned BitWidth) {
  return ((uint64_t)BitWidth + BITS_PER_WORD - 1) / BITS_PER_WORD;
}


BQInt &BQInt::clearUnusedBits() {
  unsigned WordBits = ((BitWidth - 1) % BITS_PER_WORD) + 1;
  uint64_t mask = UINT64_MAX >> (BITS_PER_WORD - WordBits);
  if (BitWidth == 0) {
    mask = 0;
  }
  if (isSingleWord()) {
    U.word &= mask;
  } else {
    U.ptr[getNumWords() - 1] &= mask;
  }
  return *this;
}

void BQInt::initSlowCase(uint64_t val, bool isSigned) {
  if (isSigned && int64_t(val) < 0) {
    U.ptr = getMemory(getNumWords());
    U.ptr[0] = val;
    std::memset(&U.ptr[1], 0xFF, WORD_SIZE * (getNumWords() - 1));
    clearUnusedBits();
  } else {
    U.ptr = getClearedMemory(getNumWords());
    U.ptr[0] = val;
  }
}

bool BQInt::ugt(uint64_t RHS) const {
  return (!isSingleWord() && getActiveBits() > 64) || getZExtValue() > RHS;
}

bool BQInt::ult(const BQInt &RHS) const {
  return compare(RHS) < 0;
}

uint64_t BQInt::getZExtValue() const {
  if (isSingleWord()) {
    return U.word;
  }
  if (getActiveBits() > 64) {
    throw std::runtime_error("Too many bits for uint64_t");
  }
  return U.ptr[0];
}

unsigned BQInt::countl_zero() const {
  if (isSingleWord()) {
    unsigned unusedBits = BITS_PER_WORD - BitWidth;
    return count_zeros(U.word) - unusedBits;
  }
  return countLeadingZerosSlowCase();
}

unsigned BQInt::countLeadingZerosSlowCase() const {
  unsigned count = 0;
  for (int i = getNumWords()-1; i >= 0; --i) {
    uint64_t V = U.ptr[i];
    if (V == 0) {
      count += BITS_PER_WORD;
    } else {
      count += count_zeros(V);
      break;
    }
  }
  unsigned mod = BitWidth % BITS_PER_WORD;
  count -= mod > 0 ? BITS_PER_WORD - mod : 0;
  return count;
}

bool BQInt::operator==(uint64_t Val) const {
  return (isSingleWord() || getActiveBits() <= 64) && getZExtValue() == Val;
}

bool BQInt::operator==(const BQInt &RHS) const {
  if (BitWidth != RHS.BitWidth) {
    throw std::runtime_error("Comparison requires equal bit widths");
  }
  return isSingleWord()? U.word == RHS.U.word: equalSlowCase(RHS);
}

} // namespace zirgen::BigInt::Bytecode
