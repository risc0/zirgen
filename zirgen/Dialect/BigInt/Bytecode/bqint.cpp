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

uint64_t lowBitMask(unsigned bits) {
  if (bits == 0 || bits > BITS_PER_WORD) {
    throw std::runtime_error("malformed bit mask");
  }
  return ~(uint64_t) 0 >> (BITS_PER_WORD - bits);
}

uint64_t lowHalf(uint64_t part) {
  return part & lowBitMask(BITS_PER_WORD / 2);
}

uint64_t highHalf(uint64_t part) {
  return part >> (BITS_PER_WORD / 2);
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

/// DST += SRC * MULTIPLIER + CARRY   if add is true
/// DST  = SRC * MULTIPLIER + CARRY   if add is false
/// Requires 0 <= DSTPARTS <= SRCPARTS + 1.  If DST overlaps SRC
/// they must start at the same point, i.e. DST == SRC.
/// If DSTPARTS == SRCPARTS + 1 no overflow occurs and zero is
/// returned.  Otherwise DST is filled with the least significant
/// DSTPARTS parts of the result, and if all of the omitted higher
/// parts were zero return zero, otherwise overflow occurred and
/// return one.
int tcMultiplyPart(uint64_t *dst, const uint64_t *src,
                          uint64_t multiplier, uint64_t carry,
                          unsigned srcParts, unsigned dstParts,
                          bool add) {
  // Otherwise our writes of DST kill our later reads of SRC.
  if (dst > src && dst < src + srcParts) {
    throw std::runtime_error("overlapping source and dest buffers");
  }
  if (dstParts > srcParts + 1) {
    throw std::runtime_error("dstParts within srcParts buffer");
  }

  // N loops; minimum of dstParts and srcParts.
  unsigned n = std::min(dstParts, srcParts);

  for (unsigned i = 0; i < n; i++) {
    // [LOW, HIGH] = MULTIPLIER * SRC[i] + DST[i] + CARRY.
    // This cannot overflow, because:
    //   (n - 1) * (n - 1) + 2 (n - 1) = (n - 1) * (n + 1)
    // which is less than n^2.
    uint64_t srcPart = src[i];
    uint64_t low, mid, high;
    if (multiplier == 0 || srcPart == 0) {
      low = carry;
      high = 0;
    } else {
      low = lowHalf(srcPart) * lowHalf(multiplier);
      high = highHalf(srcPart) * highHalf(multiplier);

      mid = lowHalf(srcPart) * highHalf(multiplier);
      high += highHalf(mid);
      mid <<= BITS_PER_WORD / 2;
      if (low + mid < low)
        high++;
      low += mid;

      mid = highHalf(srcPart) * lowHalf(multiplier);
      high += highHalf(mid);
      mid <<= BITS_PER_WORD / 2;
      if (low + mid < low)
        high++;
      low += mid;

      // Now add carry.
      if (low + carry < low)
        high++;
      low += carry;
    }

    if (add) {
      // And now DST[i], and store the new low part there.
      if (low + dst[i] < low)
        high++;
      dst[i] += low;
    } else
      dst[i] = low;

    carry = high;
  }

  if (srcParts < dstParts) {
    // Full multiplication, there is no overflow.
    if (srcParts + 1 != dstParts) {
      throw std::runtime_error("overflow");
    }
    dst[srcParts] = carry;
    return 0;
  }

  // We overflowed if there is carry.
  if (carry)
    return 1;

  // We would overflow if any significant unwritten parts would be
  // non-zero.  This is true if any remaining src parts are non-zero
  // and the multiplier is non-zero.
  if (multiplier)
    for (unsigned i = dstParts; i < srcParts; i++)
      if (src[i])
        return 1;

  // We fitted in the narrow destination.
  return 0;
}

int tcMultiply(uint64_t *dst, const uint64_t *lhs,
                      const uint64_t *rhs, unsigned parts) {
  if (dst == lhs || dst == rhs) {
    throw std::runtime_error("source & dest in same buffer");
  }

  int overflow = 0;

  for (unsigned i = 0; i < parts; i++) {
    // Don't accumulate on the first iteration so we don't need to initalize
    // dst to 0.
    overflow |=
        tcMultiplyPart(&dst[i], lhs, rhs[i], 0, parts, parts - i, i != 0);
  }

  return overflow;
}

void tcShiftLeft(uint64_t *Dst, unsigned Words, unsigned Count) {
  // Don't bother performing a no-op shift.
  if (!Count)
    return;

  // WordShift is the inter-part shift; BitShift is the intra-part shift.
  unsigned WordShift = std::min(Count / BITS_PER_WORD, Words);
  unsigned BitShift = Count % BITS_PER_WORD;

  // Fastpath for moving by whole words.
  if (BitShift == 0) {
    std::memmove(Dst + WordShift, Dst, (Words - WordShift) * WORD_SIZE);
  } else {
    while (Words-- > WordShift) {
      Dst[Words] = Dst[Words - WordShift] << BitShift;
      if (Words > WordShift)
        Dst[Words] |=
          Dst[Words - WordShift - 1] >> (BITS_PER_WORD - BitShift);
    }
  }

  // Fill in the remainder with 0s.
  std::memset(Dst, 0, WordShift * WORD_SIZE);
}

/// Return the high 32 bits of a 64 bit value.
constexpr uint32_t Hi_32(uint64_t Value) {
  return static_cast<uint32_t>(Value >> 32);
}

/// Return the low 32 bits of a 64 bit value.
constexpr uint32_t Lo_32(uint64_t Value) {
  return static_cast<uint32_t>(Value);
}

/// Make a 64-bit integer from a high / low pair of 32-bit integers.
constexpr uint64_t Make_64(uint32_t High, uint32_t Low) {
  return ((uint64_t)High << 32) | (uint64_t)Low;
}

/// Implementation of Knuth's Algorithm D (Division of nonnegative integers)
/// from "Art of Computer Programming, Volume 2", section 4.3.1, p. 272. The
/// variables here have the same names as in the algorithm. Comments explain
/// the algorithm and any deviation from it.
static void KnuthDiv(uint32_t *u, uint32_t *v, uint32_t *q, uint32_t* r,
                     unsigned m, unsigned n) {
  if (!u) {
    throw std::runtime_error("Must provide dividend");
  }
  if (!v) {
    throw std::runtime_error("Must provide divisor");
  }
  if (!q) {
    throw std::runtime_error("Must provide quotient");
  }
  if (u == v || u == q || v == q) {
    throw std::runtime_error("Must use different memory");
  }
  if (n <= 1) {
    throw std::runtime_error("n must be > 1");
  }

  // b denotes the base of the number system. In our case b is 2^32.
  const uint64_t b = uint64_t(1) << 32;

  // D1. [Normalize.] Set d = b / (v[n-1] + 1) and multiply all the digits of
  // u and v by d. Note that we have taken Knuth's advice here to use a power
  // of 2 value for d such that d * v[n-1] >= b/2 (b is the base). A power of
  // 2 allows us to shift instead of multiply and it is easy to determine the
  // shift amount from the leading zeros.  We are basically normalizing the u
  // and v so that its high bits are shifted to the top of v's range without
  // overflow. Note that this can require an extra word in u so that u must
  // be of length m+n+1.
  unsigned shift = count_zeros(v[n - 1]);
  uint32_t v_carry = 0;
  uint32_t u_carry = 0;
  if (shift) {
    for (unsigned i = 0; i < m+n; ++i) {
      uint32_t u_tmp = u[i] >> (32 - shift);
      u[i] = (u[i] << shift) | u_carry;
      u_carry = u_tmp;
    }
    for (unsigned i = 0; i < n; ++i) {
      uint32_t v_tmp = v[i] >> (32 - shift);
      v[i] = (v[i] << shift) | v_carry;
      v_carry = v_tmp;
    }
  }
  u[m+n] = u_carry;

  // D2. [Initialize j.]  Set j to m. This is the loop counter over the places.
  int j = m;
  do {
    // D3. [Calculate q'.].
    //     Set qp = (u[j+n]*b + u[j+n-1]) / v[n-1]. (qp=qprime=q')
    //     Set rp = (u[j+n]*b + u[j+n-1]) % v[n-1]. (rp=rprime=r')
    // Now test if qp == b or qp*v[n-2] > b*rp + u[j+n-2]; if so, decrease
    // qp by 1, increase rp by v[n-1], and repeat this test if rp < b. The test
    // on v[n-2] determines at high speed most of the cases in which the trial
    // value qp is one too large, and it eliminates all cases where qp is two
    // too large.
    uint64_t dividend = Make_64(u[j+n], u[j+n-1]);
    uint64_t qp = dividend / v[n-1];
    uint64_t rp = dividend % v[n-1];
    if (qp == b || qp*v[n-2] > b*rp + u[j+n-2]) {
      qp--;
      rp += v[n-1];
      if (rp < b && (qp == b || qp*v[n-2] > b*rp + u[j+n-2]))
        qp--;
    }

    // D4. [Multiply and subtract.] Replace (u[j+n]u[j+n-1]...u[j]) with
    // (u[j+n]u[j+n-1]..u[j]) - qp * (v[n-1]...v[1]v[0]). This computation
    // consists of a simple multiplication by a one-place number, combined with
    // a subtraction.
    // The digits (u[j+n]...u[j]) should be kept positive; if the result of
    // this step is actually negative, (u[j+n]...u[j]) should be left as the
    // true value plus b**(n+1), namely as the b's complement of
    // the true value, and a "borrow" to the left should be remembered.
    int64_t borrow = 0;
    for (unsigned i = 0; i < n; ++i) {
      uint64_t p = uint64_t(qp) * uint64_t(v[i]);
      int64_t subres = int64_t(u[j+i]) - borrow - Lo_32(p);
      u[j+i] = Lo_32(subres);
      borrow = Hi_32(p) - Hi_32(subres);
    }
    bool isNeg = u[j+n] < borrow;
    u[j+n] -= Lo_32(borrow);

    // D5. [Test remainder.] Set q[j] = qp. If the result of step D4 was
    // negative, go to step D6; otherwise go on to step D7.
    q[j] = Lo_32(qp);
    if (isNeg) {
      // D6. [Add back]. The probability that this step is necessary is very
      // small, on the order of only 2/b. Make sure that test data accounts for
      // this possibility. Decrease q[j] by 1
      q[j]--;
      // and add (0v[n-1]...v[1]v[0]) to (u[j+n]u[j+n-1]...u[j+1]u[j]).
      // A carry will occur to the left of u[j+n], and it should be ignored
      // since it cancels with the borrow that occurred in D4.
      bool carry = false;
      for (unsigned i = 0; i < n; i++) {
        uint32_t limit = std::min(u[j+i],v[i]);
        u[j+i] += v[i] + carry;
        carry = u[j+i] < limit || (carry && u[j+i] == limit);
      }
      u[j+n] += carry;
    }
    // D7. [Loop on j.]  Decrease j by one. Now if j >= 0, go back to D3.
  } while (--j >= 0);

  // D8. [Unnormalize]. Now q[...] is the desired quotient, and the desired
  // remainder may be obtained by dividing u[...] by d. If r is non-null we
  // compute the remainder (urem uses this).
  if (r) {
    // The value d is expressed by the "shift" value above since we avoided
    // multiplication by d by using a shift left. So, all we have to do is
    // shift right here.
    if (shift) {
      uint32_t carry = 0;
      for (int i = n-1; i >= 0; i--) {
        r[i] = (u[i] >> shift) | carry;
        carry = u[i] << (32 - shift);
      }
    } else {
      for (int i = n-1; i >= 0; i--) {
        r[i] = u[i];
      }
    }
  }
}

void divide(const uint64_t *LHS, unsigned lhsWords, const uint64_t *RHS,
                   unsigned rhsWords, uint64_t *Quotient, uint64_t *Remainder) {
  if (lhsWords < rhsWords) {
    throw std::runtime_error("Fractional result in divide");
  }

  // First, compose the values into an array of 32-bit words instead of
  // 64-bit words. This is a necessity of both the "short division" algorithm
  // and the Knuth "classical algorithm" which requires there to be native
  // operations for +, -, and * on an m bit value with an m*2 bit result. We
  // can't use 64-bit operands here because we don't have native results of
  // 128-bits. Furthermore, casting the 64-bit values to 32-bit values won't
  // work on large-endian machines.
  unsigned n = rhsWords * 2;
  unsigned m = (lhsWords * 2) - n;

  // Allocate space for the temporary values we need either on the stack, if
  // it will fit, or on the heap if it won't.
  uint32_t SPACE[128];
  uint32_t *U = nullptr;
  uint32_t *V = nullptr;
  uint32_t *Q = nullptr;
  uint32_t *R = nullptr;
  if ((Remainder?4:3)*n+2*m+1 <= 128) {
    U = &SPACE[0];
    V = &SPACE[m+n+1];
    Q = &SPACE[(m+n+1) + n];
    if (Remainder)
      R = &SPACE[(m+n+1) + n + (m+n)];
  } else {
    U = new uint32_t[m + n + 1];
    V = new uint32_t[n];
    Q = new uint32_t[m+n];
    if (Remainder)
      R = new uint32_t[n];
  }

  // Initialize the dividend
  std::memset(U, 0, (m+n+1)*sizeof(uint32_t));
  for (unsigned i = 0; i < lhsWords; ++i) {
    uint64_t tmp = LHS[i];
    U[i * 2] = Lo_32(tmp);
    U[i * 2 + 1] = Hi_32(tmp);
  }
  U[m+n] = 0; // this extra word is for "spill" in the Knuth algorithm.

  // Initialize the divisor
  std::memset(V, 0, (n)*sizeof(uint32_t));
  for (unsigned i = 0; i < rhsWords; ++i) {
    uint64_t tmp = RHS[i];
    V[i * 2] = Lo_32(tmp);
    V[i * 2 + 1] = Hi_32(tmp);
  }

  // initialize the quotient and remainder
  std::memset(Q, 0, (m+n) * sizeof(uint32_t));
  if (Remainder) {
    std::memset(R, 0, n * sizeof(uint32_t));
  }

  // Now, adjust m and n for the Knuth division. n is the number of words in
  // the divisor. m is the number of words by which the dividend exceeds the
  // divisor (i.e. m+n is the length of the dividend). These sizes must not
  // contain any zero words or the Knuth algorithm fails.
  for (unsigned i = n; i > 0 && V[i-1] == 0; i--) {
    n--;
    m++;
  }
  for (unsigned i = m+n; i > 0 && U[i-1] == 0; i--) {
    m--;
  }

  // If we're left with only a single word for the divisor, Knuth doesn't work
  // so we implement the short division algorithm here. This is much simpler
  // and faster because we are certain that we can divide a 64-bit quantity
  // by a 32-bit quantity at hardware speed and short division is simply a
  // series of such operations. This is just like doing short division but we
  // are using base 2^32 instead of base 10.
  if (n == 0) {
    throw std::runtime_error("Divide by zero?");
  }
  if (n == 1) {
    uint32_t divisor = V[0];
    uint32_t remainder = 0;
    for (int i = m; i >= 0; i--) {
      uint64_t partial_dividend = Make_64(remainder, U[i]);
      if (partial_dividend == 0) {
        Q[i] = 0;
        remainder = 0;
      } else if (partial_dividend < divisor) {
        Q[i] = 0;
        remainder = Lo_32(partial_dividend);
      } else if (partial_dividend == divisor) {
        Q[i] = 1;
        remainder = 0;
      } else {
        Q[i] = Lo_32(partial_dividend / divisor);
        remainder = Lo_32(partial_dividend - (Q[i] * divisor));
      }
    }
    if (R)
      R[0] = remainder;
  } else {
    // Now we're ready to invoke the Knuth classical divide algorithm. In this
    // case n > 1.
    KnuthDiv(U, V, Q, R, m, n);
  }

  // If the caller wants the quotient
  if (Quotient) {
    for (unsigned i = 0; i < lhsWords; ++i)
      Quotient[i] = Make_64(Q[i*2+1], Q[i*2]);
  }

  // If the caller wants the remainder
  if (Remainder) {
    for (unsigned i = 0; i < rhsWords; ++i)
      Remainder[i] = Make_64(R[i*2+1], R[i*2]);
  }

  // Clean up the memory we allocated.
  if (U != &SPACE[0]) {
    delete [] U;
    delete [] V;
    delete [] Q;
    delete [] R;
  }
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
    return res;
  }
  // The result is negative if one and only one of inputs is negative.
  if (isNegative() ^ RHS.isNegative()) {
    // Return signed minimum value
    BQInt ret(BitWidth, 0);
    uint64_t mask = 1ULL << (BitWidth % BITS_PER_WORD);
    if (isSingleWord()) {
      ret.U.word |= mask;
    } else {
      ret.U.ptr[BitWidth / BITS_PER_WORD] |= mask;
    }
    return ret;
  } else {
    BQInt ret(BitWidth, UINT64_MAX, true);
    unsigned clearWidth = BitWidth - 1;
    uint64_t mask = ~(1ULL << (clearWidth % BITS_PER_WORD));
    if (isSingleWord()) {
      ret.U.word &= mask;
    } else {
      ret.U.ptr[BitWidth / BITS_PER_WORD] &= mask;
    }
    return ret;
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

bool BQInt::isNegative() const {
  unsigned bitPosition = BitWidth - 1;
  unsigned whichBit = bitPosition % BITS_PER_WORD;
  uint64_t maskBit = 1ULL << whichBit;
  unsigned whichWord = bitPosition / BITS_PER_WORD;
  uint64_t getWord = isSingleWord() ? U.word : U.ptr[whichWord];
  return (maskBit & getWord) != 0;
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
  if (isSingleWord()) {
    return U.word == RHS.U.word;
  }
  return std::equal(U.ptr, U.ptr + getNumWords(), RHS.U.ptr);
}

} // namespace zirgen::BigInt::Bytecode
