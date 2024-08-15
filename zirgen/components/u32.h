// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/components/bytes.h"
#include "zirgen/components/iszero.h"

namespace zirgen {

// A U32 value.  May be 'denormalized' (i.e. have unpropagated carry)
struct U32Val {
  U32Val() = default;

  // Returns a value which is 0 when normalized that will protect from
  // underflow when subtracting one value
  static U32Val underflowProtect();

  // Make a U32 by hand, should generally be used only by tests of other U32 ops
  U32Val(Val b0, Val b1, Val b2, Val b3) : bytes({b0, b1, b2, b3}) {}

  Val flat() {
    return bytes[0] +             //
           (1 << 8) * bytes[1] +  //
           (1 << 16) * bytes[2] + //
           (1 << 24) * bytes[3];
  }

  static U32Val fromFlat(Val flat);

  std::array<Val, kWordSize> bytes;
};

template <> struct LogPrep<U32Val> {
  static void toLogVec(std::vector<Val>& out, U32Val x) {
    for (size_t i = 0; i < kWordSize; i++) {
      out.push_back(x.bytes[i]);
    }
  }
};

void eq(U32Val a, U32Val b, SourceLoc loc = SourceLoc::current());

class U32RegImpl : public CompImpl<U32RegImpl> {
public:
  static constexpr size_t rawSize() { return kWordSize; }

  U32RegImpl(llvm::StringRef source = "data");
  void setZero();
  void set(U32Val in);
  void setWithFactor(U32Val in, Val factor);
  U32Val get();
  Val getSmallSigned();
  Val getSmallUnsigned();
  std::vector<Val> toVals();
  void setFromVals(std::vector<Val> vals);

private:
  std::vector<Reg> bytes;
};
using U32Reg = Comp<U32RegImpl>;

template <> struct LogPrep<U32Reg> {
  static void toLogVec(std::vector<Val>& out, U32Reg x) {
    LogPrep<U32Val>::toLogVec(out, x->get());
  }
};

// Extracts the top bit from a U32 (and optionally returns the low 31)
// Does *not* normalize, and requires input to be normalized
class TopBitImpl : public CompImpl<TopBitImpl> {
public:
  void set(U32Val in);
  Val getHighBit();
  U32Val getLow31();

private:
  Bit topBit;
  ByteReg lowBits2;
  U32Val low31;
};
using TopBit = Comp<TopBitImpl>;

class IsZeroU32Impl : public CompImpl<IsZeroU32Impl> {
public:
  void set(U32Val in);
  Val isZero();

public:
  IsZero low;
  IsZero high;
};

using IsZeroU32 = Comp<IsZeroU32Impl>;

U32Val operator+(U32Val a, U32Val b);
U32Val operator-(U32Val a, U32Val b);
U32Val operator*(Val scalar, U32Val a);
U32Val operator&(U32Val a, U32Val b);

// Normalized a U32 by decomposing it into bytes + carrys
class U32NormalizeImpl : public CompImpl<U32NormalizeImpl> {
public:
  // Set from denormalized input
  void set(U32Val in);
  // Get normalized output
  U32Val getNormed();
  // Get carry
  Val getCarry();

private:
  std::array<ByteReg, kWordSize> normed;
  std::array<Twit, 2> carry;
};
using U32Normalize = Comp<U32NormalizeImpl>;

// Given a value p = [0, 31), produce a U32 holding 1 << p
// Implies range verification of p
class U32Po2Impl : public CompImpl<U32Po2Impl> {
public:
  U32Po2Impl();
  // Verify relation between bits + out
  void onVerify();
  // Set the 5 bit valus
  void set(Val in);
  // Get the 5 bit value
  Val get();
  // The the pos
  U32Val getPo2();

private:
  std::array<Bit, 5> bits;
  U32Reg out;
};
using U32Po2 = Comp<U32Po2Impl>;

// Multiply 2 (possibly signed) 32 bit values, and produce high/low parts of 64 bit product
class U32MulImpl : public CompImpl<U32MulImpl> {
public:
  void set(U32Val inA, U32Val inB, Val signedA, Val signedB);
  U32Val getHigh();
  U32Val getLow();

private:
  TopBit topA;
  TopBit topB;
  Reg subA;
  Reg subB;
  std::array<ByteReg, 11> outRegs;
  std::array<Twit, 4> carry;
};
using U32Mul = Comp<U32MulImpl>;

// Multiply 2 unsigned 32 bit values, add another, verify no overflow and return U32 result
class U32MulAccImpl : public CompImpl<U32MulAccImpl> {
public:
  // Out = A*B + C
  void set(U32Val inA, U32Val inB, U32Val inC);
  U32Val getOut();

private:
  std::array<ByteReg, kWordSize> outRegs;
  ByteReg carryByte;
  Twit carryTwit;
};
using U32MulAcc = Comp<U32MulAccImpl>;

} // namespace zirgen
