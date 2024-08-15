// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#include "zirgen/circuit/rv32im/v1/edsl/compute.h"

#include "zirgen/circuit/rv32im/v1/edsl/top.h"

namespace zirgen::rv32im_v1 {

/// Peek at the value of a BigInt in memory given a register holding it's pointer.
std::vector<Val> peekBigInt(size_t regAddr) {
  Val addr = ramPeek(regAddr).flat() / kWordSize;

  std::vector<Val> bigint;
  bigint.reserve(BigInt::kByteWidth);
  for (size_t i = 0; i < BigInt::kWordWidth; i++) {
    U32Val word = ramPeek(addr + i);
    for (Val byt : word.bytes) {
      bigint.push_back(byt);
    }
  }
  return bigint;
}

template <typename T> void logBigInt(std::string label, std::vector<T>& bigint) {
  std::string fmt = label + ": (";
  for (size_t i = 0; i < bigint.size(); i++) {
    if (i < bigint.size() - 1) {
      fmt += "%02x ";
    } else {
      fmt += "%02x)";
    }
  }

  std::vector<Val> vals;
  vals.reserve(bigint.size());
  for (T limb : bigint) {
    vals.push_back(limb);
  }
  doExtern("log", fmt, 0, vals);
}

std::vector<Val> peekX() {
  auto x = peekBigInt(RegAddr::kA2);
  // logBigInt("peekX", x);
  return x;
}

std::vector<Val> peekY() {
  auto y = peekBigInt(RegAddr::kA3);
  // logBigInt("peekY", y);
  return y;
}

std::vector<Val> peekN() {
  auto n = peekBigInt(RegAddr::kA4);
  // logBigInt("peekN", n);
  return n;
}

// Nondet calculates the denormalized multiplication result of two bigints.
// The result will not have any carries propagated, and so limbs will be out of their input range.
std::vector<Val> denormMultiply(std::vector<Val>& a, std::vector<Val>& b) {
  // Compute the result via schoolbook multiplication.
  std::vector<Val> c;
  c.reserve(a.size() + b.size() - 1);
  for (int64_t i = 0; i < int64_t(a.size() + b.size() - 1); i++) {
    Val limb = 0;
    for (int64_t j = std::max<int64_t>(0, i - b.size() + 1);
         j <= std::min<int64_t>(a.size() - 1, i);
         j++) {
      limb = limb + a.at(j) * b.at(i - j);
    }
    c.push_back(limb);
  }
  // logBigInt("denormMultiply", c);
  return c;
}

// Nondet calculates the denormalized addition result of two bigints.
// The result will not have any carries propagated, and so limbs will be out of their input range.
std::vector<Val> denormAdd(std::vector<Val>& a, std::vector<Val>& b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("bigint denormAdd does not handle unequal size inputs");
  }
  std::vector<Val> c;
  c.reserve(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    c.push_back(a.at(i) + b.at(i));
  }
  // logBigInt("denormAdd", c);
  return c;
}

// Nondet calculates the denormalized subtraction result of two bigints.
// Result limbs will be out of their input range and may be negative.
std::vector<Val> denormSub(std::vector<Val>& a, std::vector<Val>& b) {
  if (a.size() != b.size()) {
    throw std::runtime_error("bigint denormSub does not handle unequal size inputs");
  }
  std::vector<Val> c;
  c.reserve(a.size());
  for (size_t i = 0; i < a.size(); i++) {
    c.push_back(a.at(i) - b.at(i));
  }
  // logBigInt("denormSub", c);
  return c;
}

// Nondet propagates carries to calculate the normalized representation.
// This is true of multiplications of byte-limbed BigInts of up to 128 limbs.
//
// Returns a pair. The first element being the normalized value and the second is the carries.
// carries[i] is the carry-out for the ith limb and has size denorm.size().
// normalized has size denorm.size() + 1, with the last limb being equal to the final carry.
std::pair<std::vector<Val>, std::vector<Val>> normalize(std::vector<Val>& denorm) {
  std::vector<Val> normalized;
  normalized.reserve(denorm.size() + 1);
  std::vector<Val> carries;
  carries.reserve(denorm.size());

  Val carry = 0;
  for (size_t i = 0; i < denorm.size(); i++) {
    // Each limb plus previous carry is assumed to be in the range [-2^23, 2^23).
    // Each carry is, by corollary, in the range [-2^15, 2^15).
    // This is true for byte-limb bigints with up to 128 limbs.
    Val tmp = denorm.at(i) + carry + (1 << 23);
    Val limb = tmp & 0xFF;
    carry = ((tmp - limb) / 0x100) - (1 << 15);
    normalized.push_back(limb);
    carries.push_back(carry);
  }
  normalized.push_back(carry);
  // logBigInt("normalize.normalized", normalized);
  // logBigInt("normalize.carries   ", carries);
  return std::make_pair(normalized, carries);
}

// Nondet calculates the multiplication result given the word addresses of each operand.
// NOTE: If the intermediate result of denorm multiplication is used, it is more eddicient to
// calculate that value first and then call normalize. This operation wraps the two.
std::vector<Val> multiply(std::vector<Val>& a, std::vector<Val>& b) {
  std::vector<Val> denorm = denormMultiply(a, b);
  return normalize(denorm).first;
}

// Calculates the quotient and remainder for a/b.
// a must be a normalized multiplication result and b a normalized bigint.
std::vector<Val> quotient(std::vector<Val>& a, std::vector<Val>& b) {
  if ((a.size() != BigInt::kByteWidth * 2) || (b.size() != BigInt::kByteWidth)) {
    throw std::runtime_error("bigint quotient: invalid input value lengths");
  }
  // Call bigintQuotient extern with a and b.
  std::vector<Val> input;
  input.reserve(a.size() + b.size());
  input.insert(input.end(), a.begin(), a.end());
  input.insert(input.end(), b.begin(), b.end());
  return doExtern("bigintQuotient", "", BigInt::kByteWidth, input);
}

BigIntCycleImpl::BigIntCycleImpl(RamHeader ramHeader) : ram(ramHeader, 4), mix("mix") {
  bytes.resize(BigInt::kBytesSize);
  mulBuffer.resize(BigInt::kMulBufferSize);
  carryHi.resize(BigInt::kCarryHiSize);

  this->registerCallback("_builtin_verify", &BigIntCycleImpl::onVerify);
}

void BigIntCycleImpl::set(Top top) {
  // Pre-calculate all the NONDET values used below.

  // Get some basic state data
  Val cycle = top->code->cycle;
  BodyStep body = top->mux->at<StepType::BODY>();
  Val curPC = BACK(1, body->pc->get());

  // Set control registers.
  eqz(BACK(1, body->nextMajor->get()) - MajorType::kBigInt);
  Val isFirstCycle = BACK(1, body->majorSelect->at(MajorType::kECall));
  IF(isFirstCycle) {
    stageOffset->set(0);
    stage->set(0);

    // Constrain the op register to 0 (modular multiply).
    // NOTE: In future revisions of this circuit, other ops will be supported.
    ECallCycle ecall = body->majorMux->at<MajorType::kECall>();
    ECallBigInt ecallBigInt = ecall->minorMux->at<ECallType::kBigInt>();
    Val op = BACK(1, ecallBigInt->readA1->data().flat());
    eqz(op);
  }
  IF(1 - isFirstCycle) {
    stageOffset->set((1 - BACK(1, stage->at(0))) - BACK(1, stageOffset->get()));
    stage->set(BACK(1, stage->get()) + (1 - stageOffset));
  }
  mulActive->set((stage->at(2) + stage->at(4)) * stageOffset);
  finalize->set(stage->at(4) * stageOffset);
  XLOG("BIGINT: stage = %u, stageOffset = %u, mulActive = %u, finalize = %u",
       stage,
       stageOffset,
       mulActive,
       finalize);

  // In stage zero, load the four io addresses for z, x, y, and N.
  IF(stage->at(0)) {
    io.at(0)->doRead(cycle, RegAddr::kA0);
    io.at(1)->doRead(cycle, RegAddr::kA2);
    io.at(2)->doRead(cycle, RegAddr::kA3);
    io.at(3)->doRead(cycle, RegAddr::kA4);
  }

  // Fetch the read or write address.
  IF(1 - stageOffset) {
    // Depending on the stage, fetch the I/O address from the ecall cycle.
    for (size_t i = 1; i < BigInt::kStages; i++) {
      RamReg* readAX;
      switch (i) {
      case 1:
        // Access the address of N, in a4
        readAX = &io.at(3);
        break;
      case 2:
        // Access the address of x, in a2
        readAX = &io.at(1);
        break;
      case 3:
        // Access the address of y, in a3
        readAX = &io.at(2);
        break;
      case 4:
        // Access the address of z, in a0
        readAX = &io.at(0);
        break;
      }
      IF(stage->at(i)) {
        // NOTE: Dividing by word size here will result in a valid word address, in the range
        // [0, 2^26), only if the input is word-aligned (i.e. is a multiple of kWordSize) and in the
        // address-space range [0, 2^28).
        ioAddr->set(BACK(BigInt::kCyclesPerStage * i - 1, (*readAX)->data().flat()) / kWordSize);
      }
    }
  }
  IF(stageOffset) { ioAddr->set(BACK(1, ioAddr->get())); }

  // In stages 1-3, read an 4 words of input from memory.
  // Related to step 4 in the approach description.
  IF(stage->at(1) + stage->at(2) + stage->at(3)) {
    for (size_t i = 0; i < BigInt::kIoSize; i++) {
      io.at(i)->doRead(cycle, ioAddr + stageOffset * BigInt::kIoSize + i);
    }
    XLOG("  Reading: ioAddr = 0x%x, data = { %u, %u, %u, %u }",
         ioAddr * kWordSize,
         io.at(0)->data().flat(),
         io.at(1)->data().flat(),
         io.at(2)->data().flat(),
         io.at(3)->data().flat());
  }

  // Materialize NONDET calculated values into the relevant registers.
  // NOTE: This is done in one large NONDET block so that calculated values can be shared.
  NONDET {
    // Calculate all the intermediate results values we will need to populate registers below.
    auto x = peekX();
    auto y = peekY();
    auto n = peekN();

    auto denormXY = denormMultiply(x, y);
    auto xy = normalize(denormXY).first;
    auto q = quotient(xy, n);
    auto r = denormMultiply(q, n);
    auto denormZ = denormSub(denormXY, r);
    auto zc = normalize(denormZ);
    auto z = zc.first;
    auto c = zc.second;

    r.emplace_back(0);
    denormZ.emplace_back(0);

    // Materialize the quotient, carry, or output values to bytes.
    // Related to steps 2 and 8 in the approach description.
    for (size_t i = 0; i < BigInt::kBytesSize; i++) {
      IF(stage->at(0)) {
        // Bytes in stage 0 are unused, but must be set.
        bytes.at(i)->set(0);
      }
      IF(stage->at(1)) {
        IF(1 - stageOffset) { bytes.at(i)->set(q.at(i)); }
        IF(stageOffset) { bytes.at(i)->set(q.at(i + BigInt::kBytesSize)); }
      }
      // Stages 2 and 3 constrain the low carries to range [-2^15, 2^15).
      // Does so by splitting the value plus 2^15 into two bytes at adjacent indices.
      // NOTE: This code is very repetative and could probably be condensed.
      IF(stage->at(2)) {
        IF(1 - stageOffset) {
          Val ci = c.at(i / 2);
          if (i % 2 == 0) {
            bytes.at(i)->set((ci + (1 << 15)) & 0xFF);
          } else {
            bytes.at(i)->set(((ci + (1 << 15)) & 0xFF00) / 0x100);
          }
        }
        IF(stageOffset) {
          Val ci = c.at((i + BigInt::kBytesSize) / 2);
          if (i % 2 == 0) {
            bytes.at(i)->set((ci + (1 << 15)) & 0xFF);
          } else {
            bytes.at(i)->set(((ci + (1 << 15)) & 0xFF00) / 0x100);
          }
        }
      }
      IF(stage->at(3)) {
        IF(1 - stageOffset) {
          Val ci = c.at((i + BigInt::kBytesSize * 2) / 2);
          if (i % 2 == 0) {
            bytes.at(i)->set((ci + (1 << 15)) & 0xFF);
          } else {
            bytes.at(i)->set(((ci + (1 << 15)) & 0xFF00) / 0x100);
          }
        }
        IF(stageOffset) {
          Val ci = c.at((i + BigInt::kBytesSize * 3) / 2);
          if (i % 2 == 0) {
            bytes.at(i)->set((ci + (1 << 15)) & 0xFF);
          } else {
            bytes.at(i)->set(((ci + (1 << 15)) & 0xFF00) / 0x100);
          }
        }
      }
      IF(stage->at(4)) {
        IF(1 - stageOffset) { bytes.at(i)->set(z.at(i)); }
        IF(stageOffset) { bytes.at(i)->set(z.at(i + BigInt::kBytesSize)); }
      }
    }
    // logBigInt("bytes", bytes);

    // Materialize the high carry values into (unconstrained) registers.
    // Related to step 7 in the approach description.
    for (size_t i = 0; i < BigInt::kCarryHiSize; i++) {
      // Guess c and pad two to make it a multiple of the carryHi size.
      c.emplace_back(0);
      c.emplace_back(0);

      IF(stage->at(2)) {
        IF(1 - stageOffset) { carryHi.at(i)->set(c.at(i + BigInt::kByteWidth)); }
        IF(stageOffset) { carryHi.at(i)->set(c.at(i + BigInt::kByteWidth + BigInt::kCarryHiSize)); }
      }
      IF(stage->at(3)) {
        IF(1 - stageOffset) {
          carryHi.at(i)->set(c.at(i + BigInt::kByteWidth + 2 * BigInt::kCarryHiSize));
        }
        IF(stageOffset) {
          carryHi.at(i)->set(c.at(i + BigInt::kByteWidth + 3 * BigInt::kCarryHiSize));
        }
      }
    }

    // At stages 2 and 4, materialize the output values into the multiplier.
    for (size_t i = 0; i < BigInt::kMulBufferSize; i++) {
      // At stage 2 copy the denomalized reduction value r into the mulBuffer.
      IF(stage->at(2)) {
        IF(1 - stageOffset) { mulBuffer.at(i)->set(r.at(i)); }
        IF(stageOffset) { mulBuffer.at(i)->set(r.at(i + BigInt::kMulBufferSize)); }
      }
      IF(stage->at(4)) {
        IF(1 - stageOffset) { mulBuffer.at(i)->set(denormZ.at(i)); }
        IF(stageOffset) { mulBuffer.at(i)->set(denormZ.at(i + BigInt::kMulBufferSize)); }
      }
    }
  }

  // At stages 1 and 3, copy the inputs into the multiplier.
  for (size_t i = 0; i < BigInt::kMulInSize; i++) {
    // At stage 1, copy q from bytes to the first half of the mulBuffer.
    IF(stage->at(1)) { mulBuffer.at(i)->set(bytes.at(i)); }
    // At stage 3, copy x from io to the first half of the mulBuffer.
    IF(stage->at(3)) {
      mulBuffer.at(i)->set(BACK(2, io.at(i / kWordSize)->data().bytes.at(i % kWordSize)));
    }
    // Copy the second input, N in stage 1 and y in stage 3, from io into the mulBuffer.
    IF(stage->at(1) + stage->at(3)) {
      mulBuffer.at(i + BigInt::kMulInSize)
          ->set(io.at(i / kWordSize)->data().bytes.at(i % kWordSize));
    }
  }

  /*
  for (size_t i = 0; i < BigInt::kMulBufferSize; i += 8) {
    logBigInt("  mulBuffer", mulBuffer);
  }
  */

  // Check that the multiplication result combined with carries equals the output value.
  // Implements step 9, the carry checks.
  IF(finalize) {
    // Check the lower half of the carries.
    Val cIn(0);
    for (size_t i = 0; i < 2 * BigInt::kByteWidth - 1; i++) {
      Val cOut;
      Val zi;
      if (i < BigInt::kByteWidth) {
        size_t iBackCarry = ((BigInt::kByteWidth * 2) / BigInt::kBytesSize) -
                            ((i * 2) / BigInt::kBytesSize) + BigInt::kCyclesPerStage - 1;
        size_t iOffsetCarry = (i * 2) % BigInt::kBytesSize;
        // Reassemble the carry value in the range [-2^15, 2^15) from the two bytes values.
        cOut = (BACK(iBackCarry, bytes.at(iOffsetCarry)->get()) +
                BACK(iBackCarry, bytes.at(iOffsetCarry + 1)->get()) * (1 << 8)) -
               (1 << 15);

        size_t iBackZ = (BigInt::kByteWidth / BigInt::kBytesSize) - (i / BigInt::kBytesSize) - 1;
        size_t iOffsetZ = i % BigInt::kBytesSize;
        zi = BACK(iBackZ, bytes.at(iOffsetZ)->get());
      } else if (i < 2 * BigInt::kByteWidth - 2) {
        size_t iBackCarry = ((BigInt::kByteWidth * 2) / BigInt::kCarryHiSize) -
                            (i / BigInt::kCarryHiSize) + BigInt::kCyclesPerStage - 1;
        size_t iOffsetCarry = i % BigInt::kCarryHiSize;
        cOut = BACK(iBackCarry, carryHi.at(iOffsetCarry)->get());
        zi = 0;
      } else {
        cOut = 0;
        zi = 0;
      }

      size_t iBackZD =
          ((BigInt::kByteWidth * 2) / BigInt::kMulBufferSize) - (i / BigInt::kMulBufferSize) - 1;
      size_t iOffsetZD = i % BigInt::kMulBufferSize;
      Val ziDenorm = BACK(iBackZD, mulBuffer.at(iOffsetZD)->get());
      // XLOG("  i: %u, cIn: %x, cOut: %x, zi: %x, ziDenorm: %x", i, cIn, cOut, zi, ziDenorm);

      eq(ziDenorm + cIn, cOut * (1 << 8) + zi);
      cIn = cOut;
    }
  }

  // In stage 4, write the output to memory.
  IF(stage->at(4)) {
    for (size_t i = 0; i < BigInt::kIoSize; i++) {
      size_t byteOffset = kWordSize * i;
      io.at(i)->doWrite(cycle,
                        ioAddr + stageOffset * BigInt::kIoSize + i,
                        U32Val(bytes.at(byteOffset),
                               bytes.at(byteOffset + 1),
                               bytes.at(byteOffset + 2),
                               bytes.at(byteOffset + 3)));
    }
    XLOG("  Writing: ioAddr = 0x%x, data = { %u, %u, %u, %u }",
         ioAddr * kWordSize,
         io.at(0)->data().flat(),
         io.at(1)->data().flat(),
         io.at(2)->data().flat(),
         io.at(3)->data().flat());
  }

  // Set the next major cycle type and program counter.
  IF(1 - finalize) {
    body->pc->set(curPC);
    body->nextMajor->set(MajorType::kBigInt);
  }
  IF(finalize) {
    body->pc->set(curPC + 4);
    body->nextMajor->set(MajorType::kMuxSize);
  }
}

void BigIntCycleImpl::onVerify() {
  // When the multiplier contraint is active, use a randomized polynomial equality check to verify
  // the correctness of the multiplication relation for the results calculated in NONDET.
  // This method is inspired by xJSnark section IV.B with the addition of randomization.
  // https://akosba.github.io/papers/xjsnark.pdf
  // Implements steps 3 and 6, the multiplication checks.
  IF(mulActive) {
    FpExt accumX(Val(0));
    FpExt accumY(Val(0));

    for (size_t i = 0; i < BigInt::kByteWidth; i++) {
      size_t iBackIn = (BigInt::kByteWidth / BigInt::kMulInSize) - (i / BigInt::kMulInSize) +
                       BigInt::kCyclesPerStage - 1;
      size_t iOffsetIn = i % BigInt::kMulInSize;
      Val xi = BACK(iBackIn, mulBuffer.at(iOffsetIn)->get());
      accumX = accumX * mix + FpExt(xi);
      Val yi = BACK(iBackIn, mulBuffer.at(iOffsetIn + BigInt::kMulInSize)->get());
      accumY = accumY * mix + FpExt(yi);
      // XLOG("  i: %u, mix: (%u, %u, %u, %u), iBackIn: %u, iOffsetIn: %u, xi: %x, yi: %u", i,
      // mix->elem(0), mix->elem(1), mix->elem(2), mix->elem(3), iBackIn, iOffsetIn, xi, yi);
    }

    FpExt accumZ(Val(0));
    for (size_t i = 0; i < 2 * BigInt::kByteWidth - 1; i++) {
      size_t iBackOut =
          ((BigInt::kByteWidth * 2) / BigInt::kMulOutSize) - (i / BigInt::kMulOutSize) - 1;
      size_t iOffsetOut = i % BigInt::kMulOutSize;
      // In the second multiplication check, constrain the value of denormZ + r.
      Val ri =
          UNCHECKED_BACK(iBackOut + BigInt::kCyclesPerStage * 2, mulBuffer.at(iOffsetOut)->get()) *
          finalize;
      Val zi = BACK(iBackOut, mulBuffer.at(iOffsetOut)->get());
      accumZ = accumZ * mix + FpExt(zi + ri);
      // XLOG("  i: %u, mix: (%u, %u, %u, %u), iBackOut: %u, iOffsetOut: %u, zi: %x", i,
      // mix->elem(0), mix->elem(1), mix->elem(2), mix->elem(3), iBackOut, iOffsetOut, zi);
    }
    eq(accumX * accumY, accumZ);
  }
}

} // namespace zirgen::rv32im_v1
