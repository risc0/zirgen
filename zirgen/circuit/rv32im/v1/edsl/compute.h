// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#pragma once

#include "zirgen/circuit/rv32im/v1/edsl/decode.h"
#include "zirgen/circuit/rv32im/v1/platform/constants.h"
#include "zirgen/components/ram.h"

namespace zirgen::rv32im_v1 {

namespace CompEnum {

enum InA { RS1 = 0, PC = 1 };
enum InB { RS2 = 0, IMM = 1 };
enum AluOp { ADD, SUB, AND, OR, XOR, INB };
enum NextMajor { DEC = MajorType::kMuxSize, VAND = MajorType::kVerifyAnd };

}; // namespace CompEnum

// Control bits for the 'compute' cycles
struct ComputeControlImpl : public CompImpl<ComputeControlImpl> {
  // Set the control data
  void set(U32Val imm,
           CompEnum::InA aluA,
           CompEnum::InB aluB,
           CompEnum::AluOp aluOp,
           CompEnum::NextMajor nextMajor);
  // The immediate value
  U32Reg imm;
  // Controls input B to ALU, 0 = RS1, 1 = PC
  Reg aluA;
  // Controls input B to ALU, 0 = RS2, 1 = IMM
  Reg aluB;
  // ALU output = mA * A + mB * B + mC * (a & b)
  Reg mA;
  Reg mB;
  Reg mC;
  // Next major op type (or decode)
  Reg nextMajor;
};
using ComputeControl = Comp<ComputeControlImpl>;

// ALU unit, preforms +, -, &, |, ^ based on controls
class ALUImpl : public CompImpl<ALUImpl> {
public:
  void set(U32Val inA, U32Val inB, ComputeControl control);
  U32Val getInB();
  U32Val getAndVal();
  U32Val getResult();
  // The next calls presume op = SUB, (i.e. mA=1, mB=-1, mC=0)
  Val getEQ();  // Returns 1 if A == B, 0 otherwise
  Val getLT();  // Returns 1 if A < B, as singed integers
  Val getLTU(); // Returns 1 if A < B, as unsigned integers

private:
  TopBit aTop;         // Top bit of input a
  TopBit bTop;         // Top bit of input b
  U32Reg regInB;       // Input b
  U32Reg andVal;       // a & b (verified later if used)
  U32Normalize result; // ALU output
  TopBit rTop;         // Top bit of result
  Reg overflow;        // Overflow bit, set on signed overflow
  Reg lt;              // true if a < b as signed ints (and Op = SUB)
  IsZeroU32 isZero;    // True if result is zero
};
using ALU = Comp<ALUImpl>;

class TopImpl;
using Top = Comp<TopImpl>;

class VerifyAndCycleImpl;

class ComputeCycleImpl : public CompImpl<ComputeCycleImpl> {
  friend class VerifyAndCycleImpl;

public:
  ComputeCycleImpl(size_t major, RamHeader ramHeader);
  void set(Top top);

private:
  size_t major;
  RamBody ram;
  RamReg readInst;
  Decoder decoder;
  OneHot<kMinorMuxSize> minorSelect;
  ComputeControl control;
  RamReg readRS1;
  RamReg readRS2;
  ALU alu;
  IsZero rdZero;
  RamReg writeRD;
};
using ComputeCycle = Comp<ComputeCycleImpl>;

template <size_t major> struct ComputeWrapImpl : public CompImpl<ComputeWrapImpl<major>> {
  ComputeWrapImpl(RamHeader ramHeader) : inner(major, ramHeader) {}
  void set(Top top) { inner->set(top); }
  ComputeCycle inner;
};
template <size_t major> using ComputeWrap = Comp<ComputeWrapImpl<major>>;

class VerifyAndCycleImpl : public CompImpl<VerifyAndCycleImpl> {
public:
  VerifyAndCycleImpl(RamHeader ramHeader);
  void set(Top top);

private:
  RamPass ram;
  std::array<Bit, 32> aBits;
  std::array<Bit, 32> bBits;
};
using VerifyAndCycle = Comp<VerifyAndCycleImpl>;

} // namespace zirgen::rv32im_v1
