// Copyright 2022 Risc0, Inc.
// All rights reserved.

#ifndef OPC
#define OPC(...) /**/
#endif

#ifndef OPI
#define OPI(...) /**/
#endif

#ifndef OPM
#define OPM(...) /**/
#endif

#ifndef OPD
#define OPD(...) /**/
#endif

#define X -1

// OPC(  // Define a bit op
//   id,        // unique numeric value
//   mnemonic,  // The assembly mnemonic from the spec
//   opcode,    // The top 5 bits of the opcode (lowest 2 bits are always 11)
//   func3,     // The value of func3 required, X == don't care
//   func7,     // The value of func7 required, X == don't care
//   immFmt,    // The format for the immediate value
//   aluA,      // Controls input A to ALU, RS1 or PC
//   aluB,      // Controls input B to ALU, RS2 or IMM
//   aluOp,     // ALU operation, ADD, SUB, AND, OR, XOR
//   setPC,     // What to write to PC
//   setRD,     // What to write to rd regsister
//   rdEn,      // Enable write to rd
//   next)      // Next major cycle type (or decode)
//
// clang-format off
OPC(  0, ADD,   0b01100, 0,  0, R, RS1, RS2, ADD, PC4,  RES, 1, DEC)
OPC(  1, SUB,   0b01100, 0, 32, R, RS1, RS2, SUB, PC4,  RES, 1, DEC)
OPC(  2, XOR,   0b01100, 4,  0, R, RS1, RS2, XOR, PC4,  RES, 1, VAND)
OPC(  3, OR,    0b01100, 6,  0, R, RS1, RS2, OR,  PC4,  RES, 1, VAND)
OPC(  4, AND,   0b01100, 7,  0, R, RS1, RS2, AND, PC4,  RES, 1, VAND)
OPC(  5, SLT,   0b01100, 2,  0, R, RS1, RS2, SUB, PC4,  LT,  1, DEC)
OPC(  6, SLTU,  0b01100, 3,  0, R, RS1, RS2, SUB, PC4,  LTU, 1, DEC)
OPC(  7, ADDI,  0b00100, 0,  X, I, RS1, IMM, ADD, PC4,  RES, 1, DEC)
OPC(  8, XORI,  0b00100, 4,  X, I, RS1, IMM, XOR, PC4,  RES, 1, VAND)
OPC(  9, ORI,   0b00100, 6,  X, I, RS1, IMM, OR,  PC4,  RES, 1, VAND)
OPC( 10, ANDI,  0b00100, 7,  X, I, RS1, IMM, AND, PC4,  RES, 1, VAND)
OPC( 11, SLTI,  0b00100, 2,  X, I, RS1, IMM, SUB, PC4,  LT,  1, DEC)
OPC( 12, SLTIU, 0b00100, 3,  X, I, RS1, IMM, SUB, PC4,  LTU, 1, DEC)
OPC( 13, BEQ,   0b11000, 0,  X, B, RS1, RS2, SUB, BEQ,  RES, 0, DEC)
OPC( 14, BNE,   0b11000, 1,  X, B, RS1, RS2, SUB, BNE,  RES, 0, DEC)
OPC( 15, BLT,   0b11000, 4,  X, B, RS1, RS2, SUB, BLT,  RES, 0, DEC)
OPC( 16, BGE,   0b11000, 5,  X, B, RS1, RS2, SUB, BGE,  RES, 0, DEC)
OPC( 17, BLTU,  0b11000, 6,  X, B, RS1, RS2, SUB, BLTU, RES, 0, DEC)
OPC( 18, BGEU,  0b11000, 7,  X, B, RS1, RS2, SUB, BGEU, RES, 0, DEC)
OPC( 19, JAL,   0b11011, X,  X, J, RS1, IMM, ADD, PCIM, XPC, 1, DEC)
OPC( 20, JALR,  0b11001, 0,  X, I, RS1, IMM, ADD, JMP,  XPC, 1, DEC)
OPC( 21, LUI,   0b01101, X,  X, U, RS1, IMM, INB, PC4,  RES, 1, DEC)
OPC( 22, AUIPC, 0b00101, X,  X, U, PC,  IMM, ADD, PC4,  RES, 1, DEC)
// clang-format on

// OPI(  // Define a memory op
//   id,        // unique numeric value
//   mnemonic,  // The assembly mnemonic from the spec
//   opcode,    // The top 5 bits of the opcode (lowest 2 bits are always 11)
//   func3,     // The value of func3 required, X == don't care
//   func7,     // The value of func7 required, X == don't care
//   immFmt,    // The format for the immediate value
//   isRead,    // Is this a read op
//   is8Bit,    // Is this a 8 bit op
//   is16Bit,   // Is this a 16 bit op
//   signExt)   // Should I do a sign extend
//
// clang-format off
OPI( 24, LB,    0b00000, 0,  X, I, 1, 1, 0, 1)
OPI( 25, LH,    0b00000, 1,  X, I, 1, 0, 1, 1)
OPI( 26, LW,    0b00000, 2,  X, I, 1, 0, 0, 0)
OPI( 27, LBU,   0b00000, 4,  X, I, 1, 1, 0, 0)
OPI( 28, LHU,   0b00000, 5,  X, I, 1, 0, 1, 0)
OPI( 29, SB,    0b01000, 0,  X, S, 0, 1, 0, 0)
OPI( 30, SH,    0b01000, 1,  X, S, 0, 0, 1, 0)
OPI( 31, SW,    0b01000, 2,  X, S, 0, 0, 0, 0)
// clang-format on

// OPM(  // Define a multiply op
//   id,        // unique numeric value
//   mnemonic,  // The assembly mnemonic from the spec
//   opcode,    // The top 5 bits of the opcode (lowest 2 bits are always 11)
//   func3,     // The value of func3 required, X == don't care
//   func7,     // The value of func7 required, X == don't care
//   immFmt,    // The format for the immediate value
//   useImm,    // What to use for the second op of the multiply
//   usePo2,    // Use po2 of low 5 of v2
//   signedA,   // Is r1 signed
//   signedB,   // Is v2 signed
//   high)      // Set result to high part (otherwise low)
// We start at a multiple of 8 to allow skipping the check on upper ID
//
// clang-format off
OPM( 32, MUL,   0b01100, 0,  1, R, 0, 0, 0, 0, 0)
OPM( 33, MULH,  0b01100, 1,  1, R, 0, 0, 1, 1, 1)
OPM( 34, MULSU, 0b01100, 2,  1, R, 0, 0, 1, 0, 1)
OPM( 35, MULU,  0b01100, 3,  1, R, 0, 0, 0, 0, 1)
OPM( 36, SLL,   0b01100, 1,  0, R, 0, 1, 0, 0, 0)
OPM( 37, SLLI,  0b00100, 1,  0, I, 1, 1, 0, 0, 0)
// clang-format on

// OPD(  // Define a divide op
//   id,        // unique numeric value
//   mnemonic,  // The assembly mnemonic from the spec
//   opcode,    // The top 5 bits of the opcode (lowest 2 bits are always 11)
//   func3,     // The value of func3 required, X == don't care
//   func7,     // The value of func7 required, X == don't care
//   immFmt,    // The format for the immediate value
//   useImm,    // What to use for the denom op of the divide
//   usePo2,    // Use po2 of low 5 of v2
//   signed,    // Treat inputs as signed
//   oneComp,   // Do signed numbers in one's complement
//   rem)       // Return the remainder (instead of the qoutient)
// We start at a multiple of 8 to allow skipping the check on upper ID
//
// clang-format off
OPD( 40, DIV,   0b01100, 4,  1, R, 0, 0, 1, 0, 0)
OPD( 41, DIVU,  0b01100, 5,  1, R, 0, 0, 0, 0, 0)
OPD( 42, REM,   0b01100, 6,  1, R, 0, 0, 1, 0, 1)
OPD( 43, REMU,  0b01100, 7,  1, R, 0, 0, 0, 0, 1)
OPD( 44, SRL,   0b01100, 5,  0, R, 0, 1, 0, 0, 0)
OPD( 45, SRA,   0b01100, 5, 32, R, 0, 1, 1, 1, 0)
OPD( 46, SRLI,  0b00100, 5,  0, I, 1, 1, 0, 0, 0)
OPD( 47, SRAI,  0b00100, 5, 32, I, 1, 1, 1, 1, 0)
// clang-format on

#undef X
#undef OPC
#undef OPI
#undef OPM
#undef OPD
