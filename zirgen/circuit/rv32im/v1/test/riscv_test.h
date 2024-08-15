// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#define __riscv_xlen 32
#define TESTNUM x31

#define RVTEST_RV32U                                                                               \
  .macro init;                                                                                     \
  .endm

#define RVTEST_FAIL                                                                                \
  fence;                                                                                           \
  unimp
#define RVTEST_PASS                                                                                \
  li t0, 0;                                                                                        \
  li a0, 0;                                                                                        \
  li a1, 0x400;                                                                                    \
  ecall

#define RVTEST_CODE_BEGIN                                                                          \
  .text;                                                                                           \
  .globl _start;                                                                                   \
  _start:                                                                                          \
  .option push;                                                                                    \
  .option norelax;                                                                                 \
  la gp, __global_pointer$;                                                                        \
  .option pop;

#define RVTEST_CODE_END

#define RVTEST_DATA_BEGIN .data
#define RVTEST_DATA_END
