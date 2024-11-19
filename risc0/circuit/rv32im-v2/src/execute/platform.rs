// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

#![allow(unused)]

use super::addr::{ByteAddr, WordAddr};

pub const WORD_SIZE: usize = 4;
pub const PAGE_BYTES: usize = 1024;
pub const MEMORY_BYTES: usize = 1 << 32; // TODO: this only works on 64-bit machines
pub const MEMORY_PAGES: usize = MEMORY_BYTES / PAGE_BYTES;
pub const MERKLE_TREE_DEPTH: usize = MEMORY_PAGES.ilog2() as usize;
pub const LOOKUP_TABLE_CYCLES: usize = ((1 << 8) + (1 << 16)) / 16;

pub const ZERO_PAGE_START_ADDR: ByteAddr = ByteAddr(0x0000_0000);
pub const ZERO_PAGE_END_ADDR: ByteAddr = ByteAddr(0x0001_0000);
pub const USER_START_ADDR: ByteAddr = ZERO_PAGE_END_ADDR;
pub const USER_END_ADDR: ByteAddr = ByteAddr(0xc000_0000);
pub const KERNEL_START_ADDR: ByteAddr = USER_END_ADDR;
pub const KERNEL_END_ADDR: ByteAddr = ByteAddr(0xff00_0000);
pub const MACHINE_REGS_ADDR: ByteAddr = ByteAddr(0xffff_0000);
pub const USER_REGS_ADDR: ByteAddr = ByteAddr(0xffff_0080);
pub const SAFE_WRITE_ADDR: ByteAddr = ByteAddr(0xffff_0100);
pub const MEPC_ADDR: ByteAddr = ByteAddr(0xffff_0200);
pub const SUSPEND_PC_ADDR: ByteAddr = ByteAddr(0xffff_0210);
pub const SUSPEND_MODE_ADDR: ByteAddr = ByteAddr(0xffff_0214);
pub const SUSPEND_CYCLE_LOW_ADDR: ByteAddr = ByteAddr(0xffff_0218);
pub const SUSPEND_CYCLE_HIGH_ADDR: ByteAddr = ByteAddr(0xffff_021c);
pub const GLOBAL_OUTPUT_ADDR: ByteAddr = ByteAddr(0xffff_0240);
pub const GLOBAL_INPUT_ADDR: ByteAddr = ByteAddr(0xffff_0260);

pub const ECALL_DISPATCH_ADDR: ByteAddr = ByteAddr(0xffff_1000);
pub const TRAP_DISPATCH_ADDR: ByteAddr = ByteAddr(0xffff_2000);

pub const MEMORY_END_ADDR: WordAddr = WordAddr(0x4000_0000);
pub const MERKLE_TREE_START_ADDR: WordAddr = WordAddr(0x4000_0000);
pub const MERKLE_TREE_END_ADDR: WordAddr = WordAddr(0x4400_0000);

pub const REG_ZERO: usize = 0; // zero constant
pub const REG_RA: usize = 1; // return address
pub const REG_SP: usize = 2; // stack pointer
pub const REG_GP: usize = 3; // global pointer
pub const REG_TP: usize = 4; // thread pointer
pub const REG_T0: usize = 5; // temporary
pub const REG_T1: usize = 6; // temporary
pub const REG_T2: usize = 7; // temporary
pub const REG_S0: usize = 8; // saved register
pub const REG_FP: usize = 8; // frame pointer
pub const REG_S1: usize = 9; // saved register
pub const REG_A0: usize = 10; // fn arg / return value
pub const REG_A1: usize = 11; // fn arg / return value
pub const REG_A2: usize = 12; // fn arg
pub const REG_A3: usize = 13; // fn arg
pub const REG_A4: usize = 14; // fn arg
pub const REG_A5: usize = 15; // fn arg
pub const REG_A6: usize = 16; // fn arg
pub const REG_A7: usize = 17; // fn arg
pub const REG_S2: usize = 18; // saved register
pub const REG_S3: usize = 19; // saved register
pub const REG_S4: usize = 20; // saved register
pub const REG_S5: usize = 21; // saved register
pub const REG_S6: usize = 22; // saved register
pub const REG_S7: usize = 23; // saved register
pub const REG_S8: usize = 24; // saved register
pub const REG_S9: usize = 25; // saved register
pub const REG_S10: usize = 26; // saved register
pub const REG_S11: usize = 27; // saved register
pub const REG_T3: usize = 28; // temporary
pub const REG_T4: usize = 29; // temporary
pub const REG_T5: usize = 30; // temporary
pub const REG_T6: usize = 31; // temporary
pub const REG_MAX: usize = 32; // maximum number of registers

pub const HOST_ECALL_TERMINATE: u32 = 0;
pub const HOST_ECALL_READ: u32 = 1;
pub const HOST_ECALL_WRITE: u32 = 2;
pub const HOST_ECALL_POSEIDON2: u32 = 3;

pub const PFLAG_IS_ELEM: u32 = 0x8000_0000;
pub const PFLAG_CHECK_OUT: u32 = 0x4000_0000;

pub const STATE_LOAD_ROOT: u32 = 0;
pub const STATE_RESUME: u32 = 1;
pub const STATE_SUSPEND: u32 = 4;
pub const STATE_STORE_ROOT: u32 = 5;
pub const STATE_CONTROL_TABLE: u32 = 6;
pub const STATE_CONTROL_DONE: u32 = 7;

pub const STATE_MACHINE_ECALL: u32 = 8;
pub const STATE_TERMINATE: u32 = 9;
pub const STATE_HOST_READ_SETUP: u32 = 10;
pub const STATE_HOST_WRITE: u32 = 11;
pub const STATE_HOST_READ_BYTES: u32 = 12;
pub const STATE_HOST_READ_WORDS: u32 = 13;

pub const STATE_POSEIDON_ENTRY: u32 = 16;
pub const STATE_POSEIDON_LOAD_STATE: u32 = 17;
pub const STATE_POSEIDON_LOAD_IN: u32 = 18;
pub const STATE_POSEIDON_DO_OUT: u32 = 21;
pub const STATE_POSEIDON_PAGING: u32 = 22;
pub const STATE_POSEIDON_STORE_STATE: u32 = 23;

pub const STATE_POSEIDON_EXT_ROUND: u32 = 24;
pub const STATE_POSEIDON_INT_ROUND: u32 = 25;

pub const STATE_DECODE: u32 = 32;

pub const SYSCALL_MAX: u32 = 512;

pub const MAX_IO_BYTES: u32 = 1024;
pub const MAX_IO_WORDS: u32 = 4;

/// Returns whether `addr` is within user memory bounds.
pub fn is_user_memory(addr: ByteAddr) -> bool {
    addr >= USER_START_ADDR && addr < USER_END_ADDR
}

/// Returns whether `addr` is within user memory bounds.
pub fn is_kernel_memory(addr: ByteAddr) -> bool {
    addr >= KERNEL_START_ADDR && addr < KERNEL_END_ADDR
}

pub mod major {
    pub const MISC0: u8 = 0;
    pub const MISC1: u8 = 1;
    pub const MISC2: u8 = 2;
    pub const MUL0: u8 = 3;
    pub const DIV0: u8 = 4;
    pub const MEM0: u8 = 5;
    pub const MEM1: u8 = 6;
    pub const CONTROL0: u8 = 7;
    pub const ECALL0: u8 = 8;
    pub const POSEIDON0: u8 = 9;
    pub const POSEIDON1: u8 = 10;
}

pub mod control_minor {
    pub const RESUME: u8 = 1;
    pub const USER_ECALL: u8 = 2;
    pub const MRET: u8 = 3;
}

pub mod ecall_minor {
    pub const MACHINE_ECALL: u8 = 0;
    pub const TERMINATE: u8 = 1;
    pub const HOST_READ_SETUP: u8 = 2;
    pub const HOST_WRITE: u8 = 3;
    pub const HOST_READ_BYTES: u8 = 4;
    pub const HOST_READ_WORDS: u8 = 5;
}

pub mod poseidon_minor {
    pub const LOAD_STATE: u8 = 0;
    pub const LOAD_DATA: u8 = 1;
    pub const EXT_ROUND: u8 = 2;
    pub const INT_ROUNDS: u8 = 3;
    pub const STORE_STATE: u8 = 4;
}

pub mod tx {
    pub const READ: u32 = 0;
    pub const PAGE_IN: u32 = 1;
    pub const PAGE_OUT: u32 = 2;
}
