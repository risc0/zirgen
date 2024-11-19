// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

pub(crate) mod addr;
mod executor;
pub(crate) mod image;
pub(crate) mod pager;
pub(crate) mod platform;
pub(crate) mod poseidon2;
pub(crate) mod r0vm;
pub(crate) mod rv32im;
pub(crate) mod segment;
mod syscall;
#[cfg(test)]
mod tests;
pub mod testutil;
mod trace;

use risc0_zkp::core::digest::DIGEST_WORDS;

use self::{
    addr::WordAddr,
    platform::{MEMORY_PAGES, MERKLE_TREE_END_ADDR},
};

pub use self::executor::{Executor, ExecutorResult, SimpleSession};

pub const DEFAULT_SEGMENT_LIMIT_PO2: usize = 20;

fn node_idx(page_idx: u32) -> u32 {
    MEMORY_PAGES as u32 + page_idx
}

pub(crate) fn node_idx_to_addr(idx: u32) -> WordAddr {
    MERKLE_TREE_END_ADDR - idx * DIGEST_WORDS as u32
}

fn node_addr_to_idx(addr: WordAddr) -> u32 {
    (MERKLE_TREE_END_ADDR - addr).0 / DIGEST_WORDS as u32
}
