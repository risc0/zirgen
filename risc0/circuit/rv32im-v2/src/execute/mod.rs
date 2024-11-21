// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

pub(crate) mod addr;
mod executor;
pub(crate) mod image;
pub(crate) mod pager;
pub(crate) mod platform;
pub(crate) mod r0vm;
pub(crate) mod rv32im;
pub(crate) mod segment;
mod syscall;
#[cfg(test)]
mod tests;
pub mod testutil;
mod trace;

use self::platform::MEMORY_PAGES;

pub use self::executor::{Executor, ExecutorResult, SimpleSession};

pub const DEFAULT_SEGMENT_LIMIT_PO2: usize = 20;

pub(crate) fn node_idx(page_idx: u32) -> u32 {
    MEMORY_PAGES as u32 + page_idx
}
