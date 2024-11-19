// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

use anyhow::Result;

/// A host-side implementation of a system call.
pub trait Syscall {
    /// Reads from the host.
    fn host_read(&self, fd: u32, buf: &mut [u8]) -> Result<u32>;

    /// Writes to the host.
    fn host_write(&self, fd: u32, buf: &[u8]) -> Result<u32>;
}
