// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

use anyhow::Result;
use derive_more::Debug;
use serde::{Deserialize, Serialize};

/// An event traced from the running VM.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum TraceEvent {
    /// An instruction has started at the given program counter
    InstructionStart {
        /// Cycle number since startup
        cycle: u64,

        /// Program counter of the instruction being executed
        #[debug("{pc:#010x}")]
        pc: u32,

        /// Encoded instruction being executed.
        #[debug("{pc:#010x}")]
        insn: u32,
    },

    /// A register has been set
    RegisterSet {
        /// Register ID (0-16)
        idx: usize,

        /// New value in the register
        #[debug("{value:#010x}")]
        value: u32,
    },

    /// A memory location has been written
    MemorySet {
        /// Address of memory that's been written
        #[debug("{addr:#010x}")]
        addr: u32,

        /// Data that's been written
        #[debug("{region:#04x?}")]
        region: Vec<u8>,
    },
}

/// A callback used to collect [TraceEvent]s.
pub trait TraceCallback {
    fn trace_callback(&mut self, event: TraceEvent) -> Result<()>;
}

impl<F: FnMut(TraceEvent) -> Result<()>> TraceCallback for F {
    fn trace_callback(&mut self, event: TraceEvent) -> Result<()> {
        self(event)
    }
}
