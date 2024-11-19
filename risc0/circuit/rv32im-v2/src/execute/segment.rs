// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

use derive_more::Debug;
use risc0_binfmt::ExitCode;
use risc0_zkp::core::digest::Digest;
use serde::{Deserialize, Serialize};

use super::image::MemoryImage2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Segment {
    /// Initial sparse memory state for the segment
    pub partial_image: MemoryImage2,

    pub pre_digest: Digest,

    pub post_digest: Digest,

    /// Recorded host->guest IO, one entry per read
    #[debug(skip)]
    pub read_record: Vec<Vec<u8>>,

    /// Recorded rlen of guest->host IO, one entry per write
    #[debug(skip)]
    pub write_record: Vec<u32>,

    pub user_cycles: u32,

    /// Cycle at which we suspend
    pub suspend_cycle: u32,

    /// Total paging cycles
    pub paging_cycles: u32,

    pub po2: u32,

    pub exit_code: ExitCode,

    pub index: u64,

    pub input_digest: Digest,

    pub output_digest: Option<Digest>,
}
