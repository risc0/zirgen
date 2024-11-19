// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

use std::ops;

use derive_more::{Add, AddAssign, Debug, Sub};

use super::{pager::PAGE_WORDS, platform::WORD_SIZE};

#[derive(Add, AddAssign, Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Sub)]
#[debug("{_0:#010x}")]
pub struct ByteAddr(pub u32);

#[derive(Add, AddAssign, Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Sub)]
#[debug("${_0:#010x}")]
pub struct WordAddr(pub u32);

impl ByteAddr {
    pub const fn waddr(self) -> WordAddr {
        WordAddr(self.0 / WORD_SIZE as u32)
    }

    pub const fn is_aligned(&self) -> bool {
        self.0 % WORD_SIZE as u32 == 0
    }

    pub const fn is_null(&self) -> bool {
        self.0 == 0
    }

    pub fn wrapping_add(self, rhs: u32) -> Self {
        Self(self.0.wrapping_add(rhs))
    }

    pub fn subaddr(&self) -> u32 {
        self.0 % WORD_SIZE as u32
    }
}

impl WordAddr {
    pub const fn baddr(self) -> ByteAddr {
        ByteAddr(self.0 * WORD_SIZE as u32)
    }

    pub fn page_idx(&self) -> u32 {
        self.0 / PAGE_WORDS as u32
    }

    pub fn page_subaddr(&self) -> WordAddr {
        Self(self.0 % PAGE_WORDS as u32)
    }

    pub fn postfix_inc(&mut self) -> Self {
        let cur = *self;
        self.0 += 1;
        cur
    }
}

impl ops::Add<usize> for WordAddr {
    type Output = WordAddr;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs as u32)
    }
}

impl ops::Add<u32> for WordAddr {
    type Output = WordAddr;

    fn add(self, rhs: u32) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl ops::Sub<u32> for WordAddr {
    type Output = WordAddr;

    fn sub(self, rhs: u32) -> Self::Output {
        Self(self.0 - rhs)
    }
}

impl ops::AddAssign<usize> for WordAddr {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs as u32;
    }
}

impl ops::AddAssign<u32> for WordAddr {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs;
    }
}

impl ops::Add<usize> for ByteAddr {
    type Output = ByteAddr;

    fn add(self, rhs: usize) -> Self::Output {
        Self(self.0 + rhs as u32)
    }
}

impl ops::Add<u32> for ByteAddr {
    type Output = ByteAddr;

    fn add(self, rhs: u32) -> Self::Output {
        Self(self.0 + rhs)
    }
}

impl ops::AddAssign<usize> for ByteAddr {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs as u32;
    }
}

impl ops::AddAssign<u32> for ByteAddr {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs;
    }
}

impl From<ByteAddr> for WordAddr {
    fn from(addr: ByteAddr) -> Self {
        addr.waddr()
    }
}

impl From<WordAddr> for ByteAddr {
    fn from(addr: WordAddr) -> Self {
        addr.baddr()
    }
}
