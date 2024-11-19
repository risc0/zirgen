// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

use std::collections::BTreeMap;

use anyhow::{anyhow, bail, Result};
use derive_more::Debug;
use risc0_circuit_rv32im_v2_sys::{RawMemoryTransaction, RawPreflightCycle};
use risc0_core::scope;
use risc0_zkp::core::digest::DIGEST_WORDS;

use crate::execute::{
    addr::{ByteAddr, WordAddr},
    node_idx_to_addr,
    pager::PagedMemory,
    platform::*,
    poseidon2::{self, Poseidon2State},
    r0vm::{Risc0Context, Risc0Machine},
    rv32im::{DecodedInstruction, Emulator, InsnKind, Instruction},
    segment::Segment,
};

#[derive(Clone, Debug, Default)]
pub struct PreflightTrace {
    #[debug("{}", cycles.len())]
    pub cycles: Vec<RawPreflightCycle>,
    #[debug("{}", txns.len())]
    pub txns: Vec<RawMemoryTransaction>,
    #[debug("{}", extras.len())]
    pub extras: Vec<u32>,
    pub table_split_cycle: u32,
}

struct Preflight<'a> {
    pub trace: PreflightTrace,
    segment: &'a Segment,
    pager: PagedMemory,
    pc: ByteAddr,
    machine_mode: u32,
    cur_write: usize,
    cur_read: usize,
    user_cycle: u32,
    txn_idx: u32,
    extra_idx: u32,
    phys_cycles: u32,
    orig_words: BTreeMap<WordAddr, u32>,
    prev_cycle: BTreeMap<WordAddr, u32>,
    page_memory: BTreeMap<WordAddr, u32>,
}

impl Segment {
    pub fn preflight(&self) -> Result<PreflightTrace> {
        scope!("preflight");
        tracing::debug!("preflight: {self:#?}");

        let mut preflight = Preflight::new(self);
        preflight.read_pages()?;
        preflight.body()?;
        preflight.write_pages()?;
        preflight.generate_tables()?;
        preflight.wrap_memory_txns()?;

        tracing::trace!("paging_cycles: {}", preflight.pager.cycles);

        Ok(preflight.trace)
    }
}

fn get_digest_addr(idx: u32) -> WordAddr {
    MERKLE_TREE_START_ADDR + DIGEST_WORDS as u32 * (2 * MEMORY_PAGES as u32 - idx)
}

// macro_rules! track_cycles {
//     ($self:ident, $tag:expr, $body:block) => {
//         let __start = $self.trace.cycles.len();
//         $body
//         let __end = $self.trace.cycles.len();
//         tracing::trace!("{}: {}", format!($tag), __end - __start);
//     };
// }

impl<'a> Preflight<'a> {
    fn new(segment: &'a Segment) -> Self {
        tracing::debug!("po2: {}", segment.po2);

        let mut page_memory = BTreeMap::new();
        for (&node_idx, digest) in segment.partial_image.digests.iter() {
            let node_addr = node_idx_to_addr(node_idx);
            for i in 0..DIGEST_WORDS {
                page_memory.insert(node_addr + i, digest.as_words()[i]);
            }
        }
        Self {
            trace: PreflightTrace::default(),
            segment,
            pager: PagedMemory::new(segment.partial_image.clone()),
            pc: ByteAddr(0),
            machine_mode: 0,
            cur_write: 0,
            cur_read: 0,
            txn_idx: 0,
            user_cycle: 0,
            extra_idx: 0,
            phys_cycles: 0,
            orig_words: BTreeMap::new(),
            prev_cycle: BTreeMap::new(),
            page_memory,
        }
    }

    // Do page in
    pub fn read_pages(&mut self) -> Result<()> {
        self.read_root()?;
        let activity = self.pager.loaded_pages();
        poseidon2::read_start(self)?;
        for node_idx in activity.nodes {
            // track_cycles!(self, "read_node: {node_idx:#010x}", {
            poseidon2::read_node(self, node_idx)?;
            // });
        }
        self.machine_mode = 1;
        for page_idx in activity.pages {
            // track_cycles!(self, "read_page: {page_idx:#010x}", {
            poseidon2::read_page(self, page_idx)?;
            // });
        }
        self.machine_mode = 2;
        poseidon2::read_done(self)?;
        self.phys_cycles = 0;
        Ok(())
    }

    // Run main execution
    pub fn body(&mut self) -> Result<()> {
        let mut emu = Emulator::new();
        Risc0Machine::resume(self)?;
        while self.phys_cycles < self.segment.suspend_cycle {
            Risc0Machine::step(&mut emu, self)?;
        }
        Risc0Machine::suspend(self)
    }

    // Do page out
    pub fn write_pages(&mut self) -> Result<()> {
        let activity = self.pager.dirty_pages();
        self.pager.commit()?;
        poseidon2::write_start(self)?;
        for &page_idx in activity.pages.iter().rev() {
            // track_cycles!(self, "write_page", {
            poseidon2::write_page(self, page_idx)?;
            // });
        }
        self.machine_mode = 4;
        for &node_idx in activity.nodes.iter().rev() {
            // track_cycles!(self, "write_node", {
            poseidon2::write_node(self, node_idx)?;
            // });
        }
        self.machine_mode = 5;
        poseidon2::write_done(self)?;
        self.machine_mode = 0;
        self.write_root()
    }

    // Do table reification
    pub fn generate_tables(&mut self) -> Result<()> {
        self.trace.table_split_cycle = self.trace.cycles.len() as u32;
        self.fini();
        Ok(())
    }

    // Now, go back and update memory transactions to wrap around
    fn wrap_memory_txns(&mut self) -> Result<()> {
        for txn in self.trace.txns.iter_mut() {
            // tracing::trace!("{txn:?}");
            let addr = WordAddr(txn.addr);
            if txn.prev_cycle == u32::MAX {
                // If first cycle for a particular address, set 'prev_cycle' to final cycle
                txn.prev_cycle = self.prev_cycle[&addr];
            } else {
                // Otherwise, compute cycle diff and another diff
                let diff = txn.cycle - txn.prev_cycle;
                self.trace.cycles[diff as usize].diff_count += 1;
            }

            // If last cycle, set final value to original value
            if txn.cycle == self.prev_cycle[&addr] {
                txn.word = self.orig_words[&addr];
            }
        }
        Ok(())
    }

    fn fini(&mut self) {
        for i in (16..256).step_by(16) {
            self.add_cycle_special(STATE_CONTROL_TABLE, STATE_CONTROL_TABLE, i);
        }
        self.machine_mode = 1;
        for i in (0..64 * 1024).step_by(16) {
            self.add_cycle_special(STATE_CONTROL_TABLE, STATE_CONTROL_TABLE, i);
        }
        self.machine_mode = 0;
        self.add_cycle_special(STATE_CONTROL_TABLE, STATE_CONTROL_DONE, 0);
        self.add_cycle_special(STATE_CONTROL_DONE, STATE_CONTROL_DONE, 0);
    }

    fn read_root(&mut self) -> Result<()> {
        let addr = get_digest_addr(1);
        for i in 0..DIGEST_WORDS {
            self.load_u32(addr + i)?;
        }
        self.add_cycle_special(STATE_LOAD_ROOT, STATE_POSEIDON_ENTRY, 0);
        Ok(())
    }

    fn write_root(&mut self) -> Result<()> {
        let addr = get_digest_addr(1);
        for i in 0..DIGEST_WORDS {
            self.load_u32(addr + i)?;
        }
        self.add_cycle_special(STATE_STORE_ROOT, STATE_CONTROL_TABLE, 0);
        Ok(())
    }

    fn add_cycle(&mut self, state: u32, pc: u32, major: u8, minor: u8) {
        let cycle = RawPreflightCycle {
            state,
            pc,
            major,
            minor,
            machine_mode: self.machine_mode as u8,
            padding: 0,
            user_cycle: self.user_cycle,
            txn_idx: self.txn_idx,
            extra_idx: self.extra_idx,
            diff_count: 0,
        };
        // tracing::trace!("{cycle:?}");
        self.trace.cycles.push(cycle);
        self.txn_idx = self.trace.txns.len() as u32;
        self.extra_idx = self.trace.extras.len() as u32;
    }

    fn add_cycle_insn(&mut self, state: u32, pc: u32, insn: InsnKind) {
        match insn {
            InsnKind::Eany => {
                // Technically we need to switch on the machine mode *entering* the EANY
                if self.trace.cycles.last().unwrap().machine_mode != 0 {
                    self.add_cycle(state, pc, major::ECALL0, ecall_minor::MACHINE_ECALL);
                } else {
                    self.add_cycle(state, pc, major::CONTROL0, control_minor::USER_ECALL);
                }
            }
            InsnKind::Mret => {
                self.add_cycle(state, pc, major::CONTROL0, control_minor::MRET);
            }
            _ => {
                self.add_cycle(state, pc, insn.major(), insn.minor());
            }
        }
    }

    fn add_cycle_special(&mut self, cur_state: u32, next_state: u32, pc: u32) {
        let major = (7 + cur_state / 8) as u8;
        let minor = (cur_state % 8) as u8;
        // tracing::trace!("add_cycle_special(cur_state: {cur_state}, next_state: {next_state}, major: {major}, minor: {minor})");
        self.add_cycle(next_state, pc, major, minor);
    }
}

impl<'a> Risc0Context for Preflight<'a> {
    fn get_pc(&self) -> ByteAddr {
        self.pc
    }

    fn set_pc(&mut self, addr: ByteAddr) {
        self.pc = addr;
    }

    fn get_machine_mode(&self) -> u32 {
        self.machine_mode
    }

    fn set_machine_mode(&mut self, mode: u32) {
        self.machine_mode = mode;
    }

    fn resume(&mut self) -> Result<()> {
        self.add_cycle_special(STATE_RESUME, STATE_RESUME, self.pc.0);
        for i in 0..DIGEST_WORDS {
            self.store_u32(GLOBAL_INPUT_ADDR.waddr() + i, 0)?; // FIXME!
        }
        self.add_cycle_special(STATE_RESUME, STATE_DECODE, self.pc.0);
        Ok(())
    }

    fn suspend(&mut self) -> Result<()> {
        self.pc = ByteAddr(0);
        self.add_cycle_special(STATE_SUSPEND, STATE_SUSPEND, 0);
        for i in 0..DIGEST_WORDS {
            self.load_u32(GLOBAL_OUTPUT_ADDR.waddr() + i)?;
        }
        self.machine_mode = 3;
        self.add_cycle_special(STATE_SUSPEND, STATE_POSEIDON_ENTRY, 0);
        Ok(())
    }

    fn on_insn_start(&mut self, _insn: &Instruction, _decoded: &DecodedInstruction) -> Result<()> {
        Ok(())
    }

    fn on_insn_end(&mut self, insn: &Instruction, _decoded: &DecodedInstruction) -> Result<()> {
        self.add_cycle_insn(STATE_DECODE, self.pc.0, insn.kind);
        self.user_cycle += 1;
        self.phys_cycles += 1;
        Ok(())
    }

    fn trap_rewind(&mut self) {
        self.trace.txns.truncate(self.txn_idx as usize);
        self.trace.extras.truncate(self.extra_idx as usize);
    }

    fn peek_u32(&mut self, _addr: WordAddr) -> Result<u32> {
        // no-op is OK
        Ok(0)
    }

    // Pass memory ops to pager + record
    fn load_u32(&mut self, addr: WordAddr) -> Result<u32> {
        let cycle = self.trace.cycles.len();
        let word = if addr >= MERKLE_TREE_START_ADDR {
            *self
                .page_memory
                .get(&addr)
                .ok_or(anyhow!("Invalid load from page memory"))?
        } else {
            self.pager.load(addr)?
        };
        self.orig_words.entry(addr).or_insert(word);
        let prev_cycle = *self.prev_cycle.get(&addr).unwrap_or(&u32::MAX);
        let txn = RawMemoryTransaction {
            addr: addr.0,
            cycle: cycle as u32,
            word,
            prev_cycle,
            prev_word: word,
        };
        self.prev_cycle.insert(addr, txn.cycle);
        self.trace.txns.push(txn);
        Ok(word)
    }

    fn store_u32(&mut self, addr: WordAddr, word: u32) -> Result<()> {
        let cycle = self.trace.cycles.len();
        let prev_word = if addr >= MEMORY_END_ADDR {
            let prev_word = *self
                .page_memory
                .get(&addr)
                .ok_or(anyhow!("Invalid store to page memory"))?;
            self.page_memory.insert(addr, word);
            prev_word
        } else {
            let prev_word = self.pager.load(addr)?;
            self.pager.store(addr, word)?;
            prev_word
        };
        let prev_cycle = *self.prev_cycle.get(&addr).unwrap_or(&u32::MAX);
        let txn = RawMemoryTransaction {
            addr: addr.0,
            cycle: cycle as u32,
            word,
            prev_cycle,
            prev_word,
        };
        self.prev_cycle.insert(addr, txn.cycle);
        self.trace.txns.push(txn);
        Ok(())
    }

    fn on_ecall_cycle(&mut self, cur_state: u32, next_state: u32, s0: u32, s1: u32, s2: u32) {
        self.trace.extras.extend([s0, s1, s2]);
        self.add_cycle_special(cur_state, next_state, self.pc.0);
        self.phys_cycles += 1;
    }

    fn on_poseidon2_cycle(&mut self, cur_state: u32, p2: &Poseidon2State) {
        self.trace.extras.extend_from_slice(p2.as_slice());
        self.add_cycle_special(cur_state, p2.next_state(), self.pc.0);
        self.phys_cycles += 1;
    }

    fn on_terminate(&mut self, _a0: u32, _a1: u32) {
        // no-op
    }

    fn host_read(&mut self, _fd: u32, buf: &mut [u8]) -> Result<u32> {
        if self.cur_read >= self.segment.read_record.len() {
            bail!("Invalid segment: unexpected read record");
        }
        let record = &self.segment.read_record[self.cur_read];
        let rlen = record.len();
        if rlen > buf.len() {
            bail!("Invalid segment: truncated read record");
        }
        buf[..rlen].copy_from_slice(record);
        Ok(rlen as u32)
    }

    fn host_write(&mut self, _fd: u32, _buf: &[u8]) -> Result<u32> {
        if self.cur_write >= self.segment.write_record.len() {
            bail!("Invalid segment: unexpected write record");
        }
        self.cur_write += 1;
        Ok(self.segment.write_record[self.cur_write])
    }
}
