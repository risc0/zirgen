// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

use anyhow::{bail, Result};
use risc0_zkp::core::digest::DIGEST_WORDS;
use risc0_zkp::core::hash::poseidon2::{
    CELLS, M_INT_DIAG_HZN, ROUNDS_HALF_FULL, ROUNDS_PARTIAL, ROUND_CONSTANTS,
};
use risc0_zkp::field::baby_bear;

use super::{
    addr::WordAddr,
    node_idx, node_idx_to_addr,
    pager::{PAGE_WORDS, POSEIDON_PAGE_ROUNDS},
    platform::*,
    r0vm::Risc0Context,
};

const P2_HAS_STATE: usize = 0;
const P2_STATE_ADDR: usize = 1;
const P2_BUF_OUT_ADDR: usize = 2;
const P2_IS_ELEM: usize = 3;
const P2_CHECK_OUT: usize = 4;
const P2_LOAD_TX_TYPE: usize = 5;
const P2_NEXT_STATE: usize = 6;
const P2_SUB_STATE: usize = 7;
const P2_BUF_IN_ADDR: usize = 8;
const P2_COUNT: usize = 9;
const P2_MODE: usize = 10;
const P2_CELLS: usize = 11;

const BABY_BEAR_P_U32: u32 = baby_bear::P;
const BABY_BEAR_P_U64: u64 = baby_bear::P as u64;

pub(crate) type RawArray = [u32; P2_CELLS + CELLS];

// one-to-one state from inst_p2
struct RawState(RawArray);

impl RawState {
    fn new() -> Self {
        Self([0; P2_CELLS + CELLS])
    }

    fn has_state(&mut self) -> &mut u32 {
        &mut self.0[P2_HAS_STATE]
    }

    fn state_addr(&mut self) -> &mut u32 {
        &mut self.0[P2_STATE_ADDR]
    }

    fn buf_out_addr(&mut self) -> &mut u32 {
        &mut self.0[P2_BUF_OUT_ADDR]
    }

    fn is_elem(&mut self) -> &mut u32 {
        &mut self.0[P2_IS_ELEM]
    }

    fn check_out(&mut self) -> &mut u32 {
        &mut self.0[P2_CHECK_OUT]
    }

    fn load_tx_type(&mut self) -> &mut u32 {
        &mut self.0[P2_LOAD_TX_TYPE]
    }

    fn next_state(&mut self) -> &mut u32 {
        &mut self.0[P2_NEXT_STATE]
    }

    fn sub_state(&mut self) -> &mut u32 {
        &mut self.0[P2_SUB_STATE]
    }

    fn buf_in_addr(&mut self) -> &mut u32 {
        &mut self.0[P2_BUF_IN_ADDR]
    }

    fn count(&mut self) -> &mut u32 {
        &mut self.0[P2_COUNT]
    }

    fn mode(&mut self) -> &mut u32 {
        &mut self.0[P2_MODE]
    }

    fn cell(&mut self, i: usize) -> &mut u32 {
        &mut self.0[P2_CELLS + i]
    }
}

pub struct Poseidon2State(RawState);

impl Poseidon2State {
    fn new_start(mode: u32) -> Self {
        let mut p2 = RawState::new();
        *p2.buf_out_addr() = if mode == 0 {
            MERKLE_TREE_END_ADDR.0
        } else {
            MERKLE_TREE_START_ADDR.0
        };
        *p2.is_elem() = 1;
        *p2.check_out() = 1;
        *p2.mode() = mode;
        *p2.load_tx_type() = tx::PAGE_IN;
        *p2.next_state() = STATE_POSEIDON_PAGING;
        Self(p2)
    }

    fn new_done(buf_out_addr: u32, next_state: u32, mode: u32) -> Self {
        let mut p2 = RawState::new();
        *p2.buf_out_addr() = buf_out_addr;
        *p2.next_state() = next_state;
        *p2.mode() = mode;
        Self(p2)
    }

    fn new_node(node_idx: u32, is_read: bool) -> Self {
        let mut p2 = RawState::new();
        *p2.buf_out_addr() = node_idx_to_addr(node_idx).0;
        *p2.is_elem() = 1;
        *p2.check_out() = if is_read { 1 } else { 0 };
        *p2.load_tx_type() = if is_read { tx::PAGE_IN } else { tx::PAGE_OUT };
        *p2.next_state() = STATE_POSEIDON_PAGING;
        *p2.buf_in_addr() = node_idx_to_addr(2 * node_idx + 1).0;
        *p2.count() = 1;
        *p2.mode() = if is_read { 0 } else { 4 };
        Self(p2)
    }

    fn new_page(page_idx: u32, is_read: bool) -> Self {
        let node_idx = node_idx(page_idx);
        let mut p2 = RawState::new();
        *p2.buf_out_addr() = node_idx_to_addr(node_idx).0;
        *p2.check_out() = if is_read { 1 } else { 0 };
        *p2.load_tx_type() = if is_read { tx::PAGE_IN } else { tx::PAGE_OUT };
        *p2.next_state() = STATE_POSEIDON_PAGING;
        *p2.buf_in_addr() = page_idx * PAGE_WORDS as u32;
        *p2.count() = POSEIDON_PAGE_ROUNDS;
        *p2.mode() = if is_read { 1 } else { 3 };
        Self(p2)
    }

    fn new_ecall(state_addr: u32, buf_in_addr: u32, buf_out_addr: u32, bits_count: u32) -> Self {
        let is_elem = bits_count & PFLAG_IS_ELEM;
        let check_out = bits_count & PFLAG_CHECK_OUT;
        let mut p2 = RawState::new();
        *p2.state_addr() = state_addr;
        *p2.buf_in_addr() = buf_in_addr;
        *p2.buf_out_addr() = buf_out_addr;
        *p2.has_state() = if state_addr == 0 { 0 } else { 1 };
        *p2.is_elem() = if is_elem == 0 { 0 } else { 1 };
        *p2.check_out() = if check_out == 0 { 0 } else { 1 };
        *p2.count() = bits_count & 0xffff;
        *p2.mode() = 1;
        *p2.load_tx_type() = tx::READ;
        *p2.next_state() = STATE_POSEIDON_ENTRY;
        Self(p2)
    }

    pub(crate) fn as_slice(&self) -> &RawArray {
        &self.0 .0
    }

    pub(crate) fn next_state(&self) -> u32 {
        self.0 .0[P2_NEXT_STATE]
    }

    fn step(
        &mut self,
        ctx: &mut dyn Risc0Context,
        cur_state: &mut u32,
        next_state: u32,
        sub_state: u32,
    ) {
        *self.0.next_state() = next_state;
        *self.0.sub_state() = sub_state;
        ctx.on_poseidon2_cycle(*cur_state, self);
        *cur_state = next_state;
    }

    fn rest(&mut self, ctx: &mut dyn Risc0Context, final_state: u32) -> Result<()> {
        let mut cur_state = *self.0.next_state();
        let state_addr = WordAddr(*self.0.state_addr());

        // If we have state, load it
        if *self.0.has_state() == 1 {
            // tracing::trace!("has_state");
            self.step(ctx, &mut cur_state, STATE_POSEIDON_LOAD_STATE, 0);
            for i in 0..DIGEST_WORDS as usize {
                *self.0.cell(DIGEST_WORDS * 2 + i) = ctx.load_u32(state_addr + i)?;
            }
        }

        // While we have data to process
        let mut buf_in_addr = WordAddr(*self.0.buf_in_addr());
        // tracing::debug!("buf_in_addr: {buf_in_addr:?}");
        while *self.0.count() > 0 {
            // Do load
            self.step(ctx, &mut cur_state, STATE_POSEIDON_LOAD_IN, 0);
            if *self.0.is_elem() != 0 {
                for i in 0..DIGEST_WORDS {
                    *self.0.cell(i) = ctx.load_u32(buf_in_addr.postfix_inc())?;
                }
                *self.0.buf_in_addr() = buf_in_addr.0;
                self.step(ctx, &mut cur_state, STATE_POSEIDON_LOAD_IN, 1);
                for i in 0..DIGEST_WORDS {
                    *self.0.cell(DIGEST_WORDS + i) = ctx.load_u32(buf_in_addr.postfix_inc())?;
                }
                *self.0.buf_in_addr() = buf_in_addr.0;
            } else {
                for i in 0..DIGEST_WORDS {
                    let word = ctx.load_u32(buf_in_addr.postfix_inc())?;
                    *self.0.cell(2 * i + 0) = word & 0xffff;
                    *self.0.cell(2 * i + 1) = word >> 16;
                }
                *self.0.buf_in_addr() = buf_in_addr.0;
            }
            // Do the mix
            self.multiply_by_m_ext();
            for i in 0..ROUNDS_HALF_FULL {
                self.step(ctx, &mut cur_state, STATE_POSEIDON_EXT_ROUND, i as u32);
                self.do_ext_round(i);
            }
            self.step(ctx, &mut cur_state, STATE_POSEIDON_INT_ROUND, 0);
            self.do_int_rounds();
            for i in ROUNDS_HALF_FULL..ROUNDS_HALF_FULL * 2 {
                self.step(ctx, &mut cur_state, STATE_POSEIDON_EXT_ROUND, i as u32);
                self.do_ext_round(i);
            }
            *self.0.count() -= 1;
        }

        self.step(ctx, &mut cur_state, STATE_POSEIDON_DO_OUT, 0);

        let buf_out_addr = WordAddr(*self.0.buf_out_addr());
        if *self.0.check_out() != 0 {
            for i in 0..DIGEST_WORDS {
                let addr = buf_out_addr + i;
                let word = ctx.load_u32(addr)?;
                let cell = *self.0.cell(i);
                if word != cell {
                    tracing::debug!(
                        "buf_in_addr: {:?}, buf_out_addr: {buf_out_addr:?}, cell: {i}",
                        WordAddr(*self.0.buf_in_addr())
                    );
                    bail!("poseidon2 check failed: {word:#010x} != {cell:#010x}");
                }
            }
        } else {
            for i in 0..DIGEST_WORDS {
                ctx.store_u32(buf_out_addr + i, *self.0.cell(i))?;
            }
        }

        *self.0.buf_in_addr() = 0;

        if *self.0.has_state() == 1 {
            self.step(ctx, &mut cur_state, STATE_POSEIDON_STORE_STATE, 0);
            for i in 0..DIGEST_WORDS {
                ctx.store_u32(state_addr + i, *self.0.cell(DIGEST_WORDS * 2 + i))?;
            }
        }

        self.step(ctx, &mut cur_state, final_state, 0);

        Ok(())
    }

    // Optimized method for multiplication by M_EXT.
    // See appendix B of Poseidon2 paper for additional details.
    fn multiply_by_m_ext(&mut self) {
        let mut out = [0; CELLS];
        let mut tmp_sums = [0; 4];

        for i in 0..CELLS / 4 {
            let chunk = multiply_by_4x4_circulant(&[
                *self.0.cell(i * 4),
                *self.0.cell(i * 4 + 1),
                *self.0.cell(i * 4 + 2),
                *self.0.cell(i * 4 + 3),
            ]);
            for j in 0..4 {
                let to_add = chunk[j] as u64;
                let to_add = (to_add % BABY_BEAR_P_U64) as u32;
                tmp_sums[j] += to_add;
                tmp_sums[j] %= BABY_BEAR_P_U32;
                out[i * 4 + j] += to_add;
                out[i * 4 + j] %= BABY_BEAR_P_U32;
            }
        }
        for i in 0..CELLS {
            *self.0.cell(i) = (out[i] + tmp_sums[i % 4]) % BABY_BEAR_P_U32;
        }
    }

    // Exploit the fact that off-diagonal entries of M_INT are all 1.
    fn multiply_by_m_int(&mut self) {
        let mut sum = 0u64;
        for i in 0..CELLS {
            sum += *self.0.cell(i) as u64;
        }
        sum %= BABY_BEAR_P_U64;
        for i in 0..CELLS {
            let diag = M_INT_DIAG_HZN[i].as_u32() as u64;
            let cell = *self.0.cell(i) as u64;
            *self.0.cell(i) = ((sum + diag * cell) % BABY_BEAR_P_U64) as u32;
        }
    }

    fn do_ext_round(&mut self, mut idx: usize) {
        if idx >= ROUNDS_HALF_FULL {
            idx += ROUNDS_PARTIAL;
        }

        self.add_round_constants_full(idx);
        for i in 0..CELLS {
            *self.0.cell(i) = sbox2(*self.0.cell(i));
        }

        self.multiply_by_m_ext();
    }

    fn do_int_rounds(&mut self) {
        for i in 0..ROUNDS_PARTIAL {
            self.add_round_constants_partial(ROUNDS_HALF_FULL + i);
            *self.0.cell(0) = sbox2(*self.0.cell(0));
            self.multiply_by_m_int();
        }
    }

    fn add_round_constants_full(&mut self, round: usize) {
        for i in 0..CELLS {
            *self.0.cell(i) += ROUND_CONSTANTS[round * CELLS + i].as_u32();
            *self.0.cell(i) %= BABY_BEAR_P_U32;
        }
    }

    fn add_round_constants_partial(&mut self, round: usize) {
        *self.0.cell(0) += ROUND_CONSTANTS[round * CELLS].as_u32();
        *self.0.cell(0) %= BABY_BEAR_P_U32;
    }
}

fn multiply_by_4x4_circulant(x: &[u32; 4]) -> [u32; 4] {
    // See appendix B of Poseidon2 paper.
    const CIRC_FACTOR_2: u64 = 2;
    const CIRC_FACTOR_4: u64 = 4;
    let t0 = (x[0] as u64 + x[1] as u64) % BABY_BEAR_P_U64;
    let t1 = (x[2] as u64 + x[3] as u64) % BABY_BEAR_P_U64;
    let t2 = (CIRC_FACTOR_2 * x[1] as u64 + t1) % BABY_BEAR_P_U64;
    let t3 = (CIRC_FACTOR_2 * x[3] as u64 + t0) % BABY_BEAR_P_U64;
    let t4 = (CIRC_FACTOR_4 * t1 + t3) % BABY_BEAR_P_U64;
    let t5 = (CIRC_FACTOR_4 * t0 + t2) % BABY_BEAR_P_U64;
    let t6 = (t3 + t5) % BABY_BEAR_P_U64;
    let t7 = (t2 + t4) % BABY_BEAR_P_U64;
    [t6 as u32, t5 as u32, t7 as u32, t4 as u32]
}

fn sbox2(x: u32) -> u32 {
    let x = x as u64;
    let x2 = (x * x) % BABY_BEAR_P_U64;
    let x4 = (x2 * x2) % BABY_BEAR_P_U64;
    let x6 = (x4 * x2) % BABY_BEAR_P_U64;
    let x7 = (x6 * x) % BABY_BEAR_P_U64;
    x7 as u32
}

pub fn read_start(ctx: &mut dyn Risc0Context) -> Result<()> {
    tracing::trace!("read_start");
    let p2 = Poseidon2State::new_start(0);
    ctx.on_poseidon2_cycle(STATE_POSEIDON_ENTRY, &p2);
    Ok(())
}

pub fn read_node(ctx: &mut dyn Risc0Context, node_idx: u32) -> Result<()> {
    // tracing::trace!("read_node: {node_idx:#010x}");
    let mut p2 = Poseidon2State::new_node(node_idx, true);
    p2.rest(ctx, STATE_POSEIDON_PAGING)
}

pub fn read_page(ctx: &mut dyn Risc0Context, page_idx: u32) -> Result<()> {
    // tracing::trace!("read_page: {page_idx:#010x}");
    let mut p2 = Poseidon2State::new_page(page_idx, true);
    p2.rest(ctx, STATE_POSEIDON_PAGING)
}

pub fn read_done(ctx: &mut dyn Risc0Context) -> Result<()> {
    tracing::trace!("read_done");
    let p2 = Poseidon2State::new_done(MERKLE_TREE_START_ADDR.0, STATE_RESUME, 2);
    ctx.on_poseidon2_cycle(STATE_POSEIDON_PAGING, &p2);
    Ok(())
}

pub fn write_start(ctx: &mut dyn Risc0Context) -> Result<()> {
    tracing::trace!("write_start");
    let p2 = Poseidon2State::new_start(3);
    ctx.on_poseidon2_cycle(STATE_POSEIDON_ENTRY, &p2);
    Ok(())
}

pub fn write_node(ctx: &mut dyn Risc0Context, node_idx: u32) -> Result<()> {
    // tracing::trace!("write_node: {node_idx:#010x}");
    let mut p2 = Poseidon2State::new_node(node_idx, false);
    p2.rest(ctx, STATE_POSEIDON_PAGING)
}

pub fn write_page(ctx: &mut dyn Risc0Context, page_idx: u32) -> Result<()> {
    // tracing::trace!("write_page: {page_idx:#010x}");
    let mut p2 = Poseidon2State::new_page(page_idx, false);
    p2.rest(ctx, STATE_POSEIDON_PAGING)
}

pub fn write_done(ctx: &mut dyn Risc0Context) -> Result<()> {
    tracing::trace!("write_done");
    let p2 = Poseidon2State::new_done(MERKLE_TREE_END_ADDR.0, STATE_STORE_ROOT, 5);
    ctx.on_poseidon2_cycle(STATE_POSEIDON_PAGING, &p2);
    Ok(())
}

pub fn ecall(ctx: &mut dyn Risc0Context) -> Result<()> {
    tracing::trace!("ecall");
    let state_addr = ctx.load_u32(MACHINE_REGS_ADDR.waddr() + REG_A0)?;
    let buf_in_addr = ctx.load_u32(MACHINE_REGS_ADDR.waddr() + REG_A1)?;
    let buf_out_addr = ctx.load_u32(MACHINE_REGS_ADDR.waddr() + REG_A2)?;
    let bits_count = ctx.load_u32(MACHINE_REGS_ADDR.waddr() + REG_A3)?;
    let mut p2 = Poseidon2State::new_ecall(state_addr, buf_in_addr, buf_out_addr, bits_count);
    p2.rest(ctx, STATE_DECODE)
}
