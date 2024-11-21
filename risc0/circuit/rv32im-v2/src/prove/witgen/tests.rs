// Copyright (c) 2024 RISC Zero, Inc.
//
// All rights reserved.

use rand::thread_rng;
use risc0_zkp::field::{baby_bear::BabyBearExtElem, Elem as _};

use crate::execute::{
    image::MemoryImage2,
    testutil::{self, NullSyscall, DEFAULT_SESSION_LIMIT},
    DEFAULT_SEGMENT_LIMIT_PO2,
};

#[test]
fn basic() {
    let program = testutil::basic();
    let image = MemoryImage2::new(program);

    let result = testutil::execute(
        image,
        DEFAULT_SEGMENT_LIMIT_PO2,
        DEFAULT_SESSION_LIMIT,
        &NullSyscall,
        None,
    )
    .unwrap();
    let segments = result.segments;
    let segment = segments.first().unwrap();

    let mut rng = thread_rng();
    let nonce = BabyBearExtElem::random(&mut rng);

    let mut trace = segment.preflight(nonce).unwrap();
    // let expected_cycles = [
    //     add_cycle(InsnKind::LUI, 0, Some(0x4000)),
    //     add_cycle(InsnKind::LUI, 3, Some(0x4004)),
    //     add_cycle(InsnKind::ADD, 6, Some(0x4008)),
    //     add_cycle(InsnKind::LUI, 9, Some(0x400c)),
    //     add_cycle(InsnKind::EANY, 12, None),
    // ];
    // trace.body.cycles.truncate(expected_cycles.len());

    // assert_slice_eq(&trace.body.cycles, &expected_cycles);
    // assert_slice_eq(
    //     &trace.body.txns,
    //     &[
    //         MemoryTransaction::new(0, ByteAddr(0x00004000), 0x1234b137),
    //         MemoryTransaction::new(0, ByteAddr(0x0c000024), 0),
    //         MemoryTransaction::new(0, ByteAddr(0x0c00000c), 0),
    //         MemoryTransaction::new(1, ByteAddr(0x00004004), 0xf387e1b7),
    //         MemoryTransaction::new(1, ByteAddr(0x0c00003c), 0),
    //         MemoryTransaction::new(1, ByteAddr(0x0c000060), 0),
    //         MemoryTransaction::new(2, ByteAddr(0x00004008), 0x003100b3),
    //         MemoryTransaction::new(2, ByteAddr(0x0c000008), 0x1234b000),
    //         MemoryTransaction::new(2, ByteAddr(0x0c00000c), 0xf387e000),
    //         MemoryTransaction::new(3, ByteAddr(0x0000400c), 0x000045b7),
    //         MemoryTransaction::new(3, ByteAddr(0x0c000000), 0),
    //         MemoryTransaction::new(3, ByteAddr(0x0c000000), 0),
    //         MemoryTransaction::new(4, ByteAddr(0x00004010), 0x00000073),
    //         MemoryTransaction::new(4, ByteAddr(0x0c000014), 0),
    //         MemoryTransaction::new(4, ByteAddr(0x0c00002c), 0x00004000),
    //         MemoryTransaction::new(4, ByteAddr(0x0c000028), 0x00000000),
    //         // reset(1)
    //         MemoryTransaction::new(4380, ByteAddr(0x00004000), 0x1234b137),
    //         MemoryTransaction::new(4380, ByteAddr(0x00004004), 0xf387e1b7),
    //         MemoryTransaction::new(4380, ByteAddr(0x00004008), 0x003100b3),
    //         MemoryTransaction::new(4380, ByteAddr(0x0000400c), 0x000045b7),
    //         MemoryTransaction::new(4381, ByteAddr(0x00004010), 0x00000073),
    //         MemoryTransaction::new(4381, ByteAddr(0x00004014), 0x00000000),
    //         MemoryTransaction::new(4381, ByteAddr(0x00004018), 0x00000000),
    //         MemoryTransaction::new(4381, ByteAddr(0x0000401c), 0x00000000),
    //         // reset(2)
    //         MemoryTransaction::new(4382, ByteAddr(0x0d6b5ac0), 0x2ea10cf3),
    //         MemoryTransaction::new(4382, ByteAddr(0x0d6b5ac4), 0x41559d09),
    //         MemoryTransaction::new(4382, ByteAddr(0x0d6b5ac8), 0x032b0b9e),
    //         MemoryTransaction::new(4382, ByteAddr(0x0d6b5acc), 0xda56a7af),
    //         MemoryTransaction::new(4383, ByteAddr(0x0d6b5ad0), 0x7c9d8024),
    //         MemoryTransaction::new(4383, ByteAddr(0x0d6b5ad4), 0x9bfea1c1),
    //         MemoryTransaction::new(4383, ByteAddr(0x0d6b5ad8), 0xc37b44c3),
    //         MemoryTransaction::new(4383, ByteAddr(0x0d6b5adc), 0x554f49f5),
    //     ],
    // );

    // assert_eq!(trace.body.extras.len(), 0);
}
