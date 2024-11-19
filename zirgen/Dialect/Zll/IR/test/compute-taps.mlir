// RUN: zirgen-opt --compute-taps %s | FileCheck %s

module attributes {zll.buffers = #zll<buffers ("code", <16, constant>, 1), ("out", <1, global>), ("data", <16, mutable>, 2), ("mix", <1, global>), ("accum", <1, mutable>, 0)>, zll.protocolInfo = #zll<protocol_info "TEST____________">, zll.steps = #zll<steps "exec", "ram_verify", "verify_bytes", "compute_accum", "verify_accum">} {
// CHECK: zll.taps = #zll<taps
// CHECK-SAME: <0, 0, 0>,
// CHECK-SAME: <1, 0, 0>,
// CHECK-SAME: <1, 1, 0>,
// CHECK-SAME: <1, 2, 0>,
// CHECK-SAME: <1, 3, 5>,
// CHECK-SAME: <1, 4, 0>,
// CHECK-SAME: <1, 5, 0>,
// CHECK-SAME: <1, 6, 0>,
// CHECK-SAME: <1, 7, 0>,
// CHECK-SAME: <1, 8, 0>,
// CHECK-SAME: <1, 9, 0>,
// CHECK-SAME: <1, 10, 0>,
// CHECK-SAME: <1, 11, 0>,
// CHECK-SAME: <1, 12, 0>,
// CHECK-SAME: <1, 13, 0>,
// CHECK-SAME: <1, 14, 0>,
// CHECK-SAME: <1, 15, 0>,
// CHECK-SAME: <2, 0, 0>,
// CHECK-SAME: <2, 1, 0>,
// CHECK-SAME: <2, 1, 2>,
// CHECK-SAME: <2, 2, 0>,
// CHECK-SAME: <2, 3, 0>,
// CHECK-SAME: <2, 4, 0>,
// CHECK-SAME: <2, 5, 0>,
// CHECK-SAME: <2, 6, 0>,
// CHECK-SAME: <2, 7, 0>,
// CHECK-SAME: <2, 8, 0>,
// CHECK-SAME: <2, 9, 0>,
// CHECK-SAME: <2, 10, 0>,
// CHECK-SAME: <2, 11, 0>,
// CHECK-SAME: <2, 12, 0>,
// CHECK-SAME: <2, 13, 0>,
// CHECK-SAME: <2, 14, 0>,
// CHECK-SAME: <2, 15, 0>>

func.func @get(%cbuf : !zll.buffer<16, constant> {zirgen.argName = "code"},
               %unused : !zll.val<BabyBear>,
               %mbuf : !zll.buffer<16, mutable> {zirgen.argName = "data"})
               -> !zll.val<BabyBear> {
   // CHECK-LABEL: func @get
   // CHECK: zll.get
   %0 = zll.get %cbuf[3] back 5 : <16, constant>
   %1 = zll.get %mbuf[1] back 2 : <16, mutable>
   %2 = zll.get %mbuf[1] back 0 : <16, mutable>
   return %2 : !zll.val<BabyBear>
}

}
