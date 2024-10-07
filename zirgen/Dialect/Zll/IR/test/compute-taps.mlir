// RUN: zirgen-opt --compute-taps %s | FileCheck %s

module attributes {zll.buffers = #zll<buffers ("code", <16, constant>, 1), ("out", <1, global>), ("data", <16, mutable>, 2), ("mix", <1, global>), ("accum", <1, mutable>, 0)>, zll.protocolInfo = #zll<protocol_info "TEST____________">, zll.steps = #zll<steps "exec", "ram_verify", "verify_bytes", "compute_accum", "verify_accum">} {

func.func @get(%cbuf : !zll.buffer<16, constant> {zirgen.argName = "code"},
               %unused : !zll.val<BabyBear>,
               %mbuf : !zll.buffer<16, mutable> {zirgen.argName = "data"})
               -> !zll.val<BabyBear> {
   // CHECK-LABEL: func @get
   // CHECK-DAG: taps = [#zll.tap<1, 3, 5>, #zll.tap<2, 1, 0>, #zll.tap<2, 1, 2>]
   // CHECK-DAG: tapRegs = [#zll.tapReg<1, 3, [5], 1>, #zll.tapReg<2, 1, [0, 2], 0>]
   // CHECK-DAG{LITERAL}: tapCombos = [[0 : ui32, 2 : ui32], [5 : ui32]]
   // CHECK-DAG: tapType = !zll.val<BabyBear>,
   // CHECK-NEXT: zll.get
   %0 = zll.get %cbuf[3] back 5 : <16, constant>
   %1 = zll.get %mbuf[1] back 2 : <16, mutable>
   %2 = zll.get %mbuf[1] back 0 : <16, mutable>
   return %2 : !zll.val<BabyBear>
}

}
